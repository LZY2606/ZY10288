//! Growth transactions: reserve, rehash and resize of the table.
//!
//! Every growth operation follows the same transaction shape:
//!
//! 1. [`GrowthTransaction::begin`] allocates the replacement table *first*.
//!    If allocation fails, the error is returned and the original table is
//!    left completely untouched.
//! 2. [`GrowthTransaction::migrate`] copies the elements into the new table.
//!    If the hash function panics here, the transaction is rolled back by
//!    its `Drop` impl, which releases the new table's allocation *without*
//!    dropping any elements (migration only copies bytes; the original table
//!    still owns every element until commit).
//! 3. [`GrowthTransaction::commit`] swaps the new table into place and
//!    releases the old allocation.
//!
//! The in-place rehash ([`RawTableInner::rehash_in_place`]) does not allocate;
//! it instead uses a scope guard that reclaims the elements that have not
//! been rehashed yet when the hash function panics.
use crate::TryReserveError;
use crate::alloc::Allocator;
use crate::control::{Group, Tag};
use crate::scopeguard::guard;
use crate::util::{likely, unlikely};
use core::mem;
use core::ptr;

use super::storage::{Fallibility, RawTableInner, TableLayout, bucket_mask_to_capacity};

impl RawTableInner {
    /// Prepares for rehashing data in place (that is, without allocating new memory).
    /// Converts all full index `control bytes` to `Tag::DELETED` and all `Tag::DELETED` control
    /// bytes to `Tag::EMPTY`, i.e. performs the following conversion:
    ///
    /// - `Tag::EMPTY` control bytes   -> `Tag::EMPTY`;
    /// - `Tag::DELETED` control bytes -> `Tag::EMPTY`;
    /// - `FULL` control bytes    -> `Tag::DELETED`.
    ///
    /// This function does not make any changes to the `data` parts of the table,
    /// or any changes to the `items` or `growth_left` field of the table.
    ///
    /// # Safety
    ///
    /// You must observe the following safety rules when calling this function:
    ///
    /// * The [`RawTableInner`] has already been allocated;
    ///
    /// * The caller of this function must convert the `Tag::DELETED` bytes back to `FULL`
    ///   bytes when re-inserting them into their ideal position (which was impossible
    ///   to do during the first insert due to tombstones). If the caller does not do
    ///   this, then calling this function may result in a memory leak.
    ///
    /// * The [`RawTableInner`] must have properly initialized control bytes otherwise
    ///   calling this function results in [`undefined behavior`].
    ///
    /// Calling this function on a table that has not been allocated results in
    /// [`undefined behavior`].
    ///
    /// See also [`Bucket::as_ptr`] method, for more information about of properly removing
    /// or saving `data element` from / into the [`RawTable`] / [`RawTableInner`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn prepare_rehash_in_place(&mut self) {
        // Bulk convert all full control bytes to DELETED, and all DELETED control bytes to EMPTY.
        // This effectively frees up all buckets containing a DELETED entry.
        //
        // SAFETY:
        // 1. `i` is guaranteed to be within bounds since we are iterating from zero to `buckets - 1`;
        // 2. Even if `i` will be `i == self.bucket_mask`, it is safe to call `Group::load_aligned`
        //    due to the extended control bytes range, which is `self.bucket_mask + 1 + Group::WIDTH`;
        // 3. The caller of this function guarantees that [`RawTableInner`] has already been allocated;
        // 4. We can use `Group::load_aligned` and `Group::store_aligned` here since we start from 0
        //    and go to the end with a step equal to `Group::WIDTH` (see TableLayout::calculate_layout_for).
        unsafe {
            for i in (0..self.num_buckets()).step_by(Group::WIDTH) {
                let group = Group::load_aligned(self.ctrl(i));
                let group = group.convert_special_to_empty_and_full_to_deleted();
                group.store_aligned(self.ctrl(i));
            }
        }

        // Fix up the trailing control bytes. See the comments in set_ctrl
        // for the handling of tables smaller than the group width.
        if unlikely(self.num_buckets() < Group::WIDTH) {
            // SAFETY: We have `self.bucket_mask + 1 + Group::WIDTH` number of control bytes,
            // so copying `self.num_buckets() == self.bucket_mask + 1` bytes with offset equal to
            // `Group::WIDTH` is safe
            unsafe {
                self.ctrl(0)
                    .copy_to(self.ctrl(Group::WIDTH), self.num_buckets());
            }
        } else {
            // SAFETY: We have `self.bucket_mask + 1 + Group::WIDTH` number of
            // control bytes,so copying `Group::WIDTH` bytes with offset equal
            // to `self.num_buckets() == self.bucket_mask + 1` is safe
            unsafe {
                self.ctrl(0)
                    .copy_to(self.ctrl(self.num_buckets()), Group::WIDTH);
            }
        }
    }

    /// Reserves or rehashes to make room for `additional` more elements.
    ///
    /// This uses dynamic dispatch to reduce the amount of
    /// code generated, but it is eliminated by LLVM optimizations when inlined.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is
    /// [`undefined behavior`]:
    ///
    /// * The `alloc` must be the same [`Allocator`] as the `Allocator` used
    ///   to allocate this table.
    ///
    /// * The `layout` must be the same [`TableLayout`] as the `TableLayout`
    ///   used to allocate this table.
    ///
    /// * The `drop` function (`fn(*mut u8)`) must be the actual drop function of
    ///   the elements stored in the table.
    ///
    /// * The [`RawTableInner`] must have properly initialized control bytes.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub(super) unsafe fn reserve_rehash_inner<A>(
        &mut self,
        alloc: &A,
        additional: usize,
        hasher: &dyn Fn(&mut Self, usize) -> u64,
        fallibility: Fallibility,
        layout: TableLayout,
        drop: Option<unsafe fn(*mut u8)>,
    ) -> Result<(), TryReserveError>
    where
        A: Allocator,
    {
        // Avoid `Option::ok_or_else` because it bloats LLVM IR.
        let Some(new_items) = self.items.checked_add(additional) else {
            return Err(fallibility.capacity_overflow());
        };
        let full_capacity = bucket_mask_to_capacity(self.bucket_mask);
        if new_items <= full_capacity / 2 {
            // Rehash in-place without re-allocating if we have plenty of spare
            // capacity that is locked up due to DELETED entries.

            // SAFETY:
            // 1. We know for sure that `[`RawTableInner`]` has already been allocated
            //    (since new_items <= full_capacity / 2);
            // 2. The caller ensures that `drop` function is the actual drop function of
            //    the elements stored in the table.
            // 3. The caller ensures that `layout` matches the [`TableLayout`] that was
            //    used to allocate this table.
            // 4. The caller ensures that the control bytes of the `RawTableInner`
            //    are already initialized.
            unsafe {
                self.rehash_in_place(hasher, layout.size, drop);
            }
            Ok(())
        } else {
            // Otherwise, conservatively resize to at least the next size up
            // to avoid churning deletes into frequent rehashes.
            //
            // SAFETY:
            // 1. We know for sure that `capacity >= self.items`.
            // 2. The caller ensures that `alloc` and `layout` matches the [`Allocator`] and
            //    [`TableLayout`] that were used to allocate this table.
            // 3. The caller ensures that the control bytes of the `RawTableInner`
            //    are already initialized.
            unsafe {
                self.resize_inner(
                    alloc,
                    usize::max(new_items, full_capacity + 1),
                    hasher,
                    fallibility,
                    layout,
                )
            }
        }
    }

    /// Allocates a new table of a different size and moves the contents of the
    /// current table into it.
    ///
    /// This uses dynamic dispatch to reduce the amount of
    /// code generated, but it is eliminated by LLVM optimizations when inlined.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is
    /// [`undefined behavior`]:
    ///
    /// * The `alloc` must be the same [`Allocator`] as the `Allocator` used
    ///   to allocate this table;
    ///
    /// * The `layout` must be the same [`TableLayout`] as the `TableLayout`
    ///   used to allocate this table;
    ///
    /// * The [`RawTableInner`] must have properly initialized control bytes.
    ///
    /// The caller of this function must ensure that `capacity >= self.items`
    /// otherwise:
    ///
    /// * If `self.items != 0`, calling of this function with `capacity == 0`
    ///   results in [`undefined behavior`].
    ///
    /// * If `capacity_to_buckets(capacity) < Group::WIDTH` and
    ///   `self.items > capacity_to_buckets(capacity)` calling this function
    ///   results in [`undefined behavior`].
    ///
    /// * If `capacity_to_buckets(capacity) >= Group::WIDTH` and
    ///   `self.items > capacity_to_buckets(capacity)` calling this function
    ///   are never return (will go into an infinite loop).
    ///
    /// Note: It is recommended (but not required) that the new table's `capacity`
    /// be greater than or equal to `self.items`. In case if `capacity <= self.items`
    /// this function can never return. See [`RawTableInner::find_insert_index`] for
    /// more information.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[expect(clippy::inline_always)]
    #[inline(always)]
    pub(super) unsafe fn resize_inner<A>(
        &mut self,
        alloc: &A,
        capacity: usize,
        hasher: &dyn Fn(&mut Self, usize) -> u64,
        fallibility: Fallibility,
        layout: TableLayout,
    ) -> Result<(), TryReserveError>
    where
        A: Allocator,
    {
        debug_assert!(self.items <= capacity);

        // Allocate the replacement table first. If the allocation fails, the
        // error is returned and the original table is left completely
        // untouched.
        //
        // SAFETY: We know for sure that `alloc` and `layout` matches the [`Allocator`] and [`TableLayout`]
        // that were used to allocate this table.
        let mut transaction = GrowthTransaction::begin(alloc, layout, capacity, fallibility)?;

        // SAFETY: We know for sure that RawTableInner will outlive the
        // returned `FullBucketsIndices` iterator, and the caller of this
        // function ensures that the control bytes are properly initialized.
        //
        // If the hash function panics, the transaction is rolled back by its
        // `Drop` impl: the new table's allocation is released and the
        // original table is left untouched.
        unsafe {
            transaction.migrate(self, hasher, layout);
        }

        // The hash function didn't panic, so we can safely commit the
        // transaction: the new table is swapped into place and the old
        // allocation is released (without dropping the elements, since they
        // have been moved into the new table).
        //
        // SAFETY: The caller ensures that `table_layout` matches the [`TableLayout`]
        // that was used to allocate this table.
        transaction.commit(self);

        Ok(())
    }

    /// Rehashes the contents of the table in place (i.e. without changing the
    /// allocation).
    ///
    /// If `hasher` panics then some the table's contents may be lost.
    ///
    /// This uses dynamic dispatch to reduce the amount of
    /// code generated, but it is eliminated by LLVM optimizations when inlined.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is [`undefined behavior`]:
    ///
    /// * The `size_of` must be equal to the size of the elements stored in the table;
    ///
    /// * The `drop` function (`fn(*mut u8)`) must be the actual drop function of
    ///   the elements stored in the table.
    ///
    /// * The [`RawTableInner`] has already been allocated;
    ///
    /// * The [`RawTableInner`] must have properly initialized control bytes.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[cfg_attr(feature = "inline-more", expect(clippy::inline_always))]
    #[cfg_attr(feature = "inline-more", inline(always))]
    #[cfg_attr(not(feature = "inline-more"), inline)]
    pub(super) unsafe fn rehash_in_place(
        &mut self,
        hasher: &dyn Fn(&mut Self, usize) -> u64,
        size_of: usize,
        drop: Option<unsafe fn(*mut u8)>,
    ) {
        // If the hash function panics then properly clean up any elements
        // that we haven't rehashed yet. We unfortunately can't preserve the
        // element since we lost their hash and have no way of recovering it
        // without risking another panic.
        unsafe {
            self.prepare_rehash_in_place();
        }

        let mut guard = guard(self, move |self_| {
            for i in 0..self_.num_buckets() {
                unsafe {
                    // Any elements that haven't been rehashed yet have a
                    // DELETED tag. These need to be dropped and have their tag
                    // reset to EMPTY.
                    if *self_.ctrl(i) == Tag::DELETED {
                        self_.set_ctrl(i, Tag::EMPTY);
                        if let Some(drop) = drop {
                            drop(self_.bucket_ptr(i, size_of));
                        }
                        self_.items -= 1;
                    }
                }
            }
            self_.growth_left = bucket_mask_to_capacity(self_.bucket_mask) - self_.items;
        });

        // At this point, DELETED elements are elements that we haven't
        // rehashed yet. Find them and re-insert them at their ideal
        // position.
        'outer: for i in 0..guard.num_buckets() {
            unsafe {
                if *guard.ctrl(i) != Tag::DELETED {
                    continue;
                }
            }

            let i_p = unsafe { guard.bucket_ptr(i, size_of) };

            loop {
                // Hash the current item
                let hash = hasher(*guard, i);

                // Search for a suitable place to put it
                //
                // SAFETY: Caller of this function ensures that the control bytes
                // are properly initialized.
                let new_i = unsafe { guard.find_insert_index(hash) };

                // Probing works by scanning through all of the control
                // bytes in groups, which may not be aligned to the group
                // size. If both the new and old position fall within the
                // same unaligned group, then there is no benefit in moving
                // it and we can just continue to the next item.
                if likely(guard.is_in_same_group(i, new_i, hash)) {
                    unsafe { guard.set_ctrl_hash(i, hash) };
                    continue 'outer;
                }

                let new_i_p = unsafe { guard.bucket_ptr(new_i, size_of) };

                // We are moving the current item to a new position. Write
                // our H2 to the control byte of the new position.
                let prev_ctrl = unsafe { guard.replace_ctrl_hash(new_i, hash) };
                if prev_ctrl == Tag::EMPTY {
                    unsafe { guard.set_ctrl(i, Tag::EMPTY) };
                    // If the target slot is empty, simply move the current
                    // element into the new slot and clear the old control
                    // byte.
                    unsafe {
                        ptr::copy_nonoverlapping(i_p, new_i_p, size_of);
                    }
                    continue 'outer;
                }

                // If the target slot is occupied, swap the two elements
                // and then continue processing the element that we just
                // swapped into the old slot.
                debug_assert_eq!(prev_ctrl, Tag::DELETED);
                unsafe {
                    ptr::swap_nonoverlapping(i_p, new_i_p, size_of);
                }
            }
        }

        guard.growth_left = bucket_mask_to_capacity(guard.bucket_mask) - guard.items;

        mem::forget(guard);
    }
}

/// A growth transaction: allocates the replacement table up front, migrates
/// the elements into it, and then either commits (swapping the new table into
/// place and releasing the old allocation) or rolls back.
///
/// Rollback happens automatically on panic: the `Drop` impl releases the new
/// table's allocation *without* dropping any elements. This is sound because
/// migration only copies bytes out of the original table, which keeps owning
/// every element until [`GrowthTransaction::commit`] runs.
///
/// If the allocation in [`GrowthTransaction::begin`] fails, the error is
/// returned and the original table is left completely untouched.
pub(super) struct GrowthTransaction<'a, A: Allocator> {
    // The freshly allocated table the elements are migrated into. Its
    // `growth_left` and `items` fields are only set on `commit`.
    new_table: RawTableInner,
    // The allocator of the original table, used to release the allocation
    // held by `new_table` on rollback or after `commit`.
    alloc: &'a A,
    // The `TableLayout` of the original table.
    table_layout: TableLayout,
}

impl<'a, A: Allocator> GrowthTransaction<'a, A> {
    /// Attempts to allocate a new hash table with at least enough capacity
    /// for inserting the given number of elements without reallocating,
    /// and returns a transaction owning it.
    ///
    /// # Note
    ///
    /// It is recommended (but not required):
    ///
    /// * That the new table's `capacity` be greater than or equal to the
    ///   number of elements in the original table.
    ///
    /// * The `alloc` is the same [`Allocator`] as the `Allocator` used
    ///   to allocate the original table.
    ///
    /// * The `table_layout` is the same [`TableLayout`] as the `TableLayout`
    ///   used to allocate the original table.
    ///
    /// If `table_layout` does not match the `TableLayout` that was used to
    /// allocate the original table, then using `mem::swap` with the original
    /// table and the new table owned by this transaction results in
    /// [`undefined behavior`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) fn begin(
        alloc: &'a A,
        table_layout: TableLayout,
        capacity: usize,
        fallibility: Fallibility,
    ) -> Result<Self, TryReserveError> {
        // Allocate and initialize the new table. On failure the error is
        // returned to the caller and the original table is left untouched.
        let new_table =
            RawTableInner::fallible_with_capacity(alloc, table_layout, capacity, fallibility)?;

        Ok(Self {
            new_table,
            alloc,
            table_layout,
        })
    }

    /// Migrates all elements of `old` into the new table.
    ///
    /// This may panic if the hash function panics, in which case the
    /// transaction is rolled back by its `Drop` impl.
    ///
    /// # Safety
    ///
    /// * The `hasher` must be the hash function of the elements stored in
    ///   `old`;
    ///
    /// * `old` must have properly initialized control bytes and must outlive
    ///   the `FullBucketsIndices` iterator used here;
    ///
    /// * The `layout` must be the same [`TableLayout`] as the `TableLayout`
    ///   used to allocate `old`;
    ///
    /// * The new table must have enough capacity for all elements of `old`
    ///   (see [`RawTableInner::find_insert_index`] for more information).
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) unsafe fn migrate(
        &mut self,
        old: &mut RawTableInner,
        hasher: &dyn Fn(&mut RawTableInner, usize) -> u64,
        layout: TableLayout,
    ) {
        // SAFETY: The caller ensures that `old` outlives the returned
        // `FullBucketsIndices` iterator and that its control bytes are
        // properly initialized.
        unsafe {
            for full_byte_index in old.full_buckets_indices() {
                // This may panic.
                let hash = hasher(old, full_byte_index);

                // SAFETY:
                // We can use a simpler version of insert() here since:
                // 1. There are no DELETED entries.
                // 2. We know there is enough space in the table.
                // 3. All elements are unique.
                // 4. The caller of this function guarantees that the new table
                //    has some allocated memory.
                // 5. We set `growth_left` and `items` fields of the new table
                //    on `commit`.
                // 6. We insert into the table, at the returned index, the data
                //    matching the given hash immediately after calling this function.
                let (new_index, _) = self.new_table.prepare_insert_index(hash);

                // SAFETY:
                //
                // * `src` is valid for reads of `layout.size` bytes, since the
                //   table is alive and the `full_byte_index` is guaranteed to be
                //   within bounds (see `FullBucketsIndices::next_impl`);
                //
                // * `dst` is valid for writes of `layout.size` bytes, since the
                //   caller ensures that `table_layout` matches the [`TableLayout`]
                //   that was used to allocate old table and we have the `new_index`
                //   returned by `prepare_insert_index`.
                //
                // * Both `src` and `dst` are properly aligned.
                //
                // * Both `src` and `dst` point to different region of memory.
                ptr::copy_nonoverlapping(
                    old.bucket_ptr(full_byte_index, layout.size),
                    self.new_table.bucket_ptr(new_index, layout.size),
                    layout.size,
                );
            }
        }
    }

    /// Commits the transaction: the new table is swapped into `old` and the
    /// allocation previously held by `old` is released.
    ///
    /// The elements are not dropped: they have been moved into the new
    /// table. The `items` and `growth_left` fields of the new table are
    /// fixed up before the swap.
    pub(super) fn commit(mut self, old: &mut RawTableInner) {
        // The hash function didn't panic, so we can safely set the
        // `growth_left` and `items` fields of the new table.
        self.new_table.growth_left -= old.items;
        self.new_table.items = old.items;

        // We successfully copied all elements without panicking. Now replace
        // the old table with the new one. Dropping `self` releases the
        // allocation of the *old* table (which is now held by
        // `self.new_table` after the swap) without dropping the elements.
        mem::swap(old, &mut self.new_table);
    }
}

impl<A: Allocator> Drop for GrowthTransaction<'_, A> {
    fn drop(&mut self) {
        // Releases the allocation held by the transaction. Before `commit`
        // (or after a panic during `migrate`) this is the *new* table and
        // the rollback leaves the original table untouched; after `commit`
        // this is the *old* table. In both cases the elements must not be
        // dropped here: they are owned by the original table (rollback) or
        // have been moved into the new table (commit).
        if !self.new_table.is_empty_singleton() {
            // SAFETY:
            // 1. We have checked that the table is allocated.
            // 2. We know for sure that the `alloc` and `table_layout` matches the
            //    [`Allocator`] and [`TableLayout`] used to allocate this table.
            unsafe { self.new_table.free_buckets(self.alloc, self.table_layout) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::Global;
    use crate::raw::RawTable;
    use core::ptr::NonNull;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use stdalloc::sync::Arc;

    /// Counts the number of live instances, so tests can detect double drops
    /// (count goes negative / below expected) and leaks (count does not
    /// return to zero).
    struct DropTag(Arc<AtomicUsize>);

    impl DropTag {
        fn new(live: &Arc<AtomicUsize>) -> Self {
            live.fetch_add(1, Ordering::SeqCst);
            Self(live.clone())
        }
    }

    impl Drop for DropTag {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn panics_silently(f: impl FnOnce()) -> bool {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(stdalloc::boxed::Box::new(|_| {}));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err();
        std::panic::set_hook(previous_hook);
        panicked
    }

    /// A hasher closure state that panics on the Nth call.
    struct PanicOnNthCall {
        calls: AtomicUsize,
        panic_at: usize,
    }

    impl PanicOnNthCall {
        fn new(panic_at: usize) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                panic_at,
            }
        }

        fn check(&self) {
            assert!(
                self.calls.fetch_add(1, Ordering::SeqCst) != self.panic_at,
                "panic-on-Nth-hash (call #{})",
                self.panic_at
            );
        }
    }

    /// Builds a table of 100 `(u64, DropTag)` elements with 60 tombstones.
    fn build_table_with_tombstones(live: &Arc<AtomicUsize>) -> RawTable<(u64, DropTag)> {
        let mut table = RawTable::new();
        for i in 0..100u64 {
            table.insert(i, (i, DropTag::new(live)), |(k, _)| *k);
        }
        for i in 0..60u64 {
            let removed = table.remove_entry(i, |(k, _)| *k == i);
            assert_eq!(removed.map(|(k, _)| k), Some(i));
        }
        assert_eq!(table.len(), 40);
        table
    }

    /// panic-on-Nth-hash: if the hash function panics while the growth
    /// transaction migrates elements into the new table, the transaction
    /// must roll back and leave the original table completely untouched.
    #[test]
    #[cfg(panic = "unwind")]
    fn resize_hasher_panic_rolls_back_to_original_table() {
        for panic_at in 0..100 {
            let live = Arc::new(AtomicUsize::new(0));
            let mut table = RawTable::new();
            for i in 0..100u64 {
                table.insert(i, (i, DropTag::new(&live)), |(k, _)| *k);
            }
            let capacity_before = table.capacity();
            let buckets_before = table.num_buckets();

            let state = PanicOnNthCall::new(panic_at);
            let panicked = panics_silently(|| {
                // Force a resize (not an in-place rehash) whose migration
                // panics on the Nth hash call.
                table.reserve(250, |(k, _)| {
                    state.check();
                    *k
                });
            });
            assert!(panicked, "the migration should panic on call {panic_at}");

            // The original table is preserved: same shape, same elements,
            // and no element was dropped or leaked by the rollback.
            assert_eq!(table.len(), 100);
            assert_eq!(table.capacity(), capacity_before);
            assert_eq!(table.num_buckets(), buckets_before);
            assert_eq!(live.load(Ordering::SeqCst), 100);
            for i in 0..100u64 {
                assert!(table.find(i, |(k, _)| *k == i).is_some());
            }

            drop(table);
            assert_eq!(live.load(Ordering::SeqCst), 0);
        }
    }

    /// panic-on-Nth-hash: if the hash function panics during an in-place
    /// rehash, the rollback guard must reclaim (drop) exactly the elements
    /// that have not been rehashed yet — no double drops, no leaks.
    #[test]
    #[cfg(panic = "unwind")]
    fn rehash_in_place_hasher_panic_reclaims_unmoved_elements() {
        // Count the number of hash calls of a clean rehash run.
        let live = Arc::new(AtomicUsize::new(0));
        let total_calls = {
            let mut table = build_table_with_tombstones(&live);
            let calls = AtomicUsize::new(0);
            unsafe {
                table.table.rehash_in_place(
                    &|table, index| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let (k, _) = table.bucket::<(u64, DropTag)>(index).as_ref();
                        *k
                    },
                    size_of::<(u64, DropTag)>(),
                    Some(|ptr| ptr::drop_in_place(ptr.cast::<(u64, DropTag)>())),
                );
            }
            assert_eq!(table.len(), 40);
            drop(table);
            assert_eq!(live.load(Ordering::SeqCst), 0);
            calls.into_inner()
        };
        assert!(total_calls >= 40);

        for panic_at in 0..total_calls {
            let live = Arc::new(AtomicUsize::new(0));
            let mut table = build_table_with_tombstones(&live);
            let state = PanicOnNthCall::new(panic_at);
            let panicked = panics_silently(|| unsafe {
                table.table.rehash_in_place(
                    &|table, index| {
                        state.check();
                        let (k, _) = table.bucket::<(u64, DropTag)>(index).as_ref();
                        *k
                    },
                    size_of::<(u64, DropTag)>(),
                    Some(|ptr| ptr::drop_in_place(ptr.cast::<(u64, DropTag)>())),
                );
            });
            assert!(panicked, "the rehash should panic on call {panic_at}");

            // The elements that were not rehashed in time have been reclaimed
            // by the rollback guard: the live count matches the table length
            // and every survivor is still findable.
            let remaining = table.len();
            assert_eq!(live.load(Ordering::SeqCst), remaining);
            let mut found = 0;
            for i in 0..100u64 {
                if table.find(i, |(k, _)| *k == i).is_some() {
                    found += 1;
                }
            }
            assert_eq!(found, remaining);

            drop(table);
            assert_eq!(live.load(Ordering::SeqCst), 0);
        }
    }

    /// An allocator that starts failing once the shared allocation counter
    /// reaches the shared limit.
    #[derive(Clone)]
    struct FailAfterN {
        allocs: Arc<AtomicUsize>,
        fail_at: Arc<AtomicUsize>,
    }

    unsafe impl Allocator for FailAfterN {
        fn allocate(
            &self,
            layout: core::alloc::Layout,
        ) -> Result<NonNull<[u8]>, crate::alloc::AllocError> {
            if self.allocs.fetch_add(1, Ordering::SeqCst) >= self.fail_at.load(Ordering::SeqCst) {
                return Err(crate::alloc::AllocError);
            }
            Global.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: core::alloc::Layout) {
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    /// panic-on-Nth-alloc: if the allocation of the replacement table fails,
    /// the growth transaction never starts and the original table is left
    /// completely untouched.
    #[test]
    fn alloc_failure_preserves_original_table() {
        let allocs = Arc::new(AtomicUsize::new(0));
        let fail_at = Arc::new(AtomicUsize::new(usize::MAX));
        let alloc = FailAfterN {
            allocs: allocs.clone(),
            fail_at: fail_at.clone(),
        };

        let mut table = RawTable::new_in(alloc);
        for i in 0..50u64 {
            table.insert(i, i, |k| *k);
        }
        let capacity_before = table.capacity();
        let buckets_before = table.num_buckets();

        // The next allocation (the replacement table of the growth
        // transaction) fails.
        fail_at.store(allocs.load(Ordering::SeqCst), Ordering::SeqCst);
        let result = table.try_reserve(500, |k| *k);
        assert!(result.is_err());

        // The original table is preserved.
        assert_eq!(table.len(), 50);
        assert_eq!(table.capacity(), capacity_before);
        assert_eq!(table.num_buckets(), buckets_before);
        for i in 0..50u64 {
            assert!(table.find(i, |k| *k == i).is_some());
        }

        // With the failure disarmed, the same reserve succeeds.
        fail_at.store(usize::MAX, Ordering::SeqCst);
        assert!(table.try_reserve(500, |k| *k).is_ok());
        for i in 0..50u64 {
            assert!(table.find(i, |k| *k == i).is_some());
        }
    }
}
