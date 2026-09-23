//! Growth transactions: reserve, rehash and resize logic.
//!
//! A [`GrowthTransaction`] allocates the replacement table up front, migrates
//! the elements of the old table into it, and then either commits (swapping
//! the new table into place) or rolls back. Rollback happens automatically on
//! drop (e.g. when the hash function panics during migration): the allocation
//! of the table held by the transaction is freed, reclaiming the elements
//! already moved into it without dropping them, because the old table still
//! owns the elements. On commit the roles are swapped and the old table's
//! allocation is freed instead. An allocation failure leaves the original
//! table untouched.

use crate::TryReserveError;
use crate::alloc::Allocator;
use crate::control::{Group, Tag};
use crate::scopeguard::guard;
use crate::util::{likely, unlikely};
use core::mem;
use core::ptr;

use super::storage::{Fallibility, RawTableInner, TableLayout, bucket_mask_to_capacity};

/// A growth transaction: owns the replacement table while elements are
/// migrated into it.
///
/// The transaction frees the allocation of the table it holds when it is
/// dropped:
///
/// * If the hash function panics during [`GrowthTransaction::migrate`], the
///   unwinding drops the transaction, which frees the new table without
///   dropping any elements already moved into it (the old table still owns
///   the elements, so nothing is leaked or double-dropped).
///
/// * If [`GrowthTransaction::commit`] is called, the new table is swapped
///   into place and the transaction drops the old table's allocation
///   instead.
pub(super) struct GrowthTransaction<'a, A: Allocator> {
    // The table being built (or, after `commit`, the old table being
    // discarded).
    table: RawTableInner,

    // The allocator used to allocate `table`, also used to free it.
    alloc: &'a A,

    // The layout used to allocate `table`, also used to free it.
    table_layout: TableLayout,
}

impl<'a, A: Allocator> GrowthTransaction<'a, A> {
    /// Allocates the replacement table for a growth transaction, with all
    /// control bytes initialized to `Tag::EMPTY`.
    ///
    /// On allocation failure the error is returned and the original table is
    /// left untouched.
    fn prepare(
        alloc: &'a A,
        table_layout: TableLayout,
        capacity: usize,
        fallibility: Fallibility,
    ) -> Result<Self, TryReserveError> {
        // Allocate and initialize the new table.
        let table =
            RawTableInner::fallible_with_capacity(alloc, table_layout, capacity, fallibility)?;
        Ok(Self {
            table,
            alloc,
            table_layout,
        })
    }

    /// Migrates all elements from `old` into the new table held by the
    /// transaction.
    ///
    /// The hash function may panic, in which case the transaction rolls back
    /// on unwind: the new table is freed without dropping the elements that
    /// were already moved into it, and `old` is left unmodified.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is
    /// [`undefined behavior`]:
    ///
    /// * The `table_layout` passed to [`GrowthTransaction::prepare`] must be
    ///   the same [`TableLayout`] as the `TableLayout` used to allocate `old`;
    ///
    /// * `old` must have properly initialized control bytes;
    ///
    /// * The new table must have enough capacity for all elements of `old`,
    ///   otherwise this function never returns (it loops infinitely) or
    ///   writes out of bounds. See [`RawTableInner::find_insert_index`] for
    ///   more information.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn migrate(
        &mut self,
        old: &mut RawTableInner,
        hasher: &dyn Fn(&mut RawTableInner, usize) -> u64,
        layout: TableLayout,
    ) {
        // SAFETY: We know for sure that `old` will outlive the returned
        // `FullBucketsIndices` iterator, and the caller of this function
        // ensures that the control bytes are properly initialized.
        unsafe {
            for full_byte_index in old.full_buckets_indices() {
                // This may panic, in which case the transaction rolls back.
                let hash = hasher(old, full_byte_index);

                // SAFETY:
                // We can use a simpler version of insert() here since:
                // 1. There are no DELETED entries.
                // 2. We know there is enough space in the table.
                // 3. All elements are unique.
                // 4. The caller of this function guarantees that `capacity > 0`
                //    so the new table must already have some allocated memory.
                // 5. We set `growth_left` and `items` fields of the new table
                //    after the loop.
                // 6. We insert into the table, at the returned index, the data
                //    matching the given hash immediately after calling this function.
                let (new_index, _) = self.table.prepare_insert_index(hash);

                // SAFETY:
                //
                // * `src` is valid for reads of `layout.size` bytes, since the
                //   old table is alive and the `full_byte_index` is guaranteed to be
                //   within bounds (see `FullBucketsIndices::next_impl`);
                //
                // * `dst` is valid for writes of `layout.size` bytes, since the
                //   caller ensures that `table_layout` matches the [`TableLayout`]
                //   that was used to allocate the old table and we have the `new_index`
                //   returned by `prepare_insert_index`.
                //
                // * Both `src` and `dst` are properly aligned.
                //
                // * Both `src` and `dst` point to different region of memory.
                ptr::copy_nonoverlapping(
                    old.bucket_ptr(full_byte_index, layout.size),
                    self.table.bucket_ptr(new_index, layout.size),
                    layout.size,
                );
            }
        }

        // The hash function didn't panic, so we can safely set the
        // `growth_left` and `items` fields of the new table.
        self.table.growth_left -= old.items;
        self.table.items = old.items;
    }

    /// Commits the transaction: swaps the fully migrated new table into
    /// `old`. The allocation of the previous table is freed when the
    /// transaction is dropped; its elements are not dropped since they have
    /// been moved into the new table.
    fn commit(self, old: &mut RawTableInner) {
        let mut this = self;
        mem::swap(old, &mut this.table);
    }
}

impl<A: Allocator> Drop for GrowthTransaction<'_, A> {
    fn drop(&mut self) {
        if !self.table.is_empty_singleton() {
            // SAFETY:
            // 1. We have checked that the table is allocated.
            // 2. We know for sure that the `alloc` and `table_layout` match the
            //    [`Allocator`] and [`TableLayout`] used to allocate this table.
            unsafe { self.table.free_buckets(self.alloc, self.table_layout) };
        }
    }
}

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
    unsafe fn prepare_rehash_in_place(&mut self) {
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

        // Allocate the replacement table first. On allocation failure the
        // error is returned and the original table is left untouched.
        let mut txn = GrowthTransaction::prepare(alloc, layout, capacity, fallibility)?;

        // Migrate all elements into the new table. If the hash function
        // panics, the transaction rolls back: the new table is freed without
        // dropping the elements already moved into it, and `self` is left
        // unmodified.
        //
        // SAFETY:
        // 1. We know for sure that `capacity >= self.items`.
        // 2. We know for sure that `layout` matches the [`TableLayout`] that
        //    was used to allocate this table.
        // 3. The caller ensures that the control bytes of the `RawTableInner`
        //    are already initialized.
        unsafe {
            txn.migrate(self, hasher, layout);
        }

        // We successfully copied all elements without panicking. Now replace
        // self with the new table. The old table will have its memory freed
        // but the items will not be dropped (since they have been moved into
        // the new table).
        txn.commit(self);

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
