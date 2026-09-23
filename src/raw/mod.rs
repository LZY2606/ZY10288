//! The raw table implementation, split into three cooperating pieces:
//!
//! * [`storage`] owns the physical layout: control bytes (including the
//!   mirrored group), bucket pointers and the allocation itself.
//! * [`probe`] owns the probing policy: a [`probe::ProbeCursor`] produces the
//!   group positions of a probe sequence and its termination condition.
//! * [`growth`] owns reserve/rehash/resize as a transaction that allocates
//!   the new table first, migrates the elements, and rolls back (reclaiming
//!   the new allocation) if the hash function panics.
//!
//! [`RawTable`] itself is only a thin typed wrapper that combines the three.

mod growth;
mod probe;
mod storage;

use self::probe::ProbeCursor;
pub(crate) use self::storage::Bucket;
use self::storage::{
    Fallibility, RawTableInner, SizedTypeProperties, TableLayout, capacity_to_buckets, offset_from,
};

use crate::TryReserveError;

use crate::control::{BitMaskIter, Group, Tag};

use crate::scopeguard::{ScopeGuard, guard};

use crate::util::{likely, unlikely};

use core::alloc::Layout;

use core::array;

use core::iter::FusedIterator;

use core::marker::PhantomData;

use core::mem;

use core::ptr;

use core::ptr::NonNull;

#[cfg(test)]
use crate::alloc::AllocError;

use crate::alloc::{Allocator, Global};

/// A raw hash table with an unsafe API.
pub(crate) struct RawTable<T, A: Allocator = Global> {
    table: RawTableInner,
    alloc: A,
    // Tell dropck that we own instances of T.
    marker: PhantomData<T>,
}

impl<T> RawTable<T, Global> {
    /// Creates a new empty hash table without allocating any memory.
    ///
    /// In effect this returns a table with exactly 1 bucket. However we can
    /// leave the data pointer dangling since that bucket is never written to
    /// due to our load factor forcing us to always have at least 1 free bucket.
    #[inline]
    #[cfg_attr(feature = "rustc-dep-of-std", rustc_const_stable_indirect)]
    pub(crate) const fn new() -> Self {
        Self {
            table: RawTableInner::NEW,
            alloc: Global,
            marker: PhantomData,
        }
    }

    /// Allocates a new hash table with at least enough capacity for inserting
    /// the given number of elements without reallocating.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<T, A: Allocator> RawTable<T, A> {
    const TABLE_LAYOUT: TableLayout = TableLayout::new::<T>();

    /// Creates a new empty hash table without allocating any memory, using the
    /// given allocator.
    ///
    /// In effect this returns a table with exactly 1 bucket. However we can
    /// leave the data pointer dangling since that bucket is never written to
    /// due to our load factor forcing us to always have at least 1 free bucket.
    #[inline]
    #[cfg_attr(feature = "rustc-dep-of-std", rustc_const_stable_indirect)]
    pub(crate) const fn new_in(alloc: A) -> Self {
        Self {
            table: RawTableInner::NEW,
            alloc,
            marker: PhantomData,
        }
    }

    /// Allocates a new hash table with the given number of buckets.
    ///
    /// The control bytes are left uninitialized.
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn new_uninitialized(
        alloc: A,
        buckets: usize,
        fallibility: Fallibility,
    ) -> Result<Self, TryReserveError> {
        debug_assert!(buckets.is_power_of_two());

        Ok(Self {
            table: unsafe {
                RawTableInner::new_uninitialized(&alloc, Self::TABLE_LAYOUT, buckets, fallibility)
            }?,
            alloc,
            marker: PhantomData,
        })
    }

    /// Allocates a new hash table using the given allocator, with at least enough capacity for
    /// inserting the given number of elements without reallocating.
    pub(crate) fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            table: RawTableInner::with_capacity(&alloc, Self::TABLE_LAYOUT, capacity),
            alloc,
            marker: PhantomData,
        }
    }

    /// Returns a reference to the underlying allocator.
    #[inline]
    pub(crate) fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns pointer to one past last `data` element in the table as viewed from
    /// the start point of the allocation.
    ///
    /// The caller must ensure that the `RawTable` outlives the returned [`NonNull<T>`],
    /// otherwise using it may result in [`undefined behavior`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(crate) fn data_end(&self) -> NonNull<T> {
        //                        `self.table.ctrl.cast()` returns pointer that
        //                        points here (to the end of `T0`)
        //                          ∨
        // [Pad], T_n, ..., T1, T0, |CT0, CT1, ..., CT_n|, CTa_0, CTa_1, ..., CTa_m
        //                           \________  ________/
        //                                    \/
        //       `n = buckets - 1`, i.e. `RawTable::num_buckets() - 1`
        //
        // where: T0...T_n  - our stored data;
        //        CT0...CT_n - control bytes or metadata for `data`.
        //        CTa_0...CTa_m - additional control bytes, where `m = Group::WIDTH - 1` (so that the search
        //                        with loading `Group` bytes from the heap works properly, even if the result
        //                        of `h1(hash) & self.bucket_mask` is equal to `self.bucket_mask`). See also
        //                        `RawTableInner::set_ctrl` function.
        //
        // P.S. `h1(hash) & self.bucket_mask` is the same as `hash as usize % self.num_buckets()` because the number
        // of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
        self.table.ctrl.cast()
    }

    /// Returns pointer to start of data table.
    #[inline]
    #[cfg(feature = "nightly")]
    pub(crate) unsafe fn data_start(&self) -> NonNull<T> {
        unsafe { NonNull::new_unchecked(self.data_end().as_ptr().wrapping_sub(self.num_buckets())) }
    }

    /// Returns the total amount of memory allocated internally by the hash
    /// table, in bytes.
    ///
    /// The returned number is informational only. It is intended to be
    /// primarily used for memory profiling.
    #[inline]
    pub(crate) fn allocation_size(&self) -> usize {
        // SAFETY: We use the same `table_layout` that was used to allocate
        // this table.
        unsafe { self.table.allocation_size_or_zero(Self::TABLE_LAYOUT) }
    }

    /// Returns the index of a bucket from a `Bucket`.
    #[inline]
    pub(crate) unsafe fn bucket_index(&self, bucket: &Bucket<T>) -> usize {
        unsafe { bucket.to_base_index(self.data_end()) }
    }

    /// Returns a pointer to an element in the table.
    ///
    /// The caller must ensure that the `RawTable` outlives the returned [`Bucket<T>`],
    /// otherwise using it may result in [`undefined behavior`].
    ///
    /// # Safety
    ///
    /// If `size_of::<T>() != 0`, then the caller of this function must observe the
    /// following safety rules:
    ///
    /// * The table must already be allocated;
    ///
    /// * The `index` must not be greater than the number returned by the [`RawTable::num_buckets`]
    ///   function, i.e. `(index + 1) <= self.num_buckets()`.
    ///
    /// It is safe to call this function with index of zero (`index == 0`) on a table that has
    /// not been allocated, but using the returned [`Bucket`] results in [`undefined behavior`].
    ///
    /// If `size_of::<T>() == 0`, then the only requirement is that the `index` must
    /// not be greater than the number returned by the [`RawTable::num_buckets`] function, i.e.
    /// `(index + 1) <= self.num_buckets()`.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(crate) unsafe fn bucket(&self, index: usize) -> Bucket<T> {
        // If size_of::<T>() != 0 then return a pointer to the `element` in the `data part` of the table
        // (we start counting from "0", so that in the expression T[n], the "n" index actually one less than
        // the "buckets" number of our `RawTable`, i.e. "n = RawTable::num_buckets() - 1"):
        //
        //           `table.bucket(3).as_ptr()` returns a pointer that points here in the `data`
        //           part of the `RawTable`, i.e. to the start of T3 (see `Bucket::as_ptr`)
        //                  |
        //                  |               `base = self.data_end()` points here
        //                  |               (to the start of CT0 or to the end of T0)
        //                  v                 v
        // [Pad], T_n, ..., |T3|, T2, T1, T0, |CT0, CT1, CT2, CT3, ..., CT_n, CTa_0, CTa_1, ..., CTa_m
        //                     ^                                              \__________  __________/
        //        `table.bucket(3)` returns a pointer that points                        \/
        //         here in the `data` part of the `RawTable` (to              additional control bytes
        //         the end of T3)                                              `m = Group::WIDTH - 1`
        //
        // where: T0...T_n  - our stored data;
        //        CT0...CT_n - control bytes or metadata for `data`;
        //        CTa_0...CTa_m - additional control bytes (so that the search with loading `Group` bytes from
        //                        the heap works properly, even if the result of `h1(hash) & self.table.bucket_mask`
        //                        is equal to `self.table.bucket_mask`). See also `RawTableInner::set_ctrl` function.
        //
        // P.S. `h1(hash) & self.table.bucket_mask` is the same as `hash as usize % self.num_buckets()` because the number
        // of buckets is a power of two, and `self.table.bucket_mask = self.num_buckets() - 1`.
        debug_assert_ne!(self.table.bucket_mask, 0);
        debug_assert!(index < self.num_buckets());
        unsafe { Bucket::from_base_index(self.data_end(), index) }
    }

    /// Erases an element from the table without dropping it.
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn erase_no_drop(&mut self, item: &Bucket<T>) {
        unsafe {
            let index = self.bucket_index(item);
            self.table.erase(index);
        }
    }

    /// Erases an element from the table, dropping it in place.
    #[cfg_attr(feature = "inline-more", inline)]
    #[expect(clippy::needless_pass_by_value)]
    pub(crate) unsafe fn erase(&mut self, item: Bucket<T>) {
        unsafe {
            // Erase the element from the table first since drop might panic.
            self.erase_no_drop(&item);
            item.drop();
        }
    }

    /// Removes an element from the table, returning it.
    ///
    /// This also returns an index to the newly free bucket.
    #[cfg_attr(feature = "inline-more", inline)]
    #[expect(clippy::needless_pass_by_value)]
    pub(crate) unsafe fn remove(&mut self, item: Bucket<T>) -> (T, usize) {
        unsafe {
            self.erase_no_drop(&item);
            (item.read(), self.bucket_index(&item))
        }
    }

    /// Removes an element from the table, returning it.
    ///
    /// This also returns an index to the newly free bucket
    /// and the former `Tag` for that bucket.
    #[cfg_attr(feature = "inline-more", inline)]
    #[expect(clippy::needless_pass_by_value)]
    pub(crate) unsafe fn remove_tagged(&mut self, item: Bucket<T>) -> (T, usize, Tag) {
        unsafe {
            let index = self.bucket_index(&item);
            let tag = *self.table.ctrl(index);
            self.table.erase(index);
            (item.read(), index, tag)
        }
    }

    /// Finds and removes an element from the table, returning it.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn remove_entry(&mut self, hash: u64, eq: impl FnMut(&T) -> bool) -> Option<T> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.find(hash, eq) {
            Some(bucket) => Some(unsafe { self.remove(bucket).0 }),
            None => None,
        }
    }

    /// Marks all table buckets as empty without dropping their contents.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn clear_no_drop(&mut self) {
        self.table.clear_no_drop();
    }

    /// Removes all elements from the table without freeing the backing memory.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn clear(&mut self) {
        if self.is_empty() {
            // Special case empty table to avoid surprising O(capacity) time.
            return;
        }
        // Ensure that the table is reset even if one of the drops panic
        let mut self_ = guard(self, |self_| self_.clear_no_drop());
        unsafe {
            // SAFETY: ScopeGuard sets to zero the `items` field of the table
            // even in case of panic during the dropping of the elements so
            // that there will be no double drop of the elements.
            self_.table.drop_elements::<T>();
        }
    }

    /// Shrinks the table to fit `max(self.len(), min_size)` elements.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn shrink_to(&mut self, min_size: usize, hasher: impl Fn(&T) -> u64) {
        // Calculate the minimal number of elements that we need to reserve
        // space for.
        let min_size = usize::max(self.table.items, min_size);
        if min_size == 0 {
            let mut old_inner = mem::replace(&mut self.table, RawTableInner::NEW);
            unsafe {
                // SAFETY:
                // 1. We call the function only once;
                // 2. We know for sure that `alloc` and `table_layout` matches the [`Allocator`]
                //    and [`TableLayout`] that were used to allocate this table.
                // 3. If any elements' drop function panics, then there will only be a memory leak,
                //    because we have replaced the inner table with a new one.
                old_inner.drop_inner_table::<T, _>(&self.alloc, Self::TABLE_LAYOUT);
            }
            return;
        }

        // Calculate the number of buckets that we need for this number of
        // elements. If the calculation overflows then the requested bucket
        // count must be larger than what we have right and nothing needs to be
        // done.
        let Some(min_buckets) = capacity_to_buckets(min_size, Self::TABLE_LAYOUT) else {
            return;
        };

        // If we have more buckets than we need, shrink the table.
        if min_buckets < self.num_buckets() {
            // Fast path if the table is empty
            if self.table.items == 0 {
                let new_inner =
                    RawTableInner::with_capacity(&self.alloc, Self::TABLE_LAYOUT, min_size);
                let mut old_inner = mem::replace(&mut self.table, new_inner);
                unsafe {
                    // SAFETY:
                    // 1. We call the function only once;
                    // 2. We know for sure that `alloc` and `table_layout` matches the [`Allocator`]
                    //    and [`TableLayout`] that were used to allocate this table.
                    // 3. If any elements' drop function panics, then there will only be a memory leak,
                    //    because we have replaced the inner table with a new one.
                    old_inner.drop_inner_table::<T, _>(&self.alloc, Self::TABLE_LAYOUT);
                }
            } else {
                // SAFETY:
                // 1. We know for sure that `min_size >= self.table.items`.
                // 2. The [`RawTableInner`] must already have properly initialized control bytes since
                //    we will never expose RawTable::new_uninitialized in a public API.
                let result = unsafe { self.resize(min_size, hasher, Fallibility::Infallible) };

                // SAFETY: The result of calling the `resize` function cannot be an error
                // because `fallibility == Fallibility::Infallible.
                unsafe { result.unwrap_unchecked() };
            }
        }
    }

    /// Ensures that at least `additional` items can be inserted into the table
    /// without reallocation.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn reserve(&mut self, additional: usize, hasher: impl Fn(&T) -> u64) {
        if unlikely(additional > self.table.growth_left) {
            // SAFETY: The [`RawTableInner`] must already have properly initialized control
            // bytes since we will never expose RawTable::new_uninitialized in a public API.
            let result =
                unsafe { self.reserve_rehash(additional, hasher, Fallibility::Infallible) };

            // SAFETY: All allocation errors will be caught inside `RawTableInner::reserve_rehash`.
            unsafe { result.unwrap_unchecked() };
        }
    }

    /// Tries to ensure that at least `additional` items can be inserted into
    /// the table without reallocation.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn try_reserve(
        &mut self,
        additional: usize,
        hasher: impl Fn(&T) -> u64,
    ) -> Result<(), TryReserveError> {
        if additional > self.table.growth_left {
            // SAFETY: The [`RawTableInner`] must already have properly initialized control
            // bytes since we will never expose RawTable::new_uninitialized in a public API.
            unsafe { self.reserve_rehash(additional, hasher, Fallibility::Fallible) }
        } else {
            Ok(())
        }
    }

    /// Out-of-line slow path for `reserve` and `try_reserve`.
    ///
    /// # Safety
    ///
    /// The [`RawTableInner`] must have properly initialized control bytes,
    /// otherwise calling this function results in [`undefined behavior`]
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[cold]
    #[inline(never)]
    unsafe fn reserve_rehash(
        &mut self,
        additional: usize,
        hasher: impl Fn(&T) -> u64,
        fallibility: Fallibility,
    ) -> Result<(), TryReserveError> {
        unsafe {
            // SAFETY:
            // 1. We know for sure that `alloc` and `layout` matches the [`Allocator`] and
            //    [`TableLayout`] that were used to allocate this table.
            // 2. The `drop` function is the actual drop function of the elements stored in
            //    the table.
            // 3. The caller ensures that the control bytes of the `RawTableInner`
            //    are already initialized.
            self.table.reserve_rehash_inner(
                &self.alloc,
                additional,
                &|table, index| hasher(table.bucket::<T>(index).as_ref()),
                fallibility,
                Self::TABLE_LAYOUT,
                if T::NEEDS_DROP {
                    Some(|ptr| ptr::drop_in_place(ptr.cast::<T>()))
                } else {
                    None
                },
            )
        }
    }

    /// Allocates a new table of a different size and moves the contents of the
    /// current table into it.
    ///
    /// # Safety
    ///
    /// The [`RawTableInner`] must have properly initialized control bytes,
    /// otherwise calling this function results in [`undefined behavior`]
    ///
    /// The caller of this function must ensure that `capacity >= self.table.items`
    /// otherwise:
    ///
    /// * If `self.table.items != 0`, calling of this function with `capacity`
    ///   equal to 0 (`capacity == 0`) results in [`undefined behavior`].
    ///
    /// * If `self.table.items > capacity_to_buckets(capacity, Self::TABLE_LAYOUT)`
    ///   calling this function are never return (will loop infinitely).
    ///
    /// See [`RawTableInner::find_insert_index`] for more information.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    unsafe fn resize(
        &mut self,
        capacity: usize,
        hasher: impl Fn(&T) -> u64,
        fallibility: Fallibility,
    ) -> Result<(), TryReserveError> {
        // SAFETY:
        // 1. The caller of this function guarantees that `capacity >= self.table.items`.
        // 2. We know for sure that `alloc` and `layout` matches the [`Allocator`] and
        //    [`TableLayout`] that were used to allocate this table.
        // 3. The caller ensures that the control bytes of the `RawTableInner`
        //    are already initialized.
        unsafe {
            self.table.resize_inner(
                &self.alloc,
                capacity,
                &|table, index| hasher(table.bucket::<T>(index).as_ref()),
                fallibility,
                Self::TABLE_LAYOUT,
            )
        }
    }

    /// Inserts a new element into the table, and returns its raw bucket.
    ///
    /// This does not check if the given element already exists in the table.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn insert(&mut self, hash: u64, value: T, hasher: impl Fn(&T) -> u64) -> Bucket<T> {
        unsafe {
            // SAFETY:
            // 1. The [`RawTableInner`] must already have properly initialized control bytes since
            //    we will never expose `RawTable::new_uninitialized` in a public API.
            //
            // 2. We reserve additional space (if necessary) right after calling this function.
            let mut index = self.table.find_insert_index(hash);

            // We can avoid growing the table once we have reached our load factor if we are replacing
            // a tombstone. This works since the number of EMPTY slots does not change in this case.
            //
            // SAFETY: The function is guaranteed to return an index in the range `0..=self.num_buckets()`.
            let old_ctrl = *self.table.ctrl(index);
            if unlikely(self.table.growth_left == 0 && old_ctrl.special_is_empty()) {
                self.reserve(1, hasher);
                // SAFETY: We know for sure that `RawTableInner` has control bytes
                // initialized and that there is extra space in the table.
                index = self.table.find_insert_index(hash);
            }

            self.insert_at_index(hash, index, value)
        }
    }

    /// Inserts a new element into the table, and returns a mutable reference to it.
    ///
    /// This does not check if the given element already exists in the table.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn insert_entry(
        &mut self,
        hash: u64,
        value: T,
        hasher: impl Fn(&T) -> u64,
    ) -> &mut T {
        unsafe { self.insert(hash, value, hasher).as_mut() }
    }

    /// Inserts a new element into the table, without growing the table.
    ///
    /// There must be enough space in the table to insert the new element.
    ///
    /// This does not check if the given element already exists in the table.
    #[cfg_attr(feature = "inline-more", inline)]
    #[cfg(feature = "rustc-internal-api")]
    pub(crate) unsafe fn insert_no_grow(&mut self, hash: u64, value: T) -> Bucket<T> {
        unsafe {
            let (index, old_ctrl) = self.table.prepare_insert_index(hash);
            let bucket = self.table.bucket(index);

            // If we are replacing a DELETED entry then we don't need to update
            // the load counter.
            self.table.growth_left -= old_ctrl.special_is_empty() as usize;

            bucket.write(value);
            self.table.items += 1;
            bucket
        }
    }

    /// Temporarily removes a bucket, applying the given function to the removed
    /// element and optionally put back the returned value in the same bucket.
    ///
    /// Returns tag for bucket if the bucket is emptied out.
    ///
    /// This does not check if the given bucket is actually occupied.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) unsafe fn replace_bucket_with<F>(&mut self, bucket: Bucket<T>, f: F) -> Option<Tag>
    where
        F: FnOnce(T) -> Option<T>,
    {
        unsafe {
            let index = self.bucket_index(&bucket);
            let old_ctrl = *self.table.ctrl(index);
            debug_assert!(self.is_bucket_full(index));
            let old_growth_left = self.table.growth_left;
            let item = self.remove(bucket).0;
            if let Some(new_item) = f(item) {
                self.table.growth_left = old_growth_left;
                self.table.set_ctrl(index, old_ctrl);
                self.table.items += 1;
                self.bucket(index).write(new_item);
                None
            } else {
                Some(old_ctrl)
            }
        }
    }

    /// Searches for an element in the table. If the element is not found,
    /// returns `Err` with the position of a slot where an element with the
    /// same hash could be inserted.
    ///
    /// This function may resize the table if additional space is required for
    /// inserting an element.
    #[inline]
    pub(crate) fn find_or_find_insert_index(
        &mut self,
        hash: u64,
        mut eq: impl FnMut(&T) -> bool,
        hasher: impl Fn(&T) -> u64,
    ) -> Result<Bucket<T>, usize> {
        self.reserve(1, hasher);

        unsafe {
            // SAFETY:
            // 1. We know for sure that there is at least one empty `bucket` in the table.
            // 2. The [`RawTableInner`] must already have properly initialized control bytes since we will
            //    never expose `RawTable::new_uninitialized` in a public API.
            // 3. The `find_or_find_insert_index_inner` function returns the `index` of only the full bucket,
            //    which is in the range `0..self.num_buckets()` (since there is at least one empty `bucket` in
            //    the table), so calling `self.bucket(index)` and `Bucket::as_ref` is safe.
            match self
                .table
                .find_or_find_insert_index_inner(hash, &mut |index| eq(self.bucket(index).as_ref()))
            {
                // SAFETY: See explanation above.
                Ok(index) => Ok(self.bucket(index)),
                Err(index) => Err(index),
            }
        }
    }

    /// Inserts a new element into the table at the given index with the given hash,
    /// and returns its raw bucket.
    ///
    /// # Safety
    ///
    /// `index` must point to a slot previously returned by
    /// `find_or_find_insert_index`, and no mutation of the table must have
    /// occurred since that call.
    #[inline]
    pub(crate) unsafe fn insert_at_index(
        &mut self,
        hash: u64,
        index: usize,
        value: T,
    ) -> Bucket<T> {
        unsafe { self.insert_tagged_at_index(Tag::full(hash), index, value) }
    }

    /// Inserts a new element into the table at the given index with the given tag,
    /// and returns its raw bucket.
    ///
    /// # Safety
    ///
    /// `index` must point to a slot previously returned by
    /// `find_or_find_insert_index`, and no mutation of the table must have
    /// occurred since that call.
    #[inline]
    pub(crate) unsafe fn insert_tagged_at_index(
        &mut self,
        tag: Tag,
        index: usize,
        value: T,
    ) -> Bucket<T> {
        unsafe {
            let old_ctrl = *self.table.ctrl(index);
            self.table.record_item_insert_at(index, old_ctrl, tag);

            let bucket = self.bucket(index);
            bucket.write(value);
            bucket
        }
    }

    /// Searches for an element in the table.
    #[inline]
    pub(crate) fn find(&self, hash: u64, mut eq: impl FnMut(&T) -> bool) -> Option<Bucket<T>> {
        unsafe {
            // SAFETY:
            // 1. The [`RawTableInner`] must already have properly initialized control bytes since we
            //    will never expose `RawTable::new_uninitialized` in a public API.
            // 1. The `find_inner` function returns the `index` of only the full bucket, which is in
            //    the range `0..self.num_buckets()`, so calling `self.bucket(index)` and `Bucket::as_ref`
            //    is safe.
            let result = self
                .table
                .find_inner(hash, &mut |index| eq(self.bucket(index).as_ref()));

            // Avoid `Option::map` because it bloats LLVM IR.
            match result {
                // SAFETY: See explanation above.
                Some(index) => Some(self.bucket(index)),
                None => None,
            }
        }
    }

    /// Gets a reference to an element in the table.
    #[inline]
    pub(crate) fn get(&self, hash: u64, eq: impl FnMut(&T) -> bool) -> Option<&T> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.find(hash, eq) {
            Some(bucket) => Some(unsafe { bucket.as_ref() }),
            None => None,
        }
    }

    /// Gets a mutable reference to an element in the table.
    #[inline]
    pub(crate) fn get_mut(&mut self, hash: u64, eq: impl FnMut(&T) -> bool) -> Option<&mut T> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.find(hash, eq) {
            Some(bucket) => Some(unsafe { bucket.as_mut() }),
            None => None,
        }
    }

    /// Gets a reference to an element in the table at the given bucket index.
    #[inline]
    pub(crate) fn get_bucket(&self, index: usize) -> Option<&T> {
        unsafe {
            if index < self.num_buckets() && self.is_bucket_full(index) {
                Some(self.bucket(index).as_ref())
            } else {
                None
            }
        }
    }

    /// Gets a mutable reference to an element in the table at the given bucket index.
    #[inline]
    pub(crate) fn get_bucket_mut(&mut self, index: usize) -> Option<&mut T> {
        unsafe {
            if index < self.num_buckets() && self.is_bucket_full(index) {
                Some(self.bucket(index).as_mut())
            } else {
                None
            }
        }
    }

    /// Returns a pointer to an element in the table, but only after verifying that
    /// the index is in-bounds and the bucket is occupied.
    #[inline]
    pub(crate) fn checked_bucket(&self, index: usize) -> Option<Bucket<T>> {
        unsafe {
            if index < self.num_buckets() && self.is_bucket_full(index) {
                Some(self.bucket(index))
            } else {
                None
            }
        }
    }

    /// Attempts to get mutable references to `N` entries in the table at once.
    ///
    /// Returns an array of length `N` with the results of each query.
    ///
    /// At most one mutable reference will be returned to any entry. `None` will be returned if any
    /// of the hashes are duplicates. `None` will be returned if the hash is not found.
    ///
    /// The `eq` argument should be a closure such that `eq(i, k)` returns true if `k` is equal to
    /// the `i`th key to be looked up.
    pub(crate) fn get_disjoint_mut<const N: usize>(
        &mut self,
        hashes: [u64; N],
        eq: impl FnMut(usize, &T) -> bool,
    ) -> [Option<&'_ mut T>; N] {
        unsafe {
            let ptrs = self.get_disjoint_mut_pointers(hashes, eq);

            for (i, cur) in ptrs.iter().enumerate() {
                assert!(
                    !(cur.is_some() && ptrs[..i].contains(cur)),
                    "duplicate keys found"
                );
            }
            // All bucket are distinct from all previous buckets so we're clear to return the result
            // of the lookup.

            ptrs.map(|ptr| ptr.map(|mut ptr| ptr.as_mut()))
        }
    }

    pub(crate) unsafe fn get_disjoint_unchecked_mut<const N: usize>(
        &mut self,
        hashes: [u64; N],
        eq: impl FnMut(usize, &T) -> bool,
    ) -> [Option<&'_ mut T>; N] {
        let ptrs = unsafe { self.get_disjoint_mut_pointers(hashes, eq) };
        ptrs.map(|ptr| ptr.map(|mut ptr| unsafe { ptr.as_mut() }))
    }

    unsafe fn get_disjoint_mut_pointers<const N: usize>(
        &mut self,
        hashes: [u64; N],
        mut eq: impl FnMut(usize, &T) -> bool,
    ) -> [Option<NonNull<T>>; N] {
        array::from_fn(|i| {
            self.find(hashes[i], |k| eq(i, k))
                .map(|cur| cur.as_non_null())
        })
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// This number is a lower bound; the table might be able to hold
    /// more, but is guaranteed to be able to hold at least this many.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.table.items + self.table.growth_left
    }

    /// Returns the number of elements in the table.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.table.items
    }

    /// Returns `true` if the table contains no elements.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of buckets in the table.
    #[inline]
    pub(crate) fn num_buckets(&self) -> usize {
        self.table.bucket_mask + 1
    }

    /// Checks whether the bucket at `index` is full.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index` is less than the number of buckets.
    #[inline]
    pub(crate) unsafe fn is_bucket_full(&self, index: usize) -> bool {
        unsafe { self.table.is_bucket_full(index) }
    }

    /// Returns an iterator over every element in the table. It is up to
    /// the caller to ensure that the `RawTable` outlives the `RawIter`.
    /// Because we cannot make the `next` method unsafe on the `RawIter`
    /// struct, we have to make the `iter` method unsafe.
    #[inline]
    pub(crate) unsafe fn iter(&self) -> RawIter<T> {
        // SAFETY:
        // 1. The caller must uphold the safety contract for `iter` method.
        // 2. The [`RawTableInner`] must already have properly initialized control bytes since
        //    we will never expose RawTable::new_uninitialized in a public API.
        unsafe { self.table.iter() }
    }

    /// Returns an iterator over occupied buckets that could match a given hash.
    ///
    /// `RawTable` only stores 7 bits of the hash value, so this iterator may
    /// return items that have a hash value different than the one provided. You
    /// should always validate the returned values before using them.
    ///
    /// It is up to the caller to ensure that the `RawTable` outlives the
    /// `RawIterHash`. Because we cannot make the `next` method unsafe on the
    /// `RawIterHash` struct, we have to make the `iter_hash` method unsafe.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) unsafe fn iter_hash(&self, hash: u64) -> RawIterHash<T> {
        unsafe { RawIterHash::new(self, hash) }
    }

    /// Returns an iterator over occupied bucket indices that could match a given hash.
    ///
    /// `RawTable` only stores 7 bits of the hash value, so this iterator may
    /// return items that have a hash value different than the one provided. You
    /// should always validate the returned values before using them.
    ///
    /// It is up to the caller to ensure that the `RawTable` outlives the
    /// `RawIterHashIndices`. Because we cannot make the `next` method unsafe on the
    /// `RawIterHashIndices` struct, we have to make the `iter_hash_buckets` method unsafe.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) unsafe fn iter_hash_buckets(&self, hash: u64) -> RawIterHashIndices {
        unsafe { RawIterHashIndices::new(&self.table, hash) }
    }

    /// Returns an iterator over full buckets indices in the table.
    ///
    /// See [`RawTableInner::full_buckets_indices`] for safety conditions.
    #[inline(always)]
    pub(crate) unsafe fn full_buckets_indices(&self) -> FullBucketsIndices {
        unsafe { self.table.full_buckets_indices() }
    }

    /// Returns an iterator which removes all elements from the table without
    /// freeing the memory.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn drain(&mut self) -> RawDrain<'_, T, A> {
        unsafe {
            let iter = self.iter();
            self.drain_iter_from(iter)
        }
    }

    /// Returns an iterator which removes all elements from the table without
    /// freeing the memory.
    ///
    /// Iteration starts at the provided iterator's current location.
    ///
    /// It is up to the caller to ensure that the iterator is valid for this
    /// `RawTable` and covers all items that remain in the table.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) unsafe fn drain_iter_from(&mut self, iter: RawIter<T>) -> RawDrain<'_, T, A> {
        debug_assert_eq!(iter.len(), self.len());
        RawDrain {
            iter,
            table: mem::replace(&mut self.table, RawTableInner::NEW),
            orig_table: NonNull::from(&mut self.table),
            marker: PhantomData,
        }
    }

    /// Returns an iterator which consumes all elements from the table.
    ///
    /// Iteration starts at the provided iterator's current location.
    ///
    /// It is up to the caller to ensure that the iterator is valid for this
    /// `RawTable` and covers all items that remain in the table.
    pub(crate) unsafe fn into_iter_from(self, iter: RawIter<T>) -> RawIntoIter<T, A> {
        debug_assert_eq!(iter.len(), self.len());

        let allocation = self.into_allocation();
        RawIntoIter {
            iter,
            allocation,
            marker: PhantomData,
        }
    }

    /// Converts the table into a raw allocation. The contents of the table
    /// should be dropped using a `RawIter` before freeing the allocation.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn into_allocation(self) -> Option<(NonNull<u8>, Layout, A)> {
        let alloc = if self.table.is_empty_singleton() {
            None
        } else {
            let (layout, ctrl_offset) = {
                let option = Self::TABLE_LAYOUT.calculate_layout_for(self.table.num_buckets());
                unsafe { option.unwrap_unchecked() }
            };
            Some((
                unsafe { NonNull::new_unchecked(self.table.ctrl.as_ptr().sub(ctrl_offset).cast()) },
                layout,
                unsafe { ptr::read(&raw const self.alloc) },
            ))
        };
        mem::forget(self);
        alloc
    }
}

unsafe impl<T, A: Allocator> Send for RawTable<T, A>
where
    T: Send,
    A: Send,
{
}

unsafe impl<T, A: Allocator> Sync for RawTable<T, A>
where
    T: Sync,
    A: Sync,
{
}

impl<T: Clone, A: Allocator + Clone> Clone for RawTable<T, A> {
    fn clone(&self) -> Self {
        if self.table.is_empty_singleton() {
            Self::new_in(self.alloc.clone())
        } else {
            // SAFETY: This is safe as we are taking the size of an already allocated table
            // and therefore capacity overflow cannot occur, `self.table.num_buckets()` is power
            // of two and all allocator errors will be caught inside `RawTableInner::new_uninitialized`.
            let result = unsafe {
                Self::new_uninitialized(
                    self.alloc.clone(),
                    self.table.num_buckets(),
                    Fallibility::Infallible,
                )
            };

            // SAFETY: The result of calling the `new_uninitialized` function cannot be an error
            // because `fallibility == Fallibility::Infallible.
            let mut new_table = unsafe { result.unwrap_unchecked() };

            // SAFETY:
            // Cloning elements may fail (the clone function may panic). But we don't
            // need to worry about uninitialized control bits, since:
            // 1. The number of items (elements) in the table is zero, which means that
            //    the control bits will not be read by Drop function.
            // 2. The `clone_from_spec` method will first copy all control bits from
            //    `self` (thus initializing them). But this will not affect the `Drop`
            //    function, since the `clone_from_spec` function sets `items` only after
            //    successfully cloning all elements.
            unsafe { new_table.clone_from_spec(self) };
            new_table
        }
    }

    fn clone_from(&mut self, source: &Self) {
        if source.table.is_empty_singleton() {
            let mut old_inner = mem::replace(&mut self.table, RawTableInner::NEW);
            unsafe {
                // SAFETY:
                // 1. We call the function only once;
                // 2. We know for sure that `alloc` and `table_layout` matches the [`Allocator`]
                //    and [`TableLayout`] that were used to allocate this table.
                // 3. If any elements' drop function panics, then there will only be a memory leak,
                //    because we have replaced the inner table with a new one.
                old_inner.drop_inner_table::<T, _>(&self.alloc, Self::TABLE_LAYOUT);
            }
        } else {
            unsafe {
                // Make sure that if any panics occurs, we clear the table and
                // leave it in an empty state.
                let mut self_ = guard(self, |self_| {
                    self_.clear_no_drop();
                });

                // First, drop all our elements without clearing the control
                // bytes. If this panics then the scope guard will clear the
                // table, leaking any elements that were not dropped yet.
                //
                // This leak is unavoidable: we can't try dropping more elements
                // since this could lead to another panic and abort the process.
                //
                // SAFETY: If something gets wrong we clear our table right after
                // dropping the elements, so there is no double drop, since `items`
                // will be equal to zero.
                self_.table.drop_elements::<T>();

                // If necessary, resize our table to match the source.
                if self_.num_buckets() != source.num_buckets() {
                    let new_inner = {
                        let result = RawTableInner::new_uninitialized(
                            &self_.alloc,
                            Self::TABLE_LAYOUT,
                            source.num_buckets(),
                            Fallibility::Infallible,
                        );
                        result.unwrap_unchecked()
                    };
                    // Replace the old inner with new uninitialized one. It's ok, since if something gets
                    // wrong `ScopeGuard` will initialize all control bytes and leave empty table.
                    let mut old_inner = mem::replace(&mut self_.table, new_inner);
                    if !old_inner.is_empty_singleton() {
                        // SAFETY:
                        // 1. We have checked that our table is allocated.
                        // 2. We know for sure that `alloc` and `table_layout` matches
                        // the [`Allocator`] and [`TableLayout`] that were used to allocate this table.
                        old_inner.free_buckets(&self_.alloc, Self::TABLE_LAYOUT);
                    }
                }

                // Cloning elements may fail (the clone function may panic), but the `ScopeGuard`
                // inside the `clone_from_impl` function will take care of that, dropping all
                // cloned elements if necessary. Our `ScopeGuard` will clear the table.
                self_.clone_from_spec(source);

                // Disarm the scope guard if cloning was successful.
                ScopeGuard::into_inner(self_);
            }
        }
    }
}

/// Specialization of `clone_from` for `Copy` types
trait RawTableClone {
    unsafe fn clone_from_spec(&mut self, source: &Self);
}

impl<T: Clone, A: Allocator + Clone> RawTableClone for RawTable<T, A> {
    default_fn! {
        #[cfg_attr(feature = "inline-more", inline)]
        unsafe fn clone_from_spec(&mut self, source: &Self) {
            unsafe {
                self.clone_from_impl(source);
            }
        }
    }
}

#[cfg(feature = "nightly")]
impl<T: core::clone::TrivialClone, A: Allocator + Clone> RawTableClone for RawTable<T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn clone_from_spec(&mut self, source: &Self) {
        unsafe {
            source
                .table
                .ctrl(0)
                .copy_to_nonoverlapping(self.table.ctrl(0), self.table.num_ctrl_bytes());
            source
                .data_start()
                .as_ptr()
                .copy_to_nonoverlapping(self.data_start().as_ptr(), self.table.num_buckets());
        }

        self.table.items = source.table.items;
        self.table.growth_left = source.table.growth_left;
    }
}

impl<T: Clone, A: Allocator + Clone> RawTable<T, A> {
    /// Common code for `clone` and `clone_from`. Assumes:
    /// - `self.num_buckets() == source.num_buckets()`.
    /// - Any existing elements have been dropped.
    /// - The control bytes are not initialized yet.
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn clone_from_impl(&mut self, source: &Self) {
        // Copy the control bytes unchanged. We do this in a single pass
        unsafe {
            source
                .table
                .ctrl(0)
                .copy_to_nonoverlapping(self.table.ctrl(0), self.table.num_ctrl_bytes());
        }

        // The cloning of elements may panic, in which case we need
        // to make sure we drop only the elements that have been
        // cloned so far.
        let mut guard = guard((0, &mut *self), |(index, self_)| {
            if T::NEEDS_DROP {
                for i in 0..*index {
                    unsafe {
                        if self_.is_bucket_full(i) {
                            self_.bucket(i).drop();
                        }
                    }
                }
            }
        });

        unsafe {
            for from in source.iter() {
                let index = source.bucket_index(&from);
                let to = guard.1.bucket(index);
                to.write(from.as_ref().clone());

                // Update the index in case we need to unwind.
                guard.0 = index + 1;
            }
        }

        // Successfully cloned all items, no need to clean up.
        mem::forget(guard);

        self.table.items = source.table.items;
        self.table.growth_left = source.table.growth_left;
    }
}

impl<T, A: Allocator + Default> Default for RawTable<T, A> {
    #[inline]
    fn default() -> Self {
        Self::new_in(Default::default())
    }
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] T, A: Allocator> Drop for RawTable<T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        // SAFETY:
        // 1. We call the function only once;
        // 2. We know for sure that `alloc` and `table_layout` matches the [`Allocator`]
        //    and [`TableLayout`] that were used to allocate this table.
        // 3. If the drop function of any elements fails, then only a memory leak will occur,
        //    and we don't care because we are inside the `Drop` function of the `RawTable`,
        //    so there won't be any table left in an inconsistent state.
        unsafe {
            self.table
                .drop_inner_table::<T, _>(&self.alloc, Self::TABLE_LAYOUT);
        }
    }
}

#[cfg(not(feature = "nightly"))]
impl<T, A: Allocator> Drop for RawTable<T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        // SAFETY:
        // 1. We call the function only once;
        // 2. We know for sure that `alloc` and `table_layout` matches the [`Allocator`]
        //    and [`TableLayout`] that were used to allocate this table.
        // 3. If the drop function of any elements fails, then only a memory leak will occur,
        //    and we don't care because we are inside the `Drop` function of the `RawTable`,
        //    so there won't be any table left in an inconsistent state.
        unsafe {
            self.table
                .drop_inner_table::<T, _>(&self.alloc, Self::TABLE_LAYOUT);
        }
    }
}

impl<T, A: Allocator> IntoIterator for RawTable<T, A> {
    type Item = T;
    type IntoIter = RawIntoIter<T, A>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn into_iter(self) -> RawIntoIter<T, A> {
        unsafe {
            let iter = self.iter();
            self.into_iter_from(iter)
        }
    }
}

/// Find the previous power of 2. If it's already a power of 2, it's unchanged.
/// Passing zero is undefined behavior.
pub(crate) fn prev_pow2(z: usize) -> usize {
    let shift = usize::BITS as usize - 1;
    1 << (shift - (z.leading_zeros() as usize))
}

/// Iterator over a sub-range of a table. Unlike `RawIter` this iterator does
/// not track an item count.
pub(crate) struct RawIterRange<T> {
    // Mask of full buckets in the current group. Bits are cleared from this
    // mask as each element is processed.
    current_group: BitMaskIter,

    // Pointer to the buckets for the current group.
    data: Bucket<T>,

    // Pointer to the next group of control bytes,
    // Must be aligned to the group size.
    next_ctrl: *const u8,

    // Pointer one past the last control byte of this range.
    end: *const u8,
}

impl<T> RawIterRange<T> {
    /// Returns a `RawIterRange` covering a subset of a table.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is
    /// [`undefined behavior`]:
    ///
    /// * `ctrl` must be valid for reads, i.e. table outlives the `RawIterRange`;
    ///
    /// * `ctrl` must be properly aligned to the group size (`Group::WIDTH`);
    ///
    /// * `ctrl` must point to the array of properly initialized control bytes;
    ///
    /// * `data` must be the [`Bucket`] at the `ctrl` index in the table;
    ///
    /// * the value of `len` must be less than or equal to the number of table buckets,
    ///   and the returned value of `ctrl.as_ptr().add(len).offset_from(ctrl.as_ptr())`
    ///   must be positive.
    ///
    /// * The `ctrl.add(len)` pointer must be either in bounds or one
    ///   byte past the end of the same [allocated table].
    ///
    /// * The `len` must be a power of two.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn new(ctrl: *const u8, data: Bucket<T>, len: usize) -> Self {
        debug_assert_ne!(len, 0);
        debug_assert!(ctrl.cast::<Group>().is_aligned());
        // SAFETY: The caller must uphold the safety rules for the [`RawIterRange::new`]
        let end = unsafe { ctrl.add(len) };

        // Load the first group and advance ctrl to point to the next group
        // SAFETY: The caller must uphold the safety rules for the [`RawIterRange::new`]
        let (current_group, next_ctrl) = unsafe {
            (
                Group::load_aligned(ctrl.cast()).match_full(),
                ctrl.add(Group::WIDTH),
            )
        };

        Self {
            current_group: current_group.into_iter(),
            data,
            next_ctrl,
            end,
        }
    }

    /// Splits a `RawIterRange` into two halves.
    ///
    /// Returns `None` if the remaining range is smaller than or equal to the
    /// group width.
    #[cfg_attr(feature = "inline-more", inline)]
    #[cfg(feature = "rayon")]
    pub(crate) fn split(mut self) -> (Self, Option<RawIterRange<T>>) {
        unsafe {
            if self.end <= self.next_ctrl {
                // Nothing to split if the group that we are current processing
                // is the last one.
                (self, None)
            } else {
                // len is the remaining number of elements after the group that
                // we are currently processing. It must be a multiple of the
                // group size (small tables are caught by the check above).
                let len = offset_from(self.end, self.next_ctrl);
                debug_assert_eq!(len % Group::WIDTH, 0);

                // Split the remaining elements into two halves, but round the
                // midpoint down in case there is an odd number of groups
                // remaining. This ensures that:
                // - The tail is at least 1 group long.
                // - The split is roughly even considering we still have the
                //   current group to process.
                let mid = (len / 2) & !(Group::WIDTH - 1);

                let tail = Self::new(
                    self.next_ctrl.add(mid),
                    self.data.next_n(Group::WIDTH).next_n(mid),
                    len - mid,
                );
                debug_assert_eq!(
                    self.data.next_n(Group::WIDTH).next_n(mid).ptr,
                    tail.data.ptr
                );
                debug_assert_eq!(self.end, tail.end);
                self.end = self.next_ctrl.add(mid);
                debug_assert_eq!(self.end.add(Group::WIDTH), tail.next_ctrl);
                (self, Some(tail))
            }
        }
    }

    /// # Safety
    /// If `DO_CHECK_PTR_RANGE` is false, caller must ensure that we never try to iterate
    /// after yielding all elements.
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn next_impl<const DO_CHECK_PTR_RANGE: bool>(&mut self) -> Option<Bucket<T>> {
        loop {
            if let Some(index) = self.current_group.next() {
                return Some(unsafe { self.data.next_n(index) });
            }

            if DO_CHECK_PTR_RANGE && self.next_ctrl >= self.end {
                return None;
            }

            // We might read past self.end up to the next group boundary,
            // but this is fine because it only occurs on tables smaller
            // than the group size where the trailing control bytes are all
            // EMPTY. On larger tables self.end is guaranteed to be aligned
            // to the group size (since tables are power-of-two sized).
            unsafe {
                self.current_group = Group::load_aligned(self.next_ctrl.cast())
                    .match_full()
                    .into_iter();
                self.data = self.data.next_n(Group::WIDTH);
                self.next_ctrl = self.next_ctrl.add(Group::WIDTH);
            }
        }
    }

    /// Folds every element into an accumulator by applying an operation,
    /// returning the final result.
    ///
    /// `fold_impl()` takes three arguments: the number of items remaining in
    /// the iterator, an initial value, and a closure with two arguments: an
    /// 'accumulator', and an element. The closure returns the value that the
    /// accumulator should have for the next iteration.
    ///
    /// The initial value is the value the accumulator will have on the first call.
    ///
    /// After applying this closure to every element of the iterator, `fold_impl()`
    /// returns the accumulator.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is
    /// [`Undefined Behavior`]:
    ///
    /// * The [`RawTableInner`] / [`RawTable`] must be alive and not moved,
    ///   i.e. table outlives the `RawIterRange`;
    ///
    /// * The provided `n` value must match the actual number of items
    ///   in the table.
    ///
    /// [`Undefined Behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[expect(clippy::while_let_on_iterator)]
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn fold_impl<F, B>(mut self, mut n: usize, mut acc: B, mut f: F) -> B
    where
        F: FnMut(B, Bucket<T>) -> B,
    {
        loop {
            while let Some(index) = self.current_group.next() {
                // The returned `index` will always be in the range `0..Group::WIDTH`,
                // so that calling `self.data.next_n(index)` is safe (see detailed explanation below).
                debug_assert!(n != 0);
                let bucket = unsafe { self.data.next_n(index) };
                acc = f(acc, bucket);
                n -= 1;
            }

            if n == 0 {
                return acc;
            }

            // SAFETY: The caller of this function ensures that:
            //
            // 1. The provided `n` value matches the actual number of items in the table;
            // 2. The table is alive and did not moved.
            //
            // Taking the above into account, we always stay within the bounds, because:
            //
            // 1. For tables smaller than the group width (self.num_buckets() <= Group::WIDTH),
            //    we will never end up in the given branch, since we should have already
            //    yielded all the elements of the table.
            //
            // 2. For tables larger than the group width. The number of buckets is a
            //    power of two (2 ^ n), Group::WIDTH is also power of two (2 ^ k). Since
            //    `(2 ^ n) > (2 ^ k)`, than `(2 ^ n) % (2 ^ k) = 0`. As we start from the
            //    start of the array of control bytes, and never try to iterate after
            //    getting all the elements, the last `self.current_group` will read bytes
            //    from the `self.num_buckets() - Group::WIDTH` index.  We know also that
            //    `self.current_group.next()` will always return indices within the range
            //    `0..Group::WIDTH`.
            //
            //    Knowing all of the above and taking into account that we are synchronizing
            //    the `self.data` index with the index we used to read the `self.current_group`,
            //    the subsequent `self.data.next_n(index)` will always return a bucket with
            //    an index number less than `self.num_buckets()`.
            //
            //    The last `self.next_ctrl`, whose index would be `self.num_buckets()`, will never
            //    actually be read, since we should have already yielded all the elements of
            //    the table.
            unsafe {
                self.current_group = Group::load_aligned(self.next_ctrl.cast())
                    .match_full()
                    .into_iter();
                self.data = self.data.next_n(Group::WIDTH);
                self.next_ctrl = self.next_ctrl.add(Group::WIDTH);
            }
        }
    }
}

// We make raw iterators unconditionally Send and Sync, and let the PhantomData
// in the actual iterator implementations determine the real Send/Sync bounds.
unsafe impl<T> Send for RawIterRange<T> {}

unsafe impl<T> Sync for RawIterRange<T> {}

impl<T> Clone for RawIterRange<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            next_ctrl: self.next_ctrl,
            current_group: self.current_group.clone(),
            end: self.end,
        }
    }
}

impl<T> Iterator for RawIterRange<T> {
    type Item = Bucket<T>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<Bucket<T>> {
        unsafe {
            // SAFETY: We set checker flag to true.
            self.next_impl::<true>()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // We don't have an item count, so just guess based on the range size.
        let remaining_buckets = if self.end > self.next_ctrl {
            unsafe { offset_from(self.end, self.next_ctrl) }
        } else {
            0
        };

        // Add a group width to include the group we are currently processing.
        (0, Some(Group::WIDTH + remaining_buckets))
    }
}

impl<T> FusedIterator for RawIterRange<T> {}

/// Iterator which returns a raw pointer to every full bucket in the table.
///
/// For maximum flexibility this iterator is not bound by a lifetime, but you
/// must observe several rules when using it:
/// - You must not free the hash table while iterating (including via growing/shrinking).
/// - It is fine to erase a bucket that has been yielded by the iterator.
/// - Erasing a bucket that has not yet been yielded by the iterator may still
///   result in the iterator yielding that bucket (unless `reflect_remove` is called).
/// - It is unspecified whether an element inserted after the iterator was
///   created will be yielded by that iterator (unless `reflect_insert` is called).
/// - The order in which the iterator yields bucket is unspecified and may
///   change in the future.
pub(crate) struct RawIter<T> {
    pub(crate) iter: RawIterRange<T>,
    items: usize,
}

impl<T> RawIter<T> {
    unsafe fn drop_elements(&mut self) {
        unsafe {
            if T::NEEDS_DROP && self.items != 0 {
                for item in self {
                    item.drop();
                }
            }
        }
    }
}

impl<T> Clone for RawIter<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            items: self.items,
        }
    }
}

impl<T> Default for RawIter<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn default() -> Self {
        // SAFETY: Because the table is static, it always outlives the iter.
        unsafe { RawTableInner::NEW.iter() }
    }
}

impl<T> Iterator for RawIter<T> {
    type Item = Bucket<T>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<Bucket<T>> {
        // Inner iterator iterates over buckets
        // so it can do unnecessary work if we already yielded all items.
        if self.items == 0 {
            return None;
        }

        let nxt = unsafe {
            // SAFETY: We check number of items to yield using `items` field.
            self.iter.next_impl::<false>()
        };

        debug_assert!(nxt.is_some());
        self.items -= 1;

        nxt
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.items, Some(self.items))
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        unsafe { self.iter.fold_impl(self.items, init, f) }
    }
}

impl<T> ExactSizeIterator for RawIter<T> {}

impl<T> FusedIterator for RawIter<T> {}

/// Iterator which returns an index of every full bucket in the table.
///
/// For maximum flexibility this iterator is not bound by a lifetime, but you
/// must observe several rules when using it:
/// - You must not free the hash table while iterating (including via growing/shrinking).
/// - It is fine to erase a bucket that has been yielded by the iterator.
/// - Erasing a bucket that has not yet been yielded by the iterator may still
///   result in the iterator yielding index of that bucket.
/// - It is unspecified whether an element inserted after the iterator was
///   created will be yielded by that iterator.
/// - The order in which the iterator yields indices of the buckets is unspecified
///   and may change in the future.
#[derive(Clone)]
pub(crate) struct FullBucketsIndices {
    // Mask of full buckets in the current group. Bits are cleared from this
    // mask as each element is processed.
    current_group: BitMaskIter,

    // Initial value of the bytes' indices of the current group (relative
    // to the start of the control bytes).
    group_first_index: usize,

    // Pointer to the current group of control bytes,
    // Must be aligned to the group size (Group::WIDTH).
    ctrl: NonNull<u8>,

    // Number of elements in the table.
    items: usize,
}

impl Default for FullBucketsIndices {
    #[cfg_attr(feature = "inline-more", inline)]
    fn default() -> Self {
        // SAFETY: Because the table is static, it always outlives the iter.
        unsafe { RawTableInner::NEW.full_buckets_indices() }
    }
}

impl FullBucketsIndices {
    /// Advances the iterator and returns the next value.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is
    /// [`Undefined Behavior`]:
    ///
    /// * The [`RawTableInner`] / [`RawTable`] must be alive and not moved,
    ///   i.e. table outlives the `FullBucketsIndices`;
    ///
    /// * It never tries to iterate after getting all elements.
    ///
    /// [`Undefined Behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline(always)]
    unsafe fn next_impl(&mut self) -> Option<usize> {
        loop {
            if let Some(index) = self.current_group.next() {
                // The returned `self.group_first_index + index` will always
                // be in the range `0..self.num_buckets()`. See explanation below.
                return Some(self.group_first_index + index);
            }

            // SAFETY: The caller of this function ensures that:
            //
            // 1. It never tries to iterate after getting all the elements;
            // 2. The table is alive and did not moved;
            // 3. The first `self.ctrl` pointed to the start of the array of control bytes.
            //
            // Taking the above into account, we always stay within the bounds, because:
            //
            // 1. For tables smaller than the group width (self.num_buckets() <= Group::WIDTH),
            //    we will never end up in the given branch, since we should have already
            //    yielded all the elements of the table.
            //
            // 2. For tables larger than the group width. The number of buckets is a
            //    power of two (2 ^ n), Group::WIDTH is also power of two (2 ^ k). Since
            //    `(2 ^ n) > (2 ^ k)`, than `(2 ^ n) % (2 ^ k) = 0`. As we start from the
            //    the start of the array of control bytes, and never try to iterate after
            //    getting all the elements, the last `self.ctrl` will be equal to
            //    the `self.num_buckets() - Group::WIDTH`, so `self.current_group.next()`
            //    will always contains indices within the range `0..Group::WIDTH`,
            //    and subsequent `self.group_first_index + index` will always return a
            //    number less than `self.num_buckets()`.
            unsafe {
                self.ctrl = NonNull::new_unchecked(self.ctrl.as_ptr().add(Group::WIDTH));
            }

            // SAFETY: See explanation above.
            unsafe {
                self.current_group = Group::load_aligned(self.ctrl.as_ptr().cast())
                    .match_full()
                    .into_iter();
                self.group_first_index += Group::WIDTH;
            }
        }
    }
}

impl Iterator for FullBucketsIndices {
    type Item = usize;

    /// Advances the iterator and returns the next value. It is up to
    /// the caller to ensure that the `RawTable` outlives the `FullBucketsIndices`,
    /// because we cannot make the `next` method unsafe.
    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        // Return if we already yielded all items.
        if self.items == 0 {
            return None;
        }

        // SAFETY:
        // 1. We check number of items to yield using `items` field.
        // 2. The caller ensures that the table is alive and has not moved.
        let nxt = unsafe { self.next_impl() };

        debug_assert!(nxt.is_some());
        self.items -= 1;

        nxt
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.items, Some(self.items))
    }
}

impl ExactSizeIterator for FullBucketsIndices {}

impl FusedIterator for FullBucketsIndices {}

/// Iterator which consumes a table and returns elements.
pub(crate) struct RawIntoIter<T, A: Allocator = Global> {
    iter: RawIter<T>,
    allocation: Option<(NonNull<u8>, Layout, A)>,
    marker: PhantomData<T>,
}

impl<T, A: Allocator> RawIntoIter<T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn iter(&self) -> RawIter<T> {
        self.iter.clone()
    }
}

unsafe impl<T, A: Allocator> Send for RawIntoIter<T, A>
where
    T: Send,
    A: Send,
{
}

unsafe impl<T, A: Allocator> Sync for RawIntoIter<T, A>
where
    T: Sync,
    A: Sync,
{
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] T, A: Allocator> Drop for RawIntoIter<T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        unsafe {
            // Drop all remaining elements
            self.iter.drop_elements();

            // Free the table
            if let Some((ptr, layout, ref alloc)) = self.allocation {
                alloc.deallocate(ptr, layout);
            }
        }
    }
}

#[cfg(not(feature = "nightly"))]
impl<T, A: Allocator> Drop for RawIntoIter<T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        unsafe {
            // Drop all remaining elements
            self.iter.drop_elements();

            // Free the table
            if let Some((ptr, layout, ref alloc)) = self.allocation {
                alloc.deallocate(ptr, layout);
            }
        }
    }
}

impl<T, A: Allocator> Default for RawIntoIter<T, A> {
    fn default() -> Self {
        Self {
            iter: Default::default(),
            allocation: None,
            marker: PhantomData,
        }
    }
}

impl<T, A: Allocator> Iterator for RawIntoIter<T, A> {
    type Item = T;

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<T> {
        unsafe { Some(self.iter.next()?.read()) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, A: Allocator> ExactSizeIterator for RawIntoIter<T, A> {}

impl<T, A: Allocator> FusedIterator for RawIntoIter<T, A> {}

/// Iterator which consumes elements without freeing the table storage.
pub(crate) struct RawDrain<'a, T, A: Allocator = Global> {
    iter: RawIter<T>,

    // The table is moved into the iterator for the duration of the drain. This
    // ensures that an empty table is left if the drain iterator is leaked
    // without dropping.
    table: RawTableInner,
    orig_table: NonNull<RawTableInner>,

    // We don't use a &'a mut RawTable<T> because we want RawDrain to be
    // covariant over T.
    marker: PhantomData<&'a RawTable<T, A>>,
}

impl<T, A: Allocator> RawDrain<'_, T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn iter(&self) -> RawIter<T> {
        self.iter.clone()
    }
}

unsafe impl<T, A: Allocator> Send for RawDrain<'_, T, A>
where
    T: Send,
    A: Send,
{
}

unsafe impl<T, A: Allocator> Sync for RawDrain<'_, T, A>
where
    T: Sync,
    A: Sync,
{
}

impl<T, A: Allocator> Drop for RawDrain<'_, T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        unsafe {
            // Drop all remaining elements. Note that this may panic.
            self.iter.drop_elements();

            // Reset the contents of the table now that all elements have been
            // dropped.
            self.table.clear_no_drop();

            // Move the now empty table back to its original location.
            self.orig_table
                .as_ptr()
                .copy_from_nonoverlapping(&raw const self.table, 1);
        }
    }
}

impl<T, A: Allocator> Iterator for RawDrain<'_, T, A> {
    type Item = T;

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<T> {
        unsafe {
            let item = self.iter.next()?;
            Some(item.read())
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T, A: Allocator> ExactSizeIterator for RawDrain<'_, T, A> {}

impl<T, A: Allocator> FusedIterator for RawDrain<'_, T, A> {}

/// Iterator over occupied buckets that could match a given hash.
///
/// `RawTable` only stores 7 bits of the hash value, so this iterator may return
/// items that have a hash value different than the one provided. You should
/// always validate the returned values before using them.
///
/// For maximum flexibility this iterator is not bound by a lifetime, but you
/// must observe several rules when using it:
/// - You must not free the hash table while iterating (including via growing/shrinking).
/// - It is fine to erase a bucket that has been yielded by the iterator.
/// - Erasing a bucket that has not yet been yielded by the iterator may still
///   result in the iterator yielding that bucket.
/// - It is unspecified whether an element inserted after the iterator was
///   created will be yielded by that iterator.
/// - The order in which the iterator yields buckets is unspecified and may
///   change in the future.
pub(crate) struct RawIterHash<T> {
    inner: RawIterHashIndices,
    _marker: PhantomData<T>,
}

#[derive(Clone)]
pub(crate) struct RawIterHashIndices {
    // See `RawTableInner`'s corresponding fields for details.
    // We can't store a `*const RawTableInner` as it would get
    // invalidated by the user calling `&mut` methods on `RawTable`.
    ctrl: NonNull<u8>,

    // The top 7 bits of the hash.
    tag_hash: Tag,

    // The cursor producing the sequence of groups to probe in the search.
    cursor: ProbeCursor,

    group: Group,

    // The elements within the group with a matching tag-hash.
    bitmask: BitMaskIter,
}

impl<T> RawIterHash<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn new<A: Allocator>(table: &RawTable<T, A>, hash: u64) -> Self {
        RawIterHash {
            inner: unsafe { RawIterHashIndices::new(&table.table, hash) },
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for RawIterHash<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T> Default for RawIterHash<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn default() -> Self {
        Self {
            inner: RawIterHashIndices::default(),
            _marker: PhantomData,
        }
    }
}

impl Default for RawIterHashIndices {
    #[cfg_attr(feature = "inline-more", inline)]
    fn default() -> Self {
        // SAFETY: Because the table is static, it always outlives the iter.
        unsafe { RawIterHashIndices::new(&RawTableInner::NEW, 0) }
    }
}

impl RawIterHashIndices {
    #[cfg_attr(feature = "inline-more", inline)]
    unsafe fn new(table: &RawTableInner, hash: u64) -> Self {
        let tag_hash = Tag::full(hash);
        let cursor = table.probe_cursor(hash);
        let group = unsafe { Group::load(table.ctrl(cursor.pos())) };
        let bitmask = group.match_tag(tag_hash).into_iter();

        RawIterHashIndices {
            ctrl: table.ctrl,
            tag_hash,
            cursor,
            group,
            bitmask,
        }
    }
}

impl<T> Iterator for RawIterHash<T> {
    type Item = Bucket<T>;

    fn next(&mut self) -> Option<Bucket<T>> {
        unsafe {
            match self.inner.next() {
                Some(index) => {
                    // Can't use `RawTable::bucket` here as we don't have
                    // an actual `RawTable` reference to use.
                    debug_assert!(index <= self.inner.cursor.bucket_mask());
                    let bucket = Bucket::from_base_index(self.inner.ctrl.cast(), index);
                    Some(bucket)
                }
                None => None,
            }
        }
    }
}

impl Iterator for RawIterHashIndices {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            loop {
                if let Some(bit) = self.bitmask.next() {
                    let index = (self.cursor.pos() + bit) & self.cursor.bucket_mask();
                    return Some(index);
                }
                if likely(ProbeCursor::is_terminal(&self.group)) {
                    return None;
                }
                self.cursor.move_next();

                // Can't use `RawTableInner::ctrl` here as we don't have
                // an actual `RawTableInner` reference to use.
                let index = self.cursor.pos();
                debug_assert!(index < self.cursor.bucket_mask() + 1 + Group::WIDTH);
                let group_ctrl = self.ctrl.as_ptr().add(index).cast();

                self.group = Group::load(group_ctrl);
                self.bitmask = self.group.match_tag(self.tag_hash).into_iter();
            }
        }
    }
}

pub(crate) struct RawExtractIf<'a, T, A: Allocator> {
    pub iter: RawIter<T>,
    pub table: &'a mut RawTable<T, A>,
}

impl<T, A: Allocator> RawExtractIf<'_, T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn next<F>(&mut self, mut f: F) -> Option<T>
    where
        F: FnMut(&mut T) -> bool,
    {
        unsafe {
            for item in &mut self.iter {
                if f(item.as_mut()) {
                    return Some(self.table.remove(item).0);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod test_map {
    use super::*;
    use crate::control::TagSliceExt;

    #[test]
    fn test_prev_pow2() {
        // Skip 0, not defined for that input.
        let mut pow2: usize = 1;
        while (pow2 << 1) > 0 {
            let next_pow2 = pow2 << 1;
            assert_eq!(pow2, prev_pow2(pow2));
            // Need to skip 2, because it's also a power of 2, so it doesn't
            // return the previous power of 2.
            if next_pow2 > 2 {
                assert_eq!(pow2, prev_pow2(pow2 + 1));
                assert_eq!(pow2, prev_pow2(next_pow2 - 1));
            }
            pow2 = next_pow2;
        }
    }

    #[test]
    fn test_minimum_capacity_for_small_types() {
        #[track_caller]
        fn test_t<T>() {
            let raw_table: RawTable<T> = RawTable::with_capacity(1);
            let actual_buckets = raw_table.num_buckets();
            let min_buckets = Group::WIDTH / size_of::<T>();
            assert!(
                actual_buckets >= min_buckets,
                "expected at least {min_buckets} buckets, got {actual_buckets} buckets"
            );
        }

        test_t::<u8>();

        // This is only "small" for some platforms, like x86_64 with SSE2, but
        // there's no harm in running it on other platforms.
        test_t::<u16>();
    }

    fn rehash_in_place<T>(table: &mut RawTable<T>, hasher: impl Fn(&T) -> u64) {
        unsafe {
            table.table.rehash_in_place(
                &|table, index| hasher(table.bucket::<T>(index).as_ref()),
                size_of::<T>(),
                if mem::needs_drop::<T>() {
                    Some(|ptr| ptr::drop_in_place(ptr.cast::<T>()))
                } else {
                    None
                },
            );
        }
    }

    #[test]
    fn rehash() {
        let mut table = RawTable::new();
        let hasher = |i: &u64| *i;
        for i in 0..100 {
            table.insert(i, i, hasher);
        }

        for i in 0..100 {
            unsafe {
                assert_eq!(table.find(i, |x| *x == i).map(|b| b.read()), Some(i));
            }
            assert!(table.find(i + 100, |x| *x == i + 100).is_none());
        }

        rehash_in_place(&mut table, hasher);

        for i in 0..100 {
            unsafe {
                assert_eq!(table.find(i, |x| *x == i).map(|b| b.read()), Some(i));
            }
            assert!(table.find(i + 100, |x| *x == i + 100).is_none());
        }
    }

    /// CHECKING THAT WE ARE NOT TRYING TO READ THE MEMORY OF
    /// AN UNINITIALIZED TABLE DURING THE DROP
    #[test]
    fn test_drop_uninitialized() {
        use stdalloc::vec::Vec;

        let table = unsafe {
            // SAFETY: The `buckets` is power of two and we're not
            // trying to actually use the returned RawTable.
            RawTable::<(u64, Vec<i32>)>::new_uninitialized(Global, 8, Fallibility::Infallible)
                .unwrap()
        };
        drop(table);
    }

    /// CHECKING THAT WE DON'T TRY TO DROP DATA IF THE `ITEMS`
    /// ARE ZERO, EVEN IF WE HAVE `FULL` CONTROL BYTES.
    #[test]
    fn test_drop_zero_items() {
        use stdalloc::vec::Vec;
        unsafe {
            // SAFETY: The `buckets` is power of two and we're not
            // trying to actually use the returned RawTable.
            let mut table =
                RawTable::<(u64, Vec<i32>)>::new_uninitialized(Global, 8, Fallibility::Infallible)
                    .unwrap();

            // WE SIMULATE, AS IT WERE, A FULL TABLE.

            // SAFETY: We checked that the table is allocated and therefore the table already has
            // `self.bucket_mask + 1 + Group::WIDTH` number of control bytes (see TableLayout::calculate_layout_for)
            // so writing `table.table.num_ctrl_bytes() == bucket_mask + 1 + Group::WIDTH` bytes is safe.
            table.table.ctrl_slice().fill_empty();

            // SAFETY: table.capacity() is guaranteed to be smaller than table.num_buckets()
            table.table.ctrl(0).write_bytes(0, table.capacity());

            // Fix up the trailing control bytes. See the comments in set_ctrl
            // for the handling of tables smaller than the group width.
            if table.num_buckets() < Group::WIDTH {
                // SAFETY: We have `self.bucket_mask + 1 + Group::WIDTH` number of control bytes,
                // so copying `self.num_buckets() == self.bucket_mask + 1` bytes with offset equal to
                // `Group::WIDTH` is safe
                table
                    .table
                    .ctrl(0)
                    .copy_to(table.table.ctrl(Group::WIDTH), table.table.num_buckets());
            } else {
                // SAFETY: We have `self.bucket_mask + 1 + Group::WIDTH` number of
                // control bytes,so copying `Group::WIDTH` bytes with offset equal
                // to `self.num_buckets() == self.bucket_mask + 1` is safe
                table
                    .table
                    .ctrl(0)
                    .copy_to(table.table.ctrl(table.table.num_buckets()), Group::WIDTH);
            }
            drop(table);
        }
    }

    /// CHECKING THAT WE DON'T TRY TO DROP DATA IF THE `ITEMS`
    /// ARE ZERO, EVEN IF WE HAVE `FULL` CONTROL BYTES.
    #[test]
    #[cfg(panic = "unwind")]
    fn test_catch_panic_clone_from() {
        use super::{AllocError, Allocator, Global};
        use core::sync::atomic::{AtomicI8, Ordering};
        use std::thread;
        use stdalloc::sync::Arc;
        use stdalloc::vec::Vec;

        struct MyAllocInner {
            drop_count: Arc<AtomicI8>,
        }

        #[derive(Clone)]
        struct MyAlloc {
            _inner: Arc<MyAllocInner>,
        }

        impl Drop for MyAllocInner {
            fn drop(&mut self) {
                println!("MyAlloc freed.");
                self.drop_count.fetch_sub(1, Ordering::SeqCst);
            }
        }

        unsafe impl Allocator for MyAlloc {
            fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
                let g = Global;
                g.allocate(layout)
            }

            unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
                unsafe {
                    let g = Global;
                    g.deallocate(ptr, layout);
                }
            }
        }

        const DISARMED: bool = false;
        const ARMED: bool = true;

        struct CheckedCloneDrop {
            panic_in_clone: bool,
            dropped: bool,
            need_drop: Vec<u64>,
        }

        impl Clone for CheckedCloneDrop {
            fn clone(&self) -> Self {
                assert!(!self.panic_in_clone, "panic in clone");
                Self {
                    panic_in_clone: self.panic_in_clone,
                    dropped: self.dropped,
                    need_drop: self.need_drop.clone(),
                }
            }
        }

        impl Drop for CheckedCloneDrop {
            fn drop(&mut self) {
                assert!(!self.dropped, "double drop");
                self.dropped = true;
            }
        }

        let dropped: Arc<AtomicI8> = Arc::new(AtomicI8::new(2));

        let mut table = RawTable::new_in(MyAlloc {
            _inner: Arc::new(MyAllocInner {
                drop_count: dropped.clone(),
            }),
        });

        for (idx, panic_in_clone) in core::iter::repeat_n(DISARMED, 7).enumerate() {
            let idx = idx as u64;
            table.insert(
                idx,
                (
                    idx,
                    CheckedCloneDrop {
                        panic_in_clone,
                        dropped: false,
                        need_drop: vec![idx],
                    },
                ),
                |(k, _)| *k,
            );
        }

        assert_eq!(table.len(), 7);

        thread::scope(|s| {
            let result = s.spawn(|| {
                let armed_flags = [
                    DISARMED, DISARMED, ARMED, DISARMED, DISARMED, DISARMED, DISARMED,
                ];
                let mut scope_table = RawTable::new_in(MyAlloc {
                    _inner: Arc::new(MyAllocInner {
                        drop_count: dropped.clone(),
                    }),
                });
                for (idx, &panic_in_clone) in armed_flags.iter().enumerate() {
                    let idx = idx as u64;
                    scope_table.insert(
                        idx,
                        (
                            idx,
                            CheckedCloneDrop {
                                panic_in_clone,
                                dropped: false,
                                need_drop: vec![idx + 100],
                            },
                        ),
                        |(k, _)| *k,
                    );
                }
                table.clone_from(&scope_table);
            });
            assert!(result.join().is_err());
        });

        // Let's check that all iterators work fine and do not return elements
        // (especially `RawIterRange`, which does not depend on the number of
        // elements in the table, but looks directly at the control bytes)
        //
        // SAFETY: We know for sure that `RawTable` will outlive
        // the returned `RawIter / RawIterRange` iterator.
        assert_eq!(table.len(), 0);
        assert_eq!(unsafe { table.iter().count() }, 0);
        assert_eq!(unsafe { table.iter().iter.count() }, 0);

        for idx in 0..table.num_buckets() {
            let idx = idx as u64;
            assert!(
                table.find(idx, |(k, _)| *k == idx).is_none(),
                "Index: {idx}"
            );
        }

        // All allocator clones should already be dropped.
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
