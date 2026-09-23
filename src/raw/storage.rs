//! Table storage: the `ctrl`/`bucket` memory layout, the mirrored control
//! bytes, capacity computation and the allocator-facing parts of the table.
//!
//! `RawTableInner` owns the physical layout of the SwissTable: a single
//! allocation holding the data buckets followed by the control bytes and
//! their `Group::WIDTH`-byte mirror. All probe logic lives in
//! [`super::probe`], all growth/rehash logic lives in [`super::growth`].

use crate::TryReserveError;
use crate::alloc::{Allocator, do_alloc};
use crate::control::{Group, Tag, TagSliceExt};
use core::alloc::Layout;
use core::mem;
use core::ptr;
use core::ptr::NonNull;
use core::slice;
use stdalloc::alloc::handle_alloc_error;

use super::{FullBucketsIndices, RawIter, RawIterRange, prev_pow2};

#[inline]
pub(super) unsafe fn offset_from<T>(to: *const T, from: *const T) -> usize {
    unsafe { to.offset_from(from) as usize }
}

/// Whether memory allocation errors should return an error or abort.
#[derive(Copy, Clone)]
pub(super) enum Fallibility {
    Fallible,
    Infallible,
}

impl Fallibility {
    /// Error to return on capacity overflow.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn capacity_overflow(self) -> TryReserveError {
        match self {
            Fallibility::Fallible => TryReserveError::CapacityOverflow,
            Fallibility::Infallible => panic!("Hash table capacity overflow"),
        }
    }

    /// Error to return on allocation error.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn alloc_err(self, layout: Layout) -> TryReserveError {
        match self {
            Fallibility::Fallible => TryReserveError::AllocError { layout },
            Fallibility::Infallible => handle_alloc_error(layout),
        }
    }
}

pub(super) trait SizedTypeProperties: Sized {
    const IS_ZERO_SIZED: bool = size_of::<Self>() == 0;
    const NEEDS_DROP: bool = mem::needs_drop::<Self>();
}

impl<T> SizedTypeProperties for T {}

/// Returns the number of buckets needed to hold the given number of items,
/// taking the maximum load factor into account.
///
/// Returns `None` if an overflow occurs.
///
/// This ensures that `buckets * table_layout.size >= table_layout.ctrl_align`.
// Workaround for emscripten bug emscripten-core/emscripten-fastcomp#258
#[cfg_attr(target_os = "emscripten", inline(never))]
#[cfg_attr(not(target_os = "emscripten"), inline)]
pub(super) fn capacity_to_buckets(cap: usize, table_layout: TableLayout) -> Option<usize> {
    debug_assert_ne!(cap, 0);

    // For small tables we require at least 1 empty bucket so that lookups are
    // guaranteed to terminate if an element doesn't exist in the table.
    if cap < 15 {
        // Consider a small TableLayout like { size: 1, ctrl_align: 16 } on a
        // platform with Group::WIDTH of 16 (like x86_64 with SSE2). For small
        // bucket sizes, this ends up wasting quite a few bytes just to pad to
        // the relatively larger ctrl_align:
        //
        // | capacity | buckets | bytes allocated | bytes per item |
        // | -------- | ------- | --------------- | -------------- |
        // |        3 |       4 |              36 | (Yikes!)  12.0 |
        // |        7 |       8 |              40 | (Poor)     5.7 |
        // |       14 |      16 |              48 |            3.4 |
        // |       28 |      32 |              80 |            3.3 |
        //
        // In general, buckets * table_layout.size >= table_layout.ctrl_align
        // must be true to avoid these edges. This is implemented by adjusting
        // the minimum capacity upwards for small items. This code only needs
        // to handle ctrl_align which are less than or equal to Group::WIDTH,
        // because valid layout sizes are always a multiple of the alignment,
        // so anything with alignment over the Group::WIDTH won't hit this edge
        // case.

        // This is brittle, e.g. if we ever add 32 byte groups, it will select
        // 3 regardless of the table_layout.size.
        let min_cap = match (Group::WIDTH, table_layout.size) {
            (16, 0..=1) => 14,
            (16, 2..=3) | (8, 0..=1) => 7,
            _ => 3,
        };
        let cap = min_cap.max(cap);
        // We don't bother with a table size of 2 buckets since that can only
        // hold a single element. Instead, we skip directly to a 4 bucket table
        // which can hold 3 elements.
        let buckets = if cap < 4 {
            4
        } else if cap < 8 {
            8
        } else {
            16
        };
        ensure_bucket_bytes_at_least_ctrl_align(table_layout, buckets);
        return Some(buckets);
    }

    // Otherwise require 1/8 buckets to be empty (87.5% load)
    //
    // Be careful when modifying this, calculate_layout relies on the
    // overflow check here.
    let adjusted_cap = cap.checked_mul(8)? / 7;

    // Any overflows will have been caught by the checked_mul. Also, any
    // rounding errors from the division above will be cleaned up by
    // next_power_of_two (which can't overflow because of the previous division).
    let buckets = adjusted_cap.next_power_of_two();
    ensure_bucket_bytes_at_least_ctrl_align(table_layout, buckets);
    Some(buckets)
}

// `maximum_buckets_in` relies on the property that for non-ZST `T`, any
// chosen `buckets` will satisfy `buckets * table_layout.size >=
// table_layout.ctrl_align`, so `calculate_layout_for` does not need to add
// extra padding beyond `table_layout.size * buckets`. If small-table bucket
// selection or growth policy changes, revisit `maximum_buckets_in`.
#[inline]
fn ensure_bucket_bytes_at_least_ctrl_align(table_layout: TableLayout, buckets: usize) {
    if table_layout.size != 0 {
        let prod = table_layout.size.saturating_mul(buckets);
        debug_assert!(prod >= table_layout.ctrl_align);
    }
}

/// Returns the maximum effective capacity for the given bucket mask, taking
/// the maximum load factor into account.
#[inline]
pub(super) fn bucket_mask_to_capacity(bucket_mask: usize) -> usize {
    if bucket_mask < 8 {
        // For tables with 1/2/4/8 buckets, we always reserve one empty slot.
        // Keep in mind that the bucket mask is one less than the bucket count.
        bucket_mask
    } else {
        // `bucket_mask` is bounded by the maximum allocation size, so it can
        // never be `usize::MAX` and the `+ 1` below cannot overflow.
        debug_assert!(bucket_mask != usize::MAX);
        // For larger tables we reserve 12.5% of the slots as empty.
        ((bucket_mask + 1) / 8) * 7
    }
}

/// Helper which allows the max calculation for `ctrl_align` to be statically computed for each `T`
/// while keeping the rest of `calculate_layout_for` independent of `T`
#[derive(Copy, Clone)]
pub(super) struct TableLayout {
    pub(super) size: usize,
    pub(super) ctrl_align: usize,
}

impl TableLayout {
    #[inline]
    pub(super) const fn new<T>() -> Self {
        let layout = Layout::new::<T>();
        Self {
            size: layout.size(),
            ctrl_align: if layout.align() > Group::WIDTH {
                layout.align()
            } else {
                Group::WIDTH
            },
        }
    }

    #[inline]
    pub(super) fn calculate_layout_for(self, buckets: usize) -> Option<(Layout, usize)> {
        debug_assert!(buckets.is_power_of_two());

        let TableLayout { size, ctrl_align } = self;
        // Manual layout calculation since Layout methods are not yet stable.
        let ctrl_offset =
            size.checked_mul(buckets)?.checked_add(ctrl_align - 1)? & !(ctrl_align - 1);
        let len = ctrl_offset.checked_add(buckets + Group::WIDTH)?;

        // We need an additional check to ensure that the allocation doesn't
        // exceed `isize::MAX` (https://github.com/rust-lang/rust/pull/95295).
        if len > isize::MAX as usize - (ctrl_align - 1) {
            return None;
        }

        Some((
            unsafe { Layout::from_size_align_unchecked(len, ctrl_align) },
            ctrl_offset,
        ))
    }
}

/// A reference to a hash table bucket containing a `T`.
///
/// This is usually just a pointer to the element itself. However if the element
/// is a ZST, then we instead track the index of the element in the table so
/// that `erase` works properly.
pub(crate) struct Bucket<T> {
    // Actually it is pointer to next element than element itself
    // this is needed to maintain pointer arithmetic invariants
    // keeping direct pointer to element introduces difficulty.
    // Using `NonNull` for variance and niche layout
    pub(super) ptr: NonNull<T>,
}

// This Send impl is needed for rayon support. This is safe since Bucket is
// never exposed in a public API.
unsafe impl<T> Send for Bucket<T> {}

impl<T> Clone for Bucket<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<T> Bucket<T> {
    /// Creates a [`Bucket`] that contain pointer to the data.
    /// The pointer calculation is performed by calculating the
    /// offset from given `base` pointer (convenience for
    /// `base.as_ptr().sub(index)`).
    ///
    /// `index` is in units of `T`; e.g., an `index` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// If the `T` is a ZST, then we instead track the index of the element
    /// in the table so that `erase` works properly (return
    /// `NonNull::new_unchecked((index + 1) as *mut T)`)
    ///
    /// # Safety
    ///
    /// If `size_of::<T>() != 0`, then the safety rules are directly derived
    /// from the safety rules for [`<*mut T>::sub`] method of `*mut T` and the safety
    /// rules of [`NonNull::new_unchecked`] function.
    ///
    /// Thus, in order to uphold the safety contracts for the [`<*mut T>::sub`] method
    /// and [`NonNull::new_unchecked`] function, as well as for the correct
    /// logic of the work of this crate, the following rules are necessary and
    /// sufficient:
    ///
    /// * the `base` pointer must not be `dangling` and must points to the
    ///   end of the first `value element` from the `data part` of the table, i.e.
    ///   must be the pointer that returned by [`RawTable::data_end`] or by
    ///   [`RawTableInner::data_end<T>`];
    ///
    /// * `index` must not be greater than `RawTableInner.bucket_mask`, i.e.
    ///   `index <= RawTableInner.bucket_mask` or, in other words, `(index + 1)`
    ///   must be no greater than the number returned by the function
    ///   [`RawTable::num_buckets`] or [`RawTableInner::num_buckets`].
    ///
    /// If `size_of::<T>() == 0`, then the only requirement is that the
    /// `index` must not be greater than `RawTableInner.bucket_mask`, i.e.
    /// `index <= RawTableInner.bucket_mask` or, in other words, `(index + 1)`
    /// must be no greater than the number returned by the function
    /// [`RawTable::num_buckets`] or [`RawTableInner::num_buckets`].
    #[inline]
    pub(super) unsafe fn from_base_index(base: NonNull<T>, index: usize) -> Self {
        // If size_of::<T>() != 0 then return a pointer to an `element` in
        // the data part of the table (we start counting from "0", so that
        // in the expression T[last], the "last" index actually one less than the
        // "buckets" number in the table, i.e. "last = RawTableInner.bucket_mask"):
        //
        //                   `from_base_index(base, 1).as_ptr()` returns a pointer that
        //                   points here in the data part of the table
        //                   (to the start of T1)
        //                        |
        //                        |        `base: NonNull<T>` must point here
        //                        |         (to the end of T0 or to the start of C0)
        //                        v         v
        // [Padding], Tlast, ..., |T1|, T0, |C0, C1, ..., Clast
        //                           ^
        //                           `from_base_index(base, 1)` returns a pointer
        //                           that points here in the data part of the table
        //                           (to the end of T1)
        //
        // where: T0...Tlast - our stored data; C0...Clast - control bytes
        // or metadata for data.
        let ptr = if T::IS_ZERO_SIZED {
            // won't overflow because index must be less than length (bucket_mask)
            // and bucket_mask is guaranteed to be less than `isize::MAX`
            // (see TableLayout::calculate_layout_for method)
            ptr::without_provenance_mut(index + 1)
        } else {
            unsafe { base.as_ptr().sub(index) }
        };
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
        }
    }

    /// Calculates the index of a [`Bucket`] as distance between two pointers
    /// (convenience for `base.as_ptr().offset_from(self.ptr.as_ptr()) as usize`).
    /// The returned value is in units of T: the distance in bytes divided by
    /// [`size_of::<T>()`].
    ///
    /// If the `T` is a ZST, then we return the index of the element in
    /// the table so that `erase` works properly (return `self.ptr.as_ptr() as usize - 1`).
    ///
    /// This function is the inverse of [`from_base_index`].
    ///
    /// # Safety
    ///
    /// If `size_of::<T>() != 0`, then the safety rules are directly derived
    /// from the safety rules for [`<*const T>::offset_from`] method of `*const T`.
    ///
    /// Thus, in order to uphold the safety contracts for [`<*const T>::offset_from`]
    /// method, as well as for the correct logic of the work of this crate, the
    /// following rules are necessary and sufficient:
    ///
    /// * `base` contained pointer must not be `dangling` and must point to the
    ///   end of the first `element` from the `data part` of the table, i.e.
    ///   must be a pointer that returns by [`RawTable::data_end`] or by
    ///   [`RawTableInner::data_end<T>`];
    ///
    /// * `self` also must not contain dangling pointer;
    ///
    /// * both `self` and `base` must be created from the same [`RawTable`]
    ///   (or [`RawTableInner`]).
    ///
    /// If `size_of::<T>() == 0`, this function is always safe.
    #[inline]
    pub(super) unsafe fn to_base_index(&self, base: NonNull<T>) -> usize {
        // If size_of::<T>() != 0 then return an index under which we used to store the
        // `element` in the data part of the table (we start counting from "0", so
        // that in the expression T[last], the "last" index actually is one less than the
        // "buckets" number in the table, i.e. "last = RawTableInner.bucket_mask").
        // For example for 5th element in table calculation is performed like this:
        //
        //                        size_of::<T>()
        //                          |
        //                          |         `self = from_base_index(base, 5)` that returns pointer
        //                          |         that points here in the data part of the table
        //                          |         (to the end of T5)
        //                          |           |                    `base: NonNull<T>` must point here
        //                          v           |                    (to the end of T0 or to the start of C0)
        //                        /???\         v                      v
        // [Padding], Tlast, ..., |T10|, ..., T5|, T4, T3, T2, T1, T0, |C0, C1, C2, C3, C4, C5, ..., C10, ..., Clast
        //                                      \__________  __________/
        //                                                 \/
        //                                     `bucket.to_base_index(base)` = 5
        //                                     (base.as_ptr() as usize - self.ptr.as_ptr() as usize) / size_of::<T>()
        //
        // where: T0...Tlast - our stored data; C0...Clast - control bytes or metadata for data.
        if T::IS_ZERO_SIZED {
            // this can not be UB
            self.ptr.as_ptr().addr() - 1
        } else {
            unsafe { offset_from(base.as_ptr(), self.ptr.as_ptr()) }
        }
    }

    /// Acquires the underlying raw pointer `*mut T` to `data`.
    ///
    /// # Note
    ///
    /// If `T` is not [`Copy`], do not use `*mut T` methods that can cause calling the
    /// destructor of `T` (for example the [`<*mut T>::drop_in_place`] method), because
    /// for properly dropping the data we also need to clear `data` control bytes. If we
    /// drop data, but do not clear `data control byte` it leads to double drop when
    /// [`RawTable`] goes out of scope.
    ///
    /// If you modify an already initialized `value`, so [`Hash`] and [`Eq`] on the new
    /// `T` value and its borrowed form *must* match those for the old `T` value, as the map
    /// will not re-evaluate where the new value should go, meaning the value may become
    /// "lost" if their location does not reflect their state.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut T {
        if T::IS_ZERO_SIZED {
            // Just return an arbitrary ZST pointer which is properly aligned
            // invalid pointer is good enough for ZST
            ptr::without_provenance_mut(align_of::<T>())
        } else {
            unsafe { self.ptr.as_ptr().sub(1) }
        }
    }

    /// Acquires the underlying non-null pointer `*mut T` to `data`.
    #[inline]
    pub(super) fn as_non_null(&self) -> NonNull<T> {
        // SAFETY: `self.ptr` is already a `NonNull`
        unsafe { NonNull::new_unchecked(self.as_ptr()) }
    }

    /// Create a new [`Bucket`] that is offset from the `self` by the given
    /// `offset`. The pointer calculation is performed by calculating the
    /// offset from `self` pointer (convenience for `self.ptr.as_ptr().sub(offset)`).
    /// This function is used for iterators.
    ///
    /// `offset` is in units of `T`; e.g., a `offset` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If `size_of::<T>() != 0`, then the safety rules are directly derived
    /// from the safety rules for [`<*mut T>::sub`] method of `*mut T` and safety
    /// rules of [`NonNull::new_unchecked`] function.
    ///
    /// Thus, in order to uphold the safety contracts for [`<*mut T>::sub`] method
    /// and [`NonNull::new_unchecked`] function, as well as for the correct
    /// logic of the work of this crate, the following rules are necessary and
    /// sufficient:
    ///
    /// * `self` contained pointer must not be `dangling`;
    ///
    /// * `self.to_base_index() + offset` must not be greater than `RawTableInner.bucket_mask`,
    ///   i.e. `(self.to_base_index() + offset) <= RawTableInner.bucket_mask` or, in other
    ///   words, `self.to_base_index() + offset + 1` must be no greater than the number returned
    ///   by the function [`RawTable::num_buckets`] or [`RawTableInner::num_buckets`].
    ///
    /// If `size_of::<T>() == 0`, then the only requirement is that the
    /// `self.to_base_index() + offset` must not be greater than `RawTableInner.bucket_mask`,
    /// i.e. `(self.to_base_index() + offset) <= RawTableInner.bucket_mask` or, in other words,
    /// `self.to_base_index() + offset + 1` must be no greater than the number returned by the
    /// function [`RawTable::num_buckets`] or [`RawTableInner::num_buckets`].
    #[inline]
    pub(super) unsafe fn next_n(&self, offset: usize) -> Self {
        let ptr = if T::IS_ZERO_SIZED {
            // invalid pointer is good enough for ZST
            ptr::without_provenance_mut(self.ptr.as_ptr().addr() + offset)
        } else {
            unsafe { self.ptr.as_ptr().sub(offset) }
        };
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
        }
    }

    /// Executes the destructor (if any) of the pointed-to `data`.
    ///
    /// # Safety
    ///
    /// See [`ptr::drop_in_place`] for safety concerns.
    ///
    /// You should use [`RawTable::erase`] instead of this function,
    /// or be careful with calling this function directly, because for
    /// properly dropping the data we need also clear `data` control bytes.
    /// If we drop data, but do not erase `data control byte` it leads to
    /// double drop when [`RawTable`] goes out of scope.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) unsafe fn drop(&self) {
        unsafe {
            self.as_ptr().drop_in_place();
        }
    }

    /// Reads the `value` from `self` without moving it. This leaves the
    /// memory in `self` unchanged.
    ///
    /// # Safety
    ///
    /// See [`ptr::read`] for safety concerns.
    ///
    /// You should use [`RawTable::remove`] instead of this function,
    /// or be careful with calling this function directly, because compiler
    /// calls its destructor when the read `value` goes out of scope. It
    /// can cause double dropping when [`RawTable`] goes out of scope,
    /// because of not erased `data control byte`.
    #[inline]
    pub(crate) unsafe fn read(&self) -> T {
        unsafe { self.as_ptr().read() }
    }

    /// Overwrites a memory location with the given `value` without reading
    /// or dropping the old value (like [`ptr::write`] function).
    ///
    /// # Safety
    ///
    /// See [`ptr::write`] for safety concerns.
    ///
    /// # Note
    ///
    /// [`Hash`] and [`Eq`] on the new `T` value and its borrowed form *must* match
    /// those for the old `T` value, as the map will not re-evaluate where the new
    /// value should go, meaning the value may become "lost" if their location
    /// does not reflect their state.
    #[inline]
    pub(crate) unsafe fn write(&self, val: T) {
        unsafe {
            self.as_ptr().write(val);
        }
    }

    /// Returns a shared immutable reference to the `value`.
    ///
    /// # Safety
    ///
    /// See [`NonNull::as_ref`] for safety concerns.
    #[inline]
    pub(crate) unsafe fn as_ref<'a>(&self) -> &'a T {
        unsafe { &*self.as_ptr() }
    }

    /// Returns a unique mutable reference to the `value`.
    ///
    /// # Safety
    ///
    /// See [`NonNull::as_mut`] for safety concerns.
    ///
    /// # Note
    ///
    /// [`Hash`] and [`Eq`] on the new `T` value and its borrowed form *must* match
    /// those for the old `T` value, as the map will not re-evaluate where the new
    /// value should go, meaning the value may become "lost" if their location
    /// does not reflect their state.
    #[inline]
    pub(crate) unsafe fn as_mut<'a>(&self) -> &'a mut T {
        unsafe { &mut *self.as_ptr() }
    }
}
/// Non-generic part of `RawTable` which allows functions to be instantiated only once regardless
/// of how many different key-value types are used.
pub(super) struct RawTableInner {
    // Mask to get an index from a hash value. The value is one less than the
    // number of buckets in the table.
    pub(super) bucket_mask: usize,

    // [Padding], T_n, ..., T1, T0, C0, C1, ...
    //                              ^ points here
    pub(super) ctrl: NonNull<u8>,

    // Number of elements that can be inserted before we need to grow the table
    pub(super) growth_left: usize,

    // Number of elements in the table, only really used by len()
    pub(super) items: usize,
}
impl RawTableInner {
    pub(super) const NEW: Self = RawTableInner::new();

    /// Creates a new empty hash table without allocating any memory.
    ///
    /// In effect this returns a table with exactly 1 bucket. However we can
    /// leave the data pointer dangling since that bucket is never accessed
    /// due to our load factor forcing us to always have at least 1 free bucket.
    #[inline]
    pub(super) const fn new() -> Self {
        Self {
            // Be careful to cast the entire slice to a raw pointer.
            ctrl: unsafe {
                NonNull::new_unchecked(Group::static_empty().as_ptr().cast_mut().cast())
            },
            bucket_mask: 0,
            items: 0,
            growth_left: 0,
        }
    }
}

/// Finds the largest number of buckets that can fit in `allocation_size`
/// provided the given TableLayout.
///
/// This relies on some invariants of `capacity_to_buckets`, so only feed in
/// an `allocation_size` calculated from `capacity_to_buckets`.
fn maximum_buckets_in(
    allocation_size: usize,
    table_layout: TableLayout,
    group_width: usize,
) -> usize {
    // Given an equation like:
    //   z >= x * y + x + g
    // x can be maximized by doing:
    //   x = (z - g) / (y + 1)
    // If you squint:
    //   x is the number of buckets
    //   y is the table_layout.size
    //   z is the size of the allocation
    //   g is the group width
    // But this is ignoring the padding needed for ctrl_align.
    // If we remember these restrictions:
    //   x is always a power of 2
    //   Layout size for T must always be a multiple of T
    // Then the alignment can be ignored if we add the constraint:
    //   x * y >= table_layout.ctrl_align
    // This is taken care of by `capacity_to_buckets`.
    // It may be helpful to understand this if you remember that:
    //   ctrl_offset = align(x * y, ctrl_align)
    let x = (allocation_size - group_width) / (table_layout.size + 1);
    prev_pow2(x)
}

impl RawTableInner {
    /// Allocates a new [`RawTableInner`] with the given number of buckets.
    /// The control bytes and buckets are left uninitialized.
    ///
    /// # Safety
    ///
    /// The caller of this function must ensure that the `buckets` is power of two
    /// and also initialize all control bytes of the length `self.bucket_mask + 1 +
    /// Group::WIDTH` with the [`Tag::EMPTY`] bytes.
    ///
    /// See also [`Allocator`] API for other safety concerns.
    ///
    /// [`Allocator`]: stdalloc::alloc::Allocator
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) unsafe fn new_uninitialized<A>(
        alloc: &A,
        table_layout: TableLayout,
        mut buckets: usize,
        fallibility: Fallibility,
    ) -> Result<Self, TryReserveError>
    where
        A: Allocator,
    {
        debug_assert!(buckets.is_power_of_two());

        // Avoid `Option::ok_or_else` because it bloats LLVM IR.
        let Some((layout, mut ctrl_offset)) = table_layout.calculate_layout_for(buckets) else {
            return Err(fallibility.capacity_overflow());
        };

        let ptr: NonNull<u8> = match do_alloc(alloc, layout) {
            Ok(block) => {
                // The allocator can't return a value smaller than was
                // requested, so this can be != instead of >=.
                if block.len() != layout.size() {
                    // Utilize over-sized allocations.
                    let x = maximum_buckets_in(block.len(), table_layout, Group::WIDTH);
                    debug_assert!(x >= buckets);
                    // Calculate the new ctrl_offset.
                    let (oversized_layout, oversized_ctrl_offset) = {
                        let option = table_layout.calculate_layout_for(x);
                        unsafe { option.unwrap_unchecked() }
                    };
                    debug_assert!(oversized_layout.size() <= block.len());
                    debug_assert!(oversized_ctrl_offset >= ctrl_offset);
                    ctrl_offset = oversized_ctrl_offset;
                    buckets = x;
                }

                block.cast()
            }
            Err(_) => return Err(fallibility.alloc_err(layout)),
        };

        // SAFETY: null pointer will be caught in above check
        let ctrl = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(ctrl_offset)) };
        Ok(Self {
            ctrl,
            bucket_mask: buckets - 1,
            items: 0,
            growth_left: bucket_mask_to_capacity(buckets - 1),
        })
    }

    /// Attempts to allocate a new [`RawTableInner`] with at least enough
    /// capacity for inserting the given number of elements without reallocating.
    ///
    /// All the control bytes are initialized with the [`Tag::EMPTY`] bytes.
    #[inline]
    pub(super) fn fallible_with_capacity<A>(
        alloc: &A,
        table_layout: TableLayout,
        capacity: usize,
        fallibility: Fallibility,
    ) -> Result<Self, TryReserveError>
    where
        A: Allocator,
    {
        if capacity == 0 {
            Ok(Self::NEW)
        } else {
            // SAFETY: We checked that we could successfully allocate the new table, and then
            // initialized all control bytes with the constant `Tag::EMPTY` byte.
            unsafe {
                let buckets = capacity_to_buckets(capacity, table_layout)
                    .ok_or_else(|| fallibility.capacity_overflow())?;

                let mut result =
                    Self::new_uninitialized(alloc, table_layout, buckets, fallibility)?;
                // SAFETY: We checked that the table is allocated and therefore the table already has
                // `self.bucket_mask + 1 + Group::WIDTH` number of control bytes (see TableLayout::calculate_layout_for)
                // so writing `self.num_ctrl_bytes() == bucket_mask + 1 + Group::WIDTH` bytes is safe.
                result.ctrl_slice().fill_empty();

                Ok(result)
            }
        }
    }

    /// Allocates a new [`RawTableInner`] with at least enough capacity for inserting
    /// the given number of elements without reallocating.
    ///
    /// Panics if the new capacity exceeds [`isize::MAX`] bytes and [`abort`] the program
    /// in case of allocation error. Use [`fallible_with_capacity`] instead if you want to
    /// handle memory allocation failure.
    ///
    /// All the control bytes are initialized with the [`Tag::EMPTY`] bytes.
    ///
    /// [`fallible_with_capacity`]: RawTableInner::fallible_with_capacity
    /// [`abort`]: stdalloc::abort::handle_alloc_error
    pub(super) fn with_capacity<A>(alloc: &A, table_layout: TableLayout, capacity: usize) -> Self
    where
        A: Allocator,
    {
        let result =
            Self::fallible_with_capacity(alloc, table_layout, capacity, Fallibility::Infallible);

        // SAFETY: All allocation errors will be caught inside `RawTableInner::new_uninitialized`.
        unsafe { result.unwrap_unchecked() }
    }

    /// Returns an iterator over every element in the table.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result
    /// is [`undefined behavior`]:
    ///
    /// * The caller has to ensure that the `RawTableInner` outlives the
    ///   `RawIter`. Because we cannot make the `next` method unsafe on
    ///   the `RawIter` struct, we have to make the `iter` method unsafe.
    ///
    /// * The [`RawTableInner`] must have properly initialized control bytes.
    ///
    /// The type `T` must be the actual type of the elements stored in the table,
    /// otherwise using the returned [`RawIter`] results in [`undefined behavior`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn iter<T>(&self) -> RawIter<T> {
        // SAFETY:
        // 1. Since the caller of this function ensures that the control bytes
        //    are properly initialized and `self.data_end()` points to the start
        //    of the array of control bytes, therefore: `ctrl` is valid for reads,
        //    properly aligned to `Group::WIDTH` and points to the properly initialized
        //    control bytes.
        // 2. `data` bucket index in the table is equal to the `ctrl` index (i.e.
        //    equal to zero).
        // 3. We pass the exact value of buckets of the table to the function.
        //
        //                         `ctrl` points here (to the start
        //                         of the first control byte `CT0`)
        //                          ∨
        // [Pad], T_n, ..., T1, T0, |CT0, CT1, ..., CT_n|, CTa_0, CTa_1, ..., CTa_m
        //                           \________  ________/
        //                                    \/
        //       `n = buckets - 1`, i.e. `RawTableInner::num_buckets() - 1`
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
        unsafe {
            let data = Bucket::from_base_index(self.data_end(), 0);
            RawIter {
                // SAFETY: See explanation above
                iter: RawIterRange::new(self.ctrl.as_ptr(), data, self.num_buckets()),
                items: self.items,
            }
        }
    }

    /// Executes the destructors (if any) of the values stored in the table.
    ///
    /// # Note
    ///
    /// This function does not erase the control bytes of the table and does
    /// not make any changes to the `items` or `growth_left` fields of the
    /// table. If necessary, the caller of this function must manually set
    /// up these table fields, for example using the [`clear_no_drop`] function.
    ///
    /// Be careful during calling this function, because drop function of
    /// the elements can panic, and this can leave table in an inconsistent
    /// state.
    ///
    /// # Safety
    ///
    /// The type `T` must be the actual type of the elements stored in the table,
    /// otherwise calling this function may result in [`undefined behavior`].
    ///
    /// If `T` is a type that should be dropped and **the table is not empty**,
    /// calling this function more than once results in [`undefined behavior`].
    ///
    /// If `T` is not [`Copy`], attempting to use values stored in the table after
    /// calling this function may result in [`undefined behavior`].
    ///
    /// It is safe to call this function on a table that has not been allocated,
    /// on a table with uninitialized control bytes, and on a table with no actual
    /// data but with `Full` control bytes if `self.items == 0`.
    ///
    /// See also [`Bucket::drop`] / [`Bucket::as_ptr`] methods, for more information
    /// about of properly removing or saving `element` from / into the [`RawTable`] /
    /// [`RawTableInner`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub(super) unsafe fn drop_elements<T>(&mut self) {
        // Check that `self.items != 0`. Protects against the possibility
        // of creating an iterator on an table with uninitialized control bytes.
        if T::NEEDS_DROP && self.items != 0 {
            // SAFETY: We know for sure that RawTableInner will outlive the
            // returned `RawIter` iterator, and the caller of this function
            // must uphold the safety contract for `drop_elements` method.
            unsafe {
                for item in self.iter::<T>() {
                    // SAFETY: The caller must uphold the safety contract for
                    // `drop_elements` method.
                    item.drop();
                }
            }
        }
    }

    /// Executes the destructors (if any) of the values stored in the table and than
    /// deallocates the table.
    ///
    /// # Note
    ///
    /// Calling this function automatically makes invalid (dangling) all instances of
    /// buckets ([`Bucket`]) and makes invalid (dangling) the `ctrl` field of the table.
    ///
    /// This function does not make any changes to the `bucket_mask`, `items` or `growth_left`
    /// fields of the table. If necessary, the caller of this function must manually set
    /// up these table fields.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is [`undefined behavior`]:
    ///
    /// * Calling this function more than once;
    ///
    /// * The type `T` must be the actual type of the elements stored in the table.
    ///
    /// * The `alloc` must be the same [`Allocator`] as the `Allocator` that was used
    ///   to allocate this table.
    ///
    /// * The `table_layout` must be the same [`TableLayout`] as the `TableLayout` that
    ///   was used to allocate this table.
    ///
    /// The caller of this function should pay attention to the possibility of the
    /// elements' drop function panicking, because this:
    ///
    ///    * May leave the table in an inconsistent state;
    ///
    ///    * Memory is never deallocated, so a memory leak may occur.
    ///
    /// Attempt to use the `ctrl` field of the table (dereference) after calling this
    /// function results in [`undefined behavior`].
    ///
    /// It is safe to call this function on a table that has not been allocated,
    /// on a table with uninitialized control bytes, and on a table with no actual
    /// data but with `Full` control bytes if `self.items == 0`.
    ///
    /// See also [`RawTableInner::drop_elements`] or [`RawTableInner::free_buckets`]
    /// for more  information.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub(super) unsafe fn drop_inner_table<T, A: Allocator>(
        &mut self,
        alloc: &A,
        table_layout: TableLayout,
    ) {
        if !self.is_empty_singleton() {
            // SAFETY: The caller must uphold the safety contract for `drop_inner_table` method.
            unsafe {
                self.drop_elements::<T>();
            }
            // SAFETY:
            // 1. We have checked that our table is allocated.
            // 2. The caller must uphold the safety contract for `drop_inner_table` method.
            unsafe {
                self.free_buckets(alloc, table_layout);
            }
        }
    }

    /// Returns a pointer to an element in the table (convenience for
    /// `Bucket::from_base_index(self.data_end::<T>(), index)`).
    ///
    /// The caller must ensure that the `RawTableInner` outlives the returned [`Bucket<T>`],
    /// otherwise using it may result in [`undefined behavior`].
    ///
    /// # Safety
    ///
    /// If `size_of::<T>() != 0`, then the safety rules are directly derived from the
    /// safety rules of the [`Bucket::from_base_index`] function. Therefore, when calling
    /// this function, the following safety rules must be observed:
    ///
    /// * The table must already be allocated;
    ///
    /// * The `index` must not be greater than the number returned by the [`RawTableInner::num_buckets`]
    ///   function, i.e. `(index + 1) <= self.num_buckets()`.
    ///
    /// * The type `T` must be the actual type of the elements stored in the table, otherwise
    ///   using the returned [`Bucket`] may result in [`undefined behavior`].
    ///
    /// It is safe to call this function with index of zero (`index == 0`) on a table that has
    /// not been allocated, but using the returned [`Bucket`] results in [`undefined behavior`].
    ///
    /// If `size_of::<T>() == 0`, then the only requirement is that the `index` must
    /// not be greater than the number returned by the [`RawTable::num_buckets`] function, i.e.
    /// `(index + 1) <= self.num_buckets()`.
    ///
    /// ```none
    /// If size_of::<T>() != 0 then return a pointer to the `element` in the `data part` of the table
    /// (we start counting from "0", so that in the expression T[n], the "n" index actually one less than
    /// the "buckets" number of our `RawTableInner`, i.e. "n = RawTableInner::num_buckets() - 1"):
    ///
    ///           `table.bucket(3).as_ptr()` returns a pointer that points here in the `data`
    ///           part of the `RawTableInner`, i.e. to the start of T3 (see [`Bucket::as_ptr`])
    ///                  |
    ///                  |               `base = table.data_end::<T>()` points here
    ///                  |               (to the start of CT0 or to the end of T0)
    ///                  v                 v
    /// [Pad], T_n, ..., |T3|, T2, T1, T0, |CT0, CT1, CT2, CT3, ..., CT_n, CTa_0, CTa_1, ..., CTa_m
    ///                     ^                                              \__________  __________/
    ///        `table.bucket(3)` returns a pointer that points                        \/
    ///         here in the `data` part of the `RawTableInner`             additional control bytes
    ///         (to the end of T3)                                          `m = Group::WIDTH - 1`
    ///
    /// where: T0...T_n  - our stored data;
    ///        CT0...CT_n - control bytes or metadata for `data`;
    ///        CTa_0...CTa_m - additional control bytes (so that the search with loading `Group` bytes from
    ///                        the heap works properly, even if the result of `h1(hash) & self.bucket_mask`
    ///                        is equal to `self.bucket_mask`). See also `RawTableInner::set_ctrl` function.
    ///
    /// P.S. `h1(hash) & self.bucket_mask` is the same as `hash as usize % self.num_buckets()` because the number
    /// of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
    /// ```
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn bucket<T>(&self, index: usize) -> Bucket<T> {
        debug_assert_ne!(self.bucket_mask, 0);
        debug_assert!(index < self.num_buckets());
        unsafe { Bucket::from_base_index(self.data_end(), index) }
    }

    /// Returns a raw `*mut u8` pointer to the start of the `data` element in the table
    /// (convenience for `self.data_end::<u8>().as_ptr().sub((index + 1) * size_of)`).
    ///
    /// The caller must ensure that the `RawTableInner` outlives the returned `*mut u8`,
    /// otherwise using it may result in [`undefined behavior`].
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is [`undefined behavior`]:
    ///
    /// * The table must already be allocated;
    ///
    /// * The `index` must not be greater than the number returned by the [`RawTableInner::num_buckets`]
    ///   function, i.e. `(index + 1) <= self.num_buckets()`;
    ///
    /// * The `size_of` must be equal to the size of the elements stored in the table;
    ///
    /// ```none
    /// If size_of::<T>() != 0 then return a pointer to the `element` in the `data part` of the table
    /// (we start counting from "0", so that in the expression T[n], the "n" index actually one less than
    /// the "buckets" number of our `RawTableInner`, i.e. "n = RawTableInner::num_buckets() - 1"):
    ///
    ///           `table.bucket_ptr(3, size_of::<T>())` returns a pointer that points here in the
    ///           `data` part of the `RawTableInner`, i.e. to the start of T3
    ///                  |
    ///                  |               `base = table.data_end::<u8>()` points here
    ///                  |               (to the start of CT0 or to the end of T0)
    ///                  v                 v
    /// [Pad], T_n, ..., |T3|, T2, T1, T0, |CT0, CT1, CT2, CT3, ..., CT_n, CTa_0, CTa_1, ..., CTa_m
    ///                                                                    \__________  __________/
    ///                                                                               \/
    ///                                                                    additional control bytes
    ///                                                                     `m = Group::WIDTH - 1`
    ///
    /// where: T0...T_n  - our stored data;
    ///        CT0...CT_n - control bytes or metadata for `data`;
    ///        CTa_0...CTa_m - additional control bytes (so that the search with loading `Group` bytes from
    ///                        the heap works properly, even if the result of `h1(hash) & self.bucket_mask`
    ///                        is equal to `self.bucket_mask`). See also `RawTableInner::set_ctrl` function.
    ///
    /// P.S. `h1(hash) & self.bucket_mask` is the same as `hash as usize % self.num_buckets()` because the number
    /// of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
    /// ```
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn bucket_ptr(&self, index: usize, size_of: usize) -> *mut u8 {
        debug_assert_ne!(self.bucket_mask, 0);
        debug_assert!(index < self.num_buckets());
        unsafe {
            let base: *mut u8 = self.data_end().as_ptr();
            base.sub((index + 1) * size_of)
        }
    }

    /// Returns pointer to one past last `data` element in the table as viewed from
    /// the start point of the allocation (convenience for `self.ctrl.cast()`).
    ///
    /// This function actually returns a pointer to the end of the `data element` at
    /// index "0" (zero).
    ///
    /// The caller must ensure that the `RawTableInner` outlives the returned [`NonNull<T>`],
    /// otherwise using it may result in [`undefined behavior`].
    ///
    /// # Note
    ///
    /// The type `T` must be the actual type of the elements stored in the table, otherwise
    /// using the returned [`NonNull<T>`] may result in [`undefined behavior`].
    ///
    /// ```none
    ///                        `table.data_end::<T>()` returns pointer that points here
    ///                        (to the end of `T0`)
    ///                          ∨
    /// [Pad], T_n, ..., T1, T0, |CT0, CT1, ..., CT_n|, CTa_0, CTa_1, ..., CTa_m
    ///                           \________  ________/
    ///                                    \/
    ///       `n = buckets - 1`, i.e. `RawTableInner::num_buckets() - 1`
    ///
    /// where: T0...T_n  - our stored data;
    ///        CT0...CT_n - control bytes or metadata for `data`.
    ///        CTa_0...CTa_m - additional control bytes, where `m = Group::WIDTH - 1` (so that the search
    ///                        with loading `Group` bytes from the heap works properly, even if the result
    ///                        of `h1(hash) & self.bucket_mask` is equal to `self.bucket_mask`). See also
    ///                        `RawTableInner::set_ctrl` function.
    ///
    /// P.S. `h1(hash) & self.bucket_mask` is the same as `hash as usize % self.num_buckets()` because the number
    /// of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
    /// ```
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) fn data_end<T>(&self) -> NonNull<T> {
        self.ctrl.cast()
    }

    #[inline]
    pub(super) unsafe fn record_item_insert_at(
        &mut self,
        index: usize,
        old_ctrl: Tag,
        new_ctrl: Tag,
    ) {
        self.growth_left -= usize::from(old_ctrl.special_is_empty());
        unsafe {
            self.set_ctrl(index, new_ctrl);
        }
        self.items += 1;
    }

    /// Sets a control byte to the hash, and possibly also the replicated control byte at
    /// the end of the array.
    ///
    /// This function does not make any changes to the `data` parts of the table,
    /// or any changes to the `items` or `growth_left` field of the table.
    ///
    /// # Safety
    ///
    /// The safety rules are directly derived from the safety rules for [`RawTableInner::set_ctrl`]
    /// method. Thus, in order to uphold the safety contracts for the method, you must observe the
    /// following rules when calling this function:
    ///
    /// * The [`RawTableInner`] has already been allocated;
    ///
    /// * The `index` must not be greater than the `RawTableInner.bucket_mask`, i.e.
    ///   `index <= RawTableInner.bucket_mask` or, in other words, `(index + 1)` must
    ///   be no greater than the number returned by the function [`RawTableInner::num_buckets`].
    ///
    /// Calling this function on a table that has not been allocated results in [`undefined behavior`].
    ///
    /// See also [`Bucket::as_ptr`] method, for more information about of properly removing
    /// or saving `data element` from / into the [`RawTable`] / [`RawTableInner`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn set_ctrl_hash(&mut self, index: usize, hash: u64) {
        unsafe {
            // SAFETY: The caller must uphold the safety rules for the [`RawTableInner::set_ctrl_hash`]
            self.set_ctrl(index, Tag::full(hash));
        }
    }

    /// Replaces the hash in the control byte at the given index with the provided one,
    /// and possibly also replicates the new control byte at the end of the array of control
    /// bytes, returning the old control byte.
    ///
    /// This function does not make any changes to the `data` parts of the table,
    /// or any changes to the `items` or `growth_left` field of the table.
    ///
    /// # Safety
    ///
    /// The safety rules are directly derived from the safety rules for [`RawTableInner::set_ctrl_hash`]
    /// and [`RawTableInner::ctrl`] methods. Thus, in order to uphold the safety contracts for both
    /// methods, you must observe the following rules when calling this function:
    ///
    /// * The [`RawTableInner`] has already been allocated;
    ///
    /// * The `index` must not be greater than the `RawTableInner.bucket_mask`, i.e.
    ///   `index <= RawTableInner.bucket_mask` or, in other words, `(index + 1)` must
    ///   be no greater than the number returned by the function [`RawTableInner::num_buckets`].
    ///
    /// Calling this function on a table that has not been allocated results in [`undefined behavior`].
    ///
    /// See also [`Bucket::as_ptr`] method, for more information about of properly removing
    /// or saving `data element` from / into the [`RawTable`] / [`RawTableInner`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn replace_ctrl_hash(&mut self, index: usize, hash: u64) -> Tag {
        unsafe {
            // SAFETY: The caller must uphold the safety rules for the [`RawTableInner::replace_ctrl_hash`]
            let prev_ctrl = *self.ctrl(index);
            self.set_ctrl_hash(index, hash);
            prev_ctrl
        }
    }

    /// Sets a control byte, and possibly also the replicated control byte at
    /// the end of the array.
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
    /// * The `index` must not be greater than the `RawTableInner.bucket_mask`, i.e.
    ///   `index <= RawTableInner.bucket_mask` or, in other words, `(index + 1)` must
    ///   be no greater than the number returned by the function [`RawTableInner::num_buckets`].
    ///
    /// Calling this function on a table that has not been allocated results in [`undefined behavior`].
    ///
    /// See also [`Bucket::as_ptr`] method, for more information about of properly removing
    /// or saving `data element` from / into the [`RawTable`] / [`RawTableInner`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn set_ctrl(&mut self, index: usize, ctrl: Tag) {
        // Replicate the first Group::WIDTH control bytes at the end of
        // the array without using a branch. If the tables smaller than
        // the group width (self.num_buckets() < Group::WIDTH),
        // `index2 = Group::WIDTH + index`, otherwise `index2` is:
        //
        // - If index >= Group::WIDTH then index == index2.
        // - Otherwise index2 == self.bucket_mask + 1 + index.
        //
        // The very last replicated control byte is never actually read because
        // we mask the initial index for unaligned loads, but we write it
        // anyways because it makes the set_ctrl implementation simpler.
        //
        // If there are fewer buckets than Group::WIDTH then this code will
        // replicate the buckets at the end of the trailing group. For example
        // with 2 buckets and a group size of 4, the control bytes will look
        // like this:
        //
        //     Real    |             Replicated
        // ---------------------------------------------
        // | [A] | [B] | [Tag::EMPTY] | [EMPTY] | [A] | [B] |
        // ---------------------------------------------

        // This is the same as `(index.wrapping_sub(Group::WIDTH)) % self.num_buckets() + Group::WIDTH`
        // because the number of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
        let index2 = ((index.wrapping_sub(Group::WIDTH)) & self.bucket_mask) + Group::WIDTH;

        // SAFETY: The caller must uphold the safety rules for the [`RawTableInner::set_ctrl`]
        unsafe {
            *self.ctrl(index) = ctrl;
            *self.ctrl(index2) = ctrl;
        }
    }

    /// Returns a pointer to a control byte.
    ///
    /// # Safety
    ///
    /// For the allocated [`RawTableInner`], the result is [`Undefined Behavior`],
    /// if the `index` is greater than the `self.bucket_mask + 1 + Group::WIDTH`.
    /// In that case, calling this function with `index == self.bucket_mask + 1 + Group::WIDTH`
    /// will return a pointer to the end of the allocated table and it is useless on its own.
    ///
    /// Calling this function with `index >= self.bucket_mask + 1 + Group::WIDTH` on a
    /// table that has not been allocated results in [`Undefined Behavior`].
    ///
    /// So to satisfy both requirements you should always follow the rule that
    /// `index < self.bucket_mask + 1 + Group::WIDTH`
    ///
    /// Calling this function on [`RawTableInner`] that are not already allocated is safe
    /// for read-only purpose.
    ///
    /// See also [`Bucket::as_ptr()`] method, for more information about of properly removing
    /// or saving `data element` from / into the [`RawTable`] / [`RawTableInner`].
    ///
    /// [`Undefined Behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn ctrl(&self, index: usize) -> *mut Tag {
        debug_assert!(index < self.num_ctrl_bytes());
        // SAFETY: The caller must uphold the safety rules for the [`RawTableInner::ctrl`]
        unsafe { self.ctrl.as_ptr().add(index).cast() }
    }

    /// Gets the slice of all control bytes, as possibily uninitialized tags.
    pub(super) fn ctrl_slice(&mut self) -> &mut [mem::MaybeUninit<Tag>] {
        // SAFETY: We have the correct number of control bytes.
        unsafe { slice::from_raw_parts_mut(self.ctrl.as_ptr().cast(), self.num_ctrl_bytes()) }
    }

    #[inline]
    pub(super) fn num_buckets(&self) -> usize {
        self.bucket_mask + 1
    }

    /// Checks whether the bucket at `index` is full.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index` is less than the number of buckets.
    #[inline]
    pub(super) unsafe fn is_bucket_full(&self, index: usize) -> bool {
        debug_assert!(index < self.num_buckets());
        unsafe { (*self.ctrl(index)).is_full() }
    }

    #[inline]
    pub(super) fn num_ctrl_bytes(&self) -> usize {
        self.bucket_mask + 1 + Group::WIDTH
    }

    #[inline]
    pub(super) fn is_empty_singleton(&self) -> bool {
        self.bucket_mask == 0
    }

    /// Returns an iterator over full buckets indices in the table.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// * The caller has to ensure that the `RawTableInner` outlives the
    ///   `FullBucketsIndices`. Because we cannot make the `next` method
    ///   unsafe on the `FullBucketsIndices` struct, we have to make the
    ///   `full_buckets_indices` method unsafe.
    ///
    /// * The [`RawTableInner`] must have properly initialized control bytes.
    #[inline(always)]
    pub(super) unsafe fn full_buckets_indices(&self) -> FullBucketsIndices {
        // SAFETY:
        // 1. Since the caller of this function ensures that the control bytes
        //    are properly initialized and `self.ctrl(0)` points to the start
        //    of the array of control bytes, therefore: `ctrl` is valid for reads,
        //    properly aligned to `Group::WIDTH` and points to the properly initialized
        //    control bytes.
        // 2. The value of `items` is equal to the amount of data (values) added
        //    to the table.
        //
        //                         `ctrl` points here (to the start
        //                         of the first control byte `CT0`)
        //                          ∨
        // [Pad], T_n, ..., T1, T0, |CT0, CT1, ..., CT_n|, Group::WIDTH
        //                           \________  ________/
        //                                    \/
        //       `n = buckets - 1`, i.e. `RawTableInner::num_buckets() - 1`
        //
        // where: T0...T_n  - our stored data;
        //        CT0...CT_n - control bytes or metadata for `data`.
        unsafe {
            let ctrl = NonNull::new_unchecked(self.ctrl(0).cast::<u8>());

            FullBucketsIndices {
                // Load the first group
                // SAFETY: See explanation above.
                current_group: Group::load_aligned(ctrl.as_ptr().cast())
                    .match_full()
                    .into_iter(),
                group_first_index: 0,
                ctrl,
                items: self.items,
            }
        }
    }

    /// Deallocates the table without dropping any entries.
    ///
    /// # Note
    ///
    /// This function must be called only after [`drop_elements`](RawTableInner::drop_elements),
    /// else it can lead to leaking of memory. Also calling this function automatically
    /// makes invalid (dangling) all instances of buckets ([`Bucket`]) and makes invalid
    /// (dangling) the `ctrl` field of the table.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is [`Undefined Behavior`]:
    ///
    /// * The [`RawTableInner`] has already been allocated;
    ///
    /// * The `alloc` must be the same [`Allocator`] as the `Allocator` that was used
    ///   to allocate this table.
    ///
    /// * The `table_layout` must be the same [`TableLayout`] as the `TableLayout` that was used
    ///   to allocate this table.
    ///
    /// See also [`GlobalAlloc::dealloc`] or [`Allocator::deallocate`] for more  information.
    ///
    /// [`Undefined Behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`GlobalAlloc::dealloc`]: stdalloc::alloc::GlobalAlloc::dealloc
    /// [`Allocator::deallocate`]: stdalloc::alloc::Allocator::deallocate
    #[inline]
    pub(super) unsafe fn free_buckets<A>(&mut self, alloc: &A, table_layout: TableLayout)
    where
        A: Allocator,
    {
        unsafe {
            // SAFETY: The caller must uphold the safety contract for `free_buckets`
            // method.
            let (ptr, layout) = self.allocation_info(table_layout);
            alloc.deallocate(ptr, layout);
        }
    }

    /// Returns a pointer to the allocated memory and the layout that was used to
    /// allocate the table.
    ///
    /// # Safety
    ///
    /// Caller of this function must observe the following safety rules:
    ///
    /// * The [`RawTableInner`] has already been allocated, otherwise
    ///   calling this function results in [`undefined behavior`]
    ///
    /// * The `table_layout` must be the same [`TableLayout`] as the `TableLayout`
    ///   that was used to allocate this table. Failure to comply with this condition
    ///   may result in [`undefined behavior`].
    ///
    /// See also [`GlobalAlloc::dealloc`] or [`Allocator::deallocate`] for more  information.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    /// [`GlobalAlloc::dealloc`]: stdalloc::GlobalAlloc::dealloc
    /// [`Allocator::deallocate`]: stdalloc::Allocator::deallocate
    #[inline]
    pub(super) unsafe fn allocation_info(
        &self,
        table_layout: TableLayout,
    ) -> (NonNull<u8>, Layout) {
        debug_assert!(
            !self.is_empty_singleton(),
            "this function can only be called on non-empty tables"
        );

        let (layout, ctrl_offset) = {
            let option = table_layout.calculate_layout_for(self.num_buckets());
            unsafe { option.unwrap_unchecked() }
        };
        (
            // SAFETY: The caller must uphold the safety contract for `allocation_info` method.
            unsafe { NonNull::new_unchecked(self.ctrl.as_ptr().sub(ctrl_offset)) },
            layout,
        )
    }

    /// Returns the total amount of memory allocated internally by the hash
    /// table, in bytes.
    ///
    /// The returned number is informational only. It is intended to be
    /// primarily used for memory profiling.
    ///
    /// # Safety
    ///
    /// The `table_layout` must be the same [`TableLayout`] as the `TableLayout`
    /// that was used to allocate this table. Failure to comply with this condition
    /// may result in [`undefined behavior`].
    ///
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn allocation_size_or_zero(&self, table_layout: TableLayout) -> usize {
        if self.is_empty_singleton() {
            0
        } else {
            // SAFETY:
            // 1. We have checked that our table is allocated.
            // 2. The caller ensures that `table_layout` matches the [`TableLayout`]
            // that was used to allocate this table.
            unsafe { self.allocation_info(table_layout).1.size() }
        }
    }

    /// Marks all table buckets as empty without dropping their contents.
    #[inline]
    pub(super) fn clear_no_drop(&mut self) {
        if !self.is_empty_singleton() {
            self.ctrl_slice().fill_empty();
        }
        self.items = 0;
        self.growth_left = bucket_mask_to_capacity(self.bucket_mask);
    }

    /// Erases the [`Bucket`]'s control byte at the given index so that it does not
    /// triggered as full, decreases the `items` of the table and, if it can be done,
    /// increases `self.growth_left`.
    ///
    /// This function does not actually erase / drop the [`Bucket`] itself, i.e. it
    /// does not make any changes to the `data` parts of the table. The caller of this
    /// function must take care to properly drop the `data`, otherwise calling this
    /// function may result in a memory leak.
    ///
    /// # Safety
    ///
    /// You must observe the following safety rules when calling this function:
    ///
    /// * The [`RawTableInner`] has already been allocated;
    ///
    /// * It must be the full control byte at the given position;
    ///
    /// * The `index` must not be greater than the `RawTableInner.bucket_mask`, i.e.
    ///   `index <= RawTableInner.bucket_mask` or, in other words, `(index + 1)` must
    ///   be no greater than the number returned by the function [`RawTableInner::num_buckets`].
    ///
    /// Calling this function on a table that has not been allocated results in [`undefined behavior`].
    ///
    /// Calling this function on a table with no elements is unspecified, but calling subsequent
    /// functions is likely to result in [`undefined behavior`] due to overflow subtraction
    /// (`self.items -= 1 cause overflow when self.items == 0`).
    ///
    /// See also [`Bucket::as_ptr`] method, for more information about of properly removing
    /// or saving `data element` from / into the [`RawTable`] / [`RawTableInner`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn erase(&mut self, index: usize) {
        unsafe {
            debug_assert!(self.is_bucket_full(index));
        }

        // This is the same as `index.wrapping_sub(Group::WIDTH) % self.num_buckets()` because
        // the number of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
        let index_before = index.wrapping_sub(Group::WIDTH) & self.bucket_mask;
        // SAFETY:
        // - The caller must uphold the safety contract for `erase` method;
        // - `index_before` is guaranteed to be in range due to masking with `self.bucket_mask`
        let (empty_before, empty_after) = unsafe {
            (
                Group::load(self.ctrl(index_before)).match_empty(),
                Group::load(self.ctrl(index)).match_empty(),
            )
        };

        // Inserting and searching in the map is performed by two key functions:
        //
        // - The `find_insert_index` function that looks up the index of any `Tag::EMPTY` or `Tag::DELETED`
        //   slot in a group to be able to insert. If it doesn't find an `Tag::EMPTY` or `Tag::DELETED`
        //   slot immediately in the first group, it jumps to the next `Group` looking for it,
        //   and so on until it has gone through all the groups in the control bytes.
        //
        // - The `find_inner` function that looks for the index of the desired element by looking
        //   at all the `FULL` bytes in the group. If it did not find the element right away, and
        //   there is no `Tag::EMPTY` byte in the group, then this means that the `find_insert_index`
        //   function may have found a suitable slot in the next group. Therefore, `find_inner`
        //   jumps further, and if it does not find the desired element and again there is no `Tag::EMPTY`
        //   byte, then it jumps further, and so on. The search stops only if `find_inner` function
        //   finds the desired element or hits an `Tag::EMPTY` slot/byte.
        //
        // Accordingly, this leads to two consequences:
        //
        // - The map must have `Tag::EMPTY` slots (bytes);
        //
        // - You can't just mark the byte to be erased as `Tag::EMPTY`, because otherwise the `find_inner`
        //   function may stumble upon an `Tag::EMPTY` byte before finding the desired element and stop
        //   searching.
        //
        // Thus it is necessary to check all bytes after and before the erased element. If we are in
        // a contiguous `Group` of `FULL` or `Tag::DELETED` bytes (the number of `FULL` or `Tag::DELETED` bytes
        // before and after is greater than or equal to `Group::WIDTH`), then we must mark our byte as
        // `Tag::DELETED` in order for the `find_inner` function to go further. On the other hand, if there
        // is at least one `Tag::EMPTY` slot in the `Group`, then the `find_inner` function will still stumble
        // upon an `Tag::EMPTY` byte, so we can safely mark our erased byte as `Tag::EMPTY` as well.
        //
        // Finally, since `index_before == (index.wrapping_sub(Group::WIDTH) & self.bucket_mask) == index`
        // and given all of the above, tables smaller than the group width (self.num_buckets() < Group::WIDTH)
        // cannot have `Tag::DELETED` bytes.
        //
        // Note that in this context `leading_zeros` refers to the bytes at the end of a group, while
        // `trailing_zeros` refers to the bytes at the beginning of a group.
        let ctrl = if empty_before.leading_zeros() + empty_after.trailing_zeros() >= Group::WIDTH {
            Tag::DELETED
        } else {
            self.growth_left += 1;
            Tag::EMPTY
        };
        // SAFETY: the caller must uphold the safety contract for `erase` method.
        unsafe {
            self.set_ctrl(index, ctrl);
        }
        self.items -= 1;
    }
}
