//! Hash probing: the `ProbeCursor` and the group-scanning loops of the table.
//!
//! A [`ProbeCursor`] only produces group positions and the probe termination
//! condition; it never reads control bytes or touches buckets itself. The
//! termination condition is part of the shared contract of all control group
//! backends (SSE2, NEON, LSX and the generic fallback): a probe must stop at
//! the first group that contains an `Tag::EMPTY` control byte, which every
//! backend reports via [`Group::match_empty`].

use crate::control::Group;
use crate::control::Tag;
use crate::util::{likely, unlikely};

use super::storage::RawTableInner;

/// Primary hash function, used to select the initial bucket to probe from.
#[inline]
#[expect(clippy::cast_possible_truncation)]
fn h1(hash: u64) -> usize {
    // On 32-bit platforms we simply ignore the higher hash bits.
    hash as usize
}

/// Probe sequence based on triangular numbers, which is guaranteed (since our
/// table size is a power of two) to visit every group of elements exactly once.
///
/// A triangular probe has us jump by 1 more group every time. So first we
/// jump by 1 group (meaning we just continue our linear scan), then 2 groups
/// (skipping over 1 group), then 3 groups (skipping over 2 groups), and so on.
///
/// Proof that the probe will visit every group in the table:
/// <https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/>
#[derive(Clone)]
struct ProbeSeq {
    pos: usize,
    stride: usize,
}

impl ProbeSeq {
    #[inline]
    fn move_next(&mut self, bucket_mask: usize) {
        // We should have found an empty bucket by now and ended the probe.
        debug_assert!(
            self.stride <= bucket_mask,
            "Went past end of probe sequence"
        );

        self.stride = self.stride.wrapping_add(Group::WIDTH);
        self.pos = self.pos.wrapping_add(self.stride) & bucket_mask;
    }
}

/// A cursor over the probe sequence of a table.
///
/// The cursor only yields group positions and the probe termination
/// condition; it never reads control bytes or touches buckets itself.
///
/// The cursor never terminates on its own, but is guaranteed to visit each
/// bucket group exactly once. The loop using the cursor must terminate upon
/// reaching a group for which [`ProbeCursor::is_terminal`] returns `true`.
#[derive(Clone)]
pub(super) struct ProbeCursor {
    // The underlying triangular probe sequence.
    seq: ProbeSeq,

    // Mask to get an index from a hash value. The value is one less than the
    // number of buckets in the table.
    bucket_mask: usize,
}

impl ProbeCursor {
    /// Creates a cursor starting at the group selected by `h1(hash)`.
    ///
    /// This is the same as `hash as usize % num_buckets` because the number
    /// of buckets is a power of two, and `bucket_mask = num_buckets - 1`.
    #[inline]
    fn new(hash: u64, bucket_mask: usize) -> Self {
        Self {
            seq: ProbeSeq {
                pos: h1(hash) & bucket_mask,
                stride: 0,
            },
            bucket_mask,
        }
    }

    /// Returns the position of the first control byte of the current group.
    ///
    /// The returned position is always in the range `0..=bucket_mask`.
    #[inline]
    pub(super) fn group_pos(&self) -> usize {
        self.seq.pos
    }

    /// Advances the cursor to the next group in the probe sequence.
    #[inline]
    pub(super) fn move_next(&mut self) {
        self.seq.move_next(self.bucket_mask);
    }

    /// Maps a bit index within the current group to a bucket index in the
    /// table.
    ///
    /// This is the same as `(group_pos + bit) % num_buckets` because the
    /// number of buckets is a power of two, and `bucket_mask = num_buckets - 1`.
    #[inline]
    pub(super) fn bucket_index(&self, bit: usize) -> usize {
        (self.seq.pos + bit) & self.bucket_mask
    }

    /// The termination condition of a probe: probing must stop at the first
    /// group that contains an `Tag::EMPTY` control byte.
    ///
    /// This is the shared contract of all control group backends (SSE2,
    /// NEON, LSX and the generic fallback).
    #[inline]
    pub(super) fn is_terminal(group: &Group) -> bool {
        group.match_empty().any_bit_set()
    }
}

impl RawTableInner {
    /// Fixes up an insertion index returned by the [`RawTableInner::find_insert_index_in_group`] method.
    ///
    /// In tables smaller than the group width (`self.num_buckets() < Group::WIDTH`), trailing control
    /// bytes outside the range of the table are filled with [`Tag::EMPTY`] entries. These will unfortunately
    /// trigger a match of [`RawTableInner::find_insert_index_in_group`] function. This is because
    /// the `Some(bit)` returned by `group.match_empty_or_deleted().lowest_set_bit()` after masking
    /// (`(probe_seq.pos + bit) & self.bucket_mask`) may point to a full bucket that is already occupied.
    /// We detect this situation here and perform a second scan starting at the beginning of the table.
    /// This second scan is guaranteed to find an empty slot (due to the load factor) before hitting the
    /// trailing control bytes (containing [`Tag::EMPTY`] bytes).
    ///
    /// If this function is called correctly, it is guaranteed to return an index of an empty or
    /// deleted bucket in the range `0..self.num_buckets()` (see `Warning` and `Safety`).
    ///
    /// # Warning
    ///
    /// The table must have at least 1 empty or deleted `bucket`, otherwise if the table is less than
    /// the group width (`self.num_buckets() < Group::WIDTH`) this function returns an index outside of the
    /// table indices range `0..self.num_buckets()` (`0..=self.bucket_mask`). Attempt to write data at that
    /// index will cause immediate [`undefined behavior`].
    ///
    /// # Safety
    ///
    /// The safety rules are directly derived from the safety rules for [`RawTableInner::ctrl`] method.
    /// Thus, in order to uphold those safety contracts, as well as for the correct logic of the work
    /// of this crate, the following rules are necessary and sufficient:
    ///
    /// * The [`RawTableInner`] must have properly initialized control bytes otherwise calling this
    ///   function results in [`undefined behavior`].
    ///
    /// * This function must only be used on insertion indices found by [`RawTableInner::find_insert_index_in_group`]
    ///   (after the `find_insert_index_in_group` function, but before insertion into the table).
    ///
    /// * The `index` must not be greater than the `self.bucket_mask`, i.e. `(index + 1) <= self.num_buckets()`
    ///   (this one is provided by the [`RawTableInner::find_insert_index_in_group`] function).
    ///
    /// Calling this function with an index not provided by [`RawTableInner::find_insert_index_in_group`]
    /// may result in [`undefined behavior`] even if the index satisfies the safety rules of the
    /// [`RawTableInner::ctrl`] function (`index < self.bucket_mask + 1 + Group::WIDTH`).
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    unsafe fn fix_insert_index(&self, mut index: usize) -> usize {
        // SAFETY: The caller of this function ensures that `index` is in the range `0..=self.bucket_mask`.
        if unlikely(unsafe { self.is_bucket_full(index) }) {
            debug_assert!(self.bucket_mask < Group::WIDTH);
            // SAFETY:
            //
            // * Since the caller of this function ensures that the control bytes are properly
            //   initialized and `ptr = self.ctrl(0)` points to the start of the array of control
            //   bytes, therefore: `ctrl` is valid for reads, properly aligned to `Group::WIDTH`
            //   and points to the properly initialized control bytes (see also
            //   `TableLayout::calculate_layout_for` and `ptr::read`);
            //
            // * Because the caller of this function ensures that the index was provided by the
            //   `self.find_insert_index_in_group()` function, so for for tables larger than the
            //   group width (self.num_buckets() >= Group::WIDTH), we will never end up in the given
            //   branch, since `(probe_seq.pos + bit) & self.bucket_mask` in `find_insert_index_in_group`
            //   cannot return a full bucket index. For tables smaller than the group width, calling
            //   the `unwrap_unchecked` function is also safe, as the trailing control bytes outside
            //   the range of the table are filled with EMPTY bytes (and we know for sure that there
            //   is at least one FULL bucket), so this second scan either finds an empty slot (due to
            //   the load factor) or hits the trailing control bytes (containing EMPTY).
            index = unsafe {
                Group::load_aligned(self.ctrl(0))
                    .match_empty_or_deleted()
                    .lowest_set_bit()
                    .unwrap_unchecked()
            };
        }
        index
    }

    /// Finds the position to insert something in a group.
    ///
    /// **This may have false positives and must be fixed up with `fix_insert_index`
    /// before it's used.**
    ///
    /// The function is guaranteed to return the index of an empty or deleted [`Bucket`]
    /// in the range `0..self.num_buckets()` (`0..=self.bucket_mask`).
    #[inline]
    fn find_insert_index_in_group(&self, group: &Group, cursor: &ProbeCursor) -> Option<usize> {
        let bit = group.match_empty_or_deleted().lowest_set_bit();

        if likely(bit.is_some()) {
            // This is the same as `(group_pos + bit) % self.num_buckets()` because the number
            // of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
            Some(cursor.bucket_index(bit.unwrap()))
        } else {
            None
        }
    }

    /// Searches for an element in the table, or a potential slot where that element could
    /// be inserted (an empty or deleted [`Bucket`] index).
    ///
    /// This uses dynamic dispatch to reduce the amount of code generated, but that is
    /// eliminated by LLVM optimizations.
    ///
    /// This function does not make any changes to the `data` part of the table, or any
    /// changes to the `items` or `growth_left` field of the table.
    ///
    /// The table must have at least 1 empty or deleted `bucket`, otherwise, if the
    /// `eq: &mut dyn FnMut(usize) -> bool` function does not return `true`, this function
    /// will never return (will go into an infinite loop) for tables larger than the group
    /// width, or return an index outside of the table indices range if the table is less
    /// than the group width.
    ///
    /// This function is guaranteed to provide the `eq: &mut dyn FnMut(usize) -> bool`
    /// function with only `FULL` buckets' indices and return the `index` of the found
    /// element (as `Ok(index)`). If the element is not found and there is at least 1
    /// empty or deleted [`Bucket`] in the table, the function is guaranteed to return
    /// an index in the range `0..self.num_buckets()`, but in any case, if this function
    /// returns `Err`, it will contain an index in the range `0..=self.num_buckets()`.
    ///
    /// # Safety
    ///
    /// The [`RawTableInner`] must have properly initialized control bytes otherwise calling
    /// this function results in [`undefined behavior`].
    ///
    /// Attempt to write data at the index returned by this function when the table is less than
    /// the group width and if there was not at least one empty or deleted bucket in the table
    /// will cause immediate [`undefined behavior`]. This is because in this case the function
    /// will return `self.bucket_mask + 1` as an index due to the trailing [`Tag::EMPTY`] control
    /// bytes outside the table range.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn find_or_find_insert_index_inner(
        &self,
        hash: u64,
        eq: &mut dyn FnMut(usize) -> bool,
    ) -> Result<usize, usize> {
        let mut insert_index = None;

        let tag_hash = Tag::full(hash);
        let mut cursor = self.probe_seq(hash);

        loop {
            // SAFETY:
            // * Caller of this function ensures that the control bytes are properly initialized.
            //
            // * `ProbeSeq.pos` cannot be greater than `self.bucket_mask = self.num_buckets() - 1`
            //   of the table due to masking with `self.bucket_mask` and also because the number
            //   of buckets is a power of two (see `self.probe_seq` function).
            //
            // * Even if `ProbeSeq.pos` returns `position == self.bucket_mask`, it is safe to
            //   call `Group::load` due to the extended control bytes range, which is
            //  `self.bucket_mask + 1 + Group::WIDTH` (in fact, this means that the last control
            //   byte will never be read for the allocated table);
            //
            // * Also, even if `RawTableInner` is not already allocated, `ProbeSeq.pos` will
            //   always return "0" (zero), so Group::load will read unaligned `Group::static_empty()`
            //   bytes, which is safe (see RawTableInner::new).
            let group = unsafe { Group::load(self.ctrl(cursor.group_pos())) };

            for bit in group.match_tag(tag_hash) {
                let index = cursor.bucket_index(bit);

                if likely(eq(index)) {
                    return Ok(index);
                }
            }

            // We didn't find the element we were looking for in the group, try to get an
            // insertion slot from the group if we don't have one yet.
            if likely(insert_index.is_none()) {
                insert_index = self.find_insert_index_in_group(&group, &cursor);
            }

            if let Some(insert_index) = insert_index {
                // Only stop the search if the group contains at least one empty element.
                // Otherwise, the element that we are looking for might be in a following group.
                if likely(ProbeCursor::is_terminal(&group)) {
                    // We must have found a insert slot by now, since the current group contains at
                    // least one. For tables smaller than the group width, there will still be an
                    // empty element in the current (and only) group due to the load factor.
                    unsafe {
                        // SAFETY:
                        // * Caller of this function ensures that the control bytes are properly initialized.
                        //
                        // * We use this function with the index found by `self.find_insert_index_in_group`
                        return Err(self.fix_insert_index(insert_index));
                    }
                }
            }

            cursor.move_next();
        }
    }

    /// Searches for an empty or deleted bucket which is suitable for inserting a new
    /// element and sets the hash for that slot. Returns an index of that slot and the
    /// old control byte stored in the found index.
    ///
    /// This function does not check if the given element exists in the table. Also,
    /// this function does not check if there is enough space in the table to insert
    /// a new element. The caller of the function must make sure that the table has at
    /// least 1 empty or deleted `bucket`, otherwise this function will never return
    /// (will go into an infinite loop) for tables larger than the group width, or
    /// return an index outside of the table indices range if the table is less than
    /// the group width.
    ///
    /// If there is at least 1 empty or deleted `bucket` in the table, the function is
    /// guaranteed to return an `index` in the range `0..self.num_buckets()`, but in any case,
    /// if this function returns an `index` it will be in the range `0..=self.num_buckets()`.
    ///
    /// This function does not make any changes to the `data` parts of the table,
    /// or any changes to the `items` or `growth_left` field of the table.
    ///
    /// # Safety
    ///
    /// The safety rules are directly derived from the safety rules for the
    /// [`RawTableInner::set_ctrl_hash`] and [`RawTableInner::find_insert_index`] methods.
    /// Thus, in order to uphold the safety contracts for that methods, as well as for
    /// the correct logic of the work of this crate, you must observe the following rules
    /// when calling this function:
    ///
    /// * The [`RawTableInner`] has already been allocated and has properly initialized
    ///   control bytes otherwise calling this function results in [`undefined behavior`].
    ///
    /// * The caller of this function must ensure that the "data" parts of the table
    ///   will have an entry in the returned index (matching the given hash) right
    ///   after calling this function.
    ///
    /// Attempt to write data at the `index` returned by this function when the table is
    /// less than the group width and if there was not at least one empty or deleted bucket in
    /// the table will cause immediate [`undefined behavior`]. This is because in this case the
    /// function will return `self.bucket_mask + 1` as an index due to the trailing [`Tag::EMPTY`]
    /// control bytes outside the table range.
    ///
    /// The caller must independently increase the `items` field of the table, and also,
    /// if the old control byte was [`Tag::EMPTY`], then decrease the table's `growth_left`
    /// field, and do not change it if the old control byte was [`Tag::DELETED`].
    ///
    /// See also [`Bucket::as_ptr`] method, for more information about of properly removing
    /// or saving `element` from / into the [`RawTable`] / [`RawTableInner`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn prepare_insert_index(&mut self, hash: u64) -> (usize, Tag) {
        unsafe {
            // SAFETY: Caller of this function ensures that the control bytes are properly initialized.
            let index: usize = self.find_insert_index(hash);
            // SAFETY:
            // 1. The `find_insert_index` function either returns an `index` less than or
            //    equal to `self.num_buckets() = self.bucket_mask + 1` of the table, or never
            //    returns if it cannot find an empty or deleted slot.
            // 2. The caller of this function guarantees that the table has already been
            //    allocated
            let old_ctrl = *self.ctrl(index);
            self.set_ctrl_hash(index, hash);
            (index, old_ctrl)
        }
    }

    /// Searches for an empty or deleted bucket which is suitable for inserting
    /// a new element, returning the `index` for the new [`Bucket`].
    ///
    /// This function does not make any changes to the `data` part of the table, or any
    /// changes to the `items` or `growth_left` field of the table.
    ///
    /// The table must have at least 1 empty or deleted `bucket`, otherwise this function
    /// will never return (will go into an infinite loop) for tables larger than the group
    /// width, or return an index outside of the table indices range if the table is less
    /// than the group width.
    ///
    /// If there is at least 1 empty or deleted `bucket` in the table, the function is
    /// guaranteed to return an index in the range `0..self.num_buckets()`, but in any case,
    /// it will contain an index in the range `0..=self.num_buckets()`.
    ///
    /// # Safety
    ///
    /// The [`RawTableInner`] must have properly initialized control bytes otherwise calling
    /// this function results in [`undefined behavior`].
    ///
    /// Attempt to write data at the index returned by this function when the table is
    /// less than the group width and if there was not at least one empty or deleted bucket in
    /// the table will cause immediate [`undefined behavior`]. This is because in this case the
    /// function will return `self.bucket_mask + 1` as an index due to the trailing [`Tag::EMPTY`]
    /// control bytes outside the table range.
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline]
    pub(super) unsafe fn find_insert_index(&self, hash: u64) -> usize {
        let mut cursor = self.probe_seq(hash);
        loop {
            // SAFETY:
            // * Caller of this function ensures that the control bytes are properly initialized.
            //
            // * `ProbeSeq.pos` cannot be greater than `self.bucket_mask = self.num_buckets() - 1`
            //   of the table due to masking with `self.bucket_mask` and also because the number
            //   of buckets is a power of two (see `self.probe_seq` function).
            //
            // * Even if `ProbeSeq.pos` returns `position == self.bucket_mask`, it is safe to
            //   call `Group::load` due to the extended control bytes range, which is
            //  `self.bucket_mask + 1 + Group::WIDTH` (in fact, this means that the last control
            //   byte will never be read for the allocated table);
            //
            // * Also, even if `RawTableInner` is not already allocated, `ProbeSeq.pos` will
            //   always return "0" (zero), so Group::load will read unaligned `Group::static_empty()`
            //   bytes, which is safe (see RawTableInner::new).
            let group = unsafe { Group::load(self.ctrl(cursor.group_pos())) };

            let index = self.find_insert_index_in_group(&group, &cursor);
            if likely(index.is_some()) {
                // SAFETY:
                // * Caller of this function ensures that the control bytes are properly initialized.
                //
                // * We use this function with the slot / index found by `self.find_insert_index_in_group`
                unsafe {
                    return self.fix_insert_index(index.unwrap_unchecked());
                }
            }
            cursor.move_next();
        }
    }

    /// Searches for an element in a table, returning the `index` of the found element.
    /// This uses dynamic dispatch to reduce the amount of code generated, but it is
    /// eliminated by LLVM optimizations.
    ///
    /// This function does not make any changes to the `data` part of the table, or any
    /// changes to the `items` or `growth_left` field of the table.
    ///
    /// The table must have at least 1 empty `bucket`, otherwise, if the
    /// `eq: &mut dyn FnMut(usize) -> bool` function does not return `true`,
    /// this function will also never return (will go into an infinite loop).
    ///
    /// This function is guaranteed to provide the `eq: &mut dyn FnMut(usize) -> bool`
    /// function with only `FULL` buckets' indices and return the `index` of the found
    /// element as `Some(index)`, so the index will always be in the range
    /// `0..self.num_buckets()`.
    ///
    /// # Safety
    ///
    /// The [`RawTableInner`] must have properly initialized control bytes otherwise calling
    /// this function results in [`undefined behavior`].
    ///
    /// [`undefined behavior`]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[inline(always)]
    pub(super) unsafe fn find_inner(
        &self,
        hash: u64,
        eq: &mut dyn FnMut(usize) -> bool,
    ) -> Option<usize> {
        let tag_hash = Tag::full(hash);
        let mut cursor = self.probe_seq(hash);

        loop {
            // SAFETY:
            // * Caller of this function ensures that the control bytes are properly initialized.
            //
            // * `ProbeSeq.pos` cannot be greater than `self.bucket_mask = self.num_buckets() - 1`
            //   of the table due to masking with `self.bucket_mask`.
            //
            // * Even if `ProbeSeq.pos` returns `position == self.bucket_mask`, it is safe to
            //   call `Group::load` due to the extended control bytes range, which is
            //  `self.bucket_mask + 1 + Group::WIDTH` (in fact, this means that the last control
            //   byte will never be read for the allocated table);
            //
            // * Also, even if `RawTableInner` is not already allocated, `ProbeSeq.pos` will
            //   always return "0" (zero), so Group::load will read unaligned `Group::static_empty()`
            //   bytes, which is safe (see RawTableInner::new_in).
            let group = unsafe { Group::load(self.ctrl(cursor.group_pos())) };

            for bit in group.match_tag(tag_hash) {
                // This is the same as `cursor.bucket_index(bit) % self.num_buckets()` because the number
                // of buckets is a power of two, and `self.bucket_mask = self.num_buckets() - 1`.
                let index = cursor.bucket_index(bit);

                if likely(eq(index)) {
                    return Some(index);
                }
            }

            if likely(ProbeCursor::is_terminal(&group)) {
                return None;
            }

            cursor.move_next();
        }
    }

    /// Returns a cursor over the probe sequence on the table for the given
    /// hash.
    ///
    /// The cursor never terminates on its own, but is guaranteed to visit
    /// each bucket group exactly once. The loop using the cursor must
    /// terminate upon reaching a group containing an empty bucket.
    #[inline]
    pub(super) fn probe_seq(&self, hash: u64) -> ProbeCursor {
        ProbeCursor::new(hash, self.bucket_mask)
    }

    #[inline]
    pub(super) fn is_in_same_group(&self, i: usize, new_i: usize, hash: u64) -> bool {
        let probe_seq_pos = self.probe_seq(hash).group_pos();
        let probe_index =
            |pos: usize| (pos.wrapping_sub(probe_seq_pos) & self.bucket_mask) / Group::WIDTH;
        probe_index(i) == probe_index(new_i)
    }
}
