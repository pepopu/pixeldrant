//! Fixed-capacity top-k accumulator: a max-heap of the k smallest
//! (distance, id) pairs seen so far.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(PartialEq)]
struct Entry {
    dist: f32,
    id: u32,
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist).then_with(|| self.id.cmp(&other.id))
    }
}

pub struct TopK {
    k: usize,
    heap: BinaryHeap<Entry>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        assert!(k > 0);
        Self { k, heap: BinaryHeap::with_capacity(k + 1) }
    }

    /// Current admission threshold: once full, pushes with `dist >=` this
    /// are no-ops. Exposed so scans can prune early.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k { f32::INFINITY } else { self.heap.peek().unwrap().dist }
    }

    #[inline]
    pub fn push(&mut self, dist: f32, id: u32) {
        if self.heap.len() < self.k {
            self.heap.push(Entry { dist, id });
        } else if dist < self.heap.peek().unwrap().dist {
            // Replace the current worst in place; PeekMut sifts down on drop.
            *self.heap.peek_mut().unwrap() = Entry { dist, id };
        }
    }

    /// Results in ascending distance order.
    pub fn into_sorted(self) -> Vec<(f32, u32)> {
        self.heap.into_sorted_vec().into_iter().map(|e| (e.dist, e.id)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_k_smallest_in_order() {
        let mut topk = TopK::new(3);
        for (i, d) in [5.0, 1.0, 4.0, 2.0, 3.0, 0.5].into_iter().enumerate() {
            topk.push(d, i as u32);
        }
        let result = topk.into_sorted();
        assert_eq!(result, vec![(0.5, 5), (1.0, 1), (2.0, 3)]);
    }

    #[test]
    fn threshold_tracks_worst_kept() {
        let mut topk = TopK::new(2);
        assert_eq!(topk.threshold(), f32::INFINITY);
        topk.push(3.0, 0);
        topk.push(1.0, 1);
        assert_eq!(topk.threshold(), 3.0);
        topk.push(2.0, 2);
        assert_eq!(topk.threshold(), 2.0);
    }
}
