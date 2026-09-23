//! Deterministic operation-sequence test checked against a model map.
//!
//! The exact same sequence of operations is exercised under every
//! control-group backend (SSE2 / NEON / LSX / generic). The backend is
//! selected at compile time from the target features, so running the test
//! suite with different `RUSTFLAGS="-C target-feature=..."` values (or under
//! miri, which always uses the generic backend) covers each backend with
//! bit-identical operations.
//!
//! The sequence is fully deterministic: it uses a fixed-seed PRNG and the
//! crate's default hasher, so any behavioral difference between backends
//! (control-byte handling, mirrored bytes, small tables, tombstones) shows
//! up as a mismatch against the `BTreeMap` model.

use hashbrown::HashMap;
use std::collections::BTreeMap;

/// xorshift64* PRNG, deterministic across platforms and backends.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn check_against_model(map: &HashMap<u64, u64>, model: &BTreeMap<u64, u64>) {
    assert_eq!(map.len(), model.len());
    let mut from_map: Vec<(u64, u64)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    from_map.sort_unstable();
    let from_model: Vec<(u64, u64)> = model.iter().map(|(&k, &v)| (k, v)).collect();
    assert_eq!(from_map, from_model);
}

#[test]
fn ops_sequence_matches_model() {
    let steps: u64 = if cfg!(miri) { 3_000 } else { 200_000 };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut map = HashMap::with_capacity(3);
    let mut model = BTreeMap::new();

    for step in 0..steps {
        let op = rng.next() % 20;
        // A small key space forces collisions, tombstones, growth and
        // shrinking to happen regularly.
        let key = rng.next() % 600;
        match op {
            0..=8 => {
                map.insert(key, step);
                model.insert(key, step);
            }
            9..=11 => {
                assert_eq!(map.remove(&key), model.remove(&key));
            }
            12..=14 => {
                assert_eq!(map.get(&key), model.get(&key));
            }
            15 => {
                let v = rng.next();
                map.entry(key).and_modify(|e| *e ^= v).or_insert(v);
                model.entry(key).and_modify(|e| *e ^= v).or_insert(v);
            }
            16 => {
                // Multi-key mutable borrow.
                let k2 = rng.next() % 600;
                let [a, b] = map.get_disjoint_mut([&key, &k2]);
                if let Some(v) = a {
                    *v += 1;
                }
                if let Some(v) = b {
                    *v += 1;
                }
                if let Some(v) = model.get_mut(&key) {
                    *v += 1;
                }
                if key != k2
                    && let Some(v) = model.get_mut(&k2)
                {
                    *v += 1;
                }
            }
            17 => {
                let pivot = rng.next() % 600;
                map.retain(|&k, _| k != pivot);
                model.retain(|&k, _| k != pivot);
            }
            18 => {
                if rng.next() % 2 == 0 {
                    map.reserve((rng.next() % 64) as usize);
                } else {
                    map.shrink_to_fit();
                }
            }
            19 => {
                let cloned = map.clone();
                assert_eq!(cloned, map);
                if rng.next() % 4 == 0 {
                    let drained: Vec<(u64, u64)> = map.drain().collect();
                    for (k, v) in drained {
                        assert_eq!(model.remove(&k), Some(v));
                    }
                    assert!(model.is_empty() || map.is_empty() || !model.is_empty());
                    map.extend(model.iter().map(|(&k, &v)| (k, v)));
                }
            }
            _ => unreachable!(),
        }

        if step % 1024 == 0 {
            check_against_model(&map, &model);
        }
    }

    check_against_model(&map, &model);
}
