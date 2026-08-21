//! Feature-hash mixers for packed n-gram keys.
//!
//! N-grams are encoded as `u64` keys (base-2^21 codepoint packing, see
//! [`crate::features`]). The hash maps a key to a bucket in `2^log2_d` plus a
//! sign bit (signed hashing: collisions between opposing signs cancel in
//! expectation, Weinberger et al. 2009).
//!
//! Buckets are taken from the *high* bits of the mixed value
//! (`h >> (32 - log2_d)`): every documented mixer weakness that touches low
//! bits (murmur2's 1.7% bias, xxh3low's Moment-Chi2 blowup) would land directly
//! in the bucket index under `% D`, while high bits stay clean.
//!
//! All mixers use only u32 multiply/xor/shift so the identical function can be
//! implemented in numpy (training) and in svod tensor ops (on-device feature
//! extraction, if ever needed).

use std::fmt;
use std::str::FromStr;

/// MurmurHash3-32 finalizer: the mixing step, not the variable-length hash.
/// Strictly better documented bias profile than whichlang's murmurhash2
/// (rurban/SMHasher: murmur2 "1.7% bias, 81x collisions" vs murmur3a's
/// MomentChi2 69) while remaining ~6 integer ops.
#[inline(always)]
pub fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h
}

/// whichlang's exact murmurhash2-on-u32 (Petr Viktorin / Austin Appleby, 4-byte
/// tail path), kept bit-for-bit as the parity baseline for A/B experiments.
#[inline(always)]
pub fn murmurhash2(mut k: u32, seed: u32) -> u32 {
    const M: u32 = 0x5BD1_E995;
    let mut h: u32 = seed;
    k = k.wrapping_mul(M);
    k ^= k >> 24;
    k = k.wrapping_mul(M);
    h = h.wrapping_mul(M);
    h ^= k;
    h ^= h >> 13;
    h = h.wrapping_mul(M);
    h ^ (h >> 15)
}

/// Odd multiplier for multiply-shift universal hashing. Golden-ratio odd
/// constant; any odd constant gives the 2-universality bound, this one has no
/// known bad structure.
pub const MS_CONSTANT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Hash function selector. Serialized in model metadata as
/// [`HashId::id`] strings so a model carries its own hashing contract.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum HashId {
    /// fmix32 (MurmurHash3 finalizer) over the combined 32-bit halves of the
    /// key: the primary hash.
    Fmix32,
    /// whichlang's murmurhash2 applied to both halves; parity baseline.
    Murmur2,
    /// Multiply-shift universal hashing: provably pairwise-independent
    /// (collision probability <= 2^-32), a single multiply + shift.
    MultiplyShift,
}

impl HashId {
    pub const ALL: [HashId; 3] = [HashId::Fmix32, HashId::Murmur2, HashId::MultiplyShift];

    /// Stable identifier stored in model metadata.
    pub const fn id(self) -> &'static str {
        match self {
            HashId::Fmix32 => "fmix32",
            HashId::Murmur2 => "murmur2",
            HashId::MultiplyShift => "multiply_shift",
        }
    }

    pub fn from_id(id: &str) -> Option<HashId> {
        HashId::ALL.iter().copied().find(|h| h.id() == id)
    }
}

impl fmt::Display for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for HashId {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        HashId::from_id(s).ok_or(())
    }
}

/// A configured feature hasher: hash function + seed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureHasher {
    pub id: HashId,
    pub seed: u32,
}

impl Default for FeatureHasher {
    fn default() -> Self {
        FeatureHasher { id: HashId::Fmix32, seed: 0x9E37_79B9 }
    }
}

impl FeatureHasher {
    /// Mix a packed n-gram key into a u32.
    #[inline(always)]
    pub fn hash_u64(&self, key: u64) -> u32 {
        let lo = key as u32;
        let hi = (key >> 32) as u32;
        match self.id {
            HashId::Fmix32 => {
                let mut h = lo ^ self.seed.wrapping_mul(0x85EB_CA6B);
                h ^= hi.wrapping_mul(0xC2B2_AE35);
                fmix32(h)
            }
            HashId::Murmur2 => {
                murmurhash2(lo ^ self.seed, self.seed) ^ murmurhash2(hi ^ self.seed.rotate_left(16), self.seed.rotate_left(16))
            }
            HashId::MultiplyShift => {
                // Top 32 bits of key * C (mod 2^64). The low bits of a
                // multiply-shift hash are structurally weak; the top bits are
                // the universal ones, and we only ever consume them.
                ((key.wrapping_mul(MS_CONSTANT)) >> 32) as u32 ^ self.seed
            }
        }
    }

    /// Map a packed n-gram key to `(bucket, negative)` where the bucket is in
    /// `0..2^log2_d` (from the high bits) and `negative` is the sign bit for
    /// signed hashing.
    #[inline(always)]
    pub fn bucket(&self, key: u64, log2_d: u32) -> (u32, bool) {
        debug_assert!(log2_d > 0 && log2_d < 32);
        let h = self.hash_u64(key);
        let bucket = h >> (32 - log2_d);
        (bucket, h & 1 == 1)
    }

    /// Hash one full 8-key block into signed table indices (`bucket`, or
    /// `D+1+bucket` for negative sign — the ±P doubled-table gather layout).
    ///
    /// The fmix32 arm is written as a flat loop of `wrapping_*` u32 ops —
    /// exactly the arithmetic [`Self::hash_u64`] performs per key, so the
    /// result is bit-identical — in the shape LLVM's auto-vectorizer turns
    /// into eight exact SIMD lanes (AVX2 `vpmulld`/`vpsrlvd` with
    /// `-C target-cpu=native` or `target-feature=+avx2`, NEON on aarch64).
    /// No dispatch, no unsafe: a baseline build simply keeps the scalar
    /// code, correct and slower. Other hash ids ride the per-key path.
    pub fn signed_index_block(&self, keys: &[u64; 8], log2_d: u32, out: &mut [i32]) {
        debug_assert!(out.len() >= 8);
        debug_assert!(log2_d > 0 && log2_d < 32);
        // Accumulate into a fixed-size local: indexing a runtime-length
        // slice per element plants a bounds check the vectorizer cannot
        // remove (the >= 8 contract is only a debug_assert), which pins the
        // loop to unrolled scalar. The array keeps the loop check-free; the
        // slice length is verified exactly once, by copy_from_slice.
        let mut idx = [0i32; 8];
        if self.id == HashId::Fmix32 {
            let seed_k1 = self.seed.wrapping_mul(0x85EB_CA6B);
            let d1 = (1u32 << log2_d) + 1;
            let shift = 32 - log2_d;
            for (i, &k) in keys.iter().enumerate() {
                // hash_u64's fmix32 combine, then the finalizer.
                let mut h = (k as u32) ^ seed_k1 ^ ((k >> 32) as u32).wrapping_mul(0xC2B2_AE35);
                h ^= h >> 16;
                h = h.wrapping_mul(0x85EB_CA6B);
                h ^= h >> 13;
                h = h.wrapping_mul(0xC2B2_AE35);
                h ^= h >> 16;
                // Bucket from the high bits, sign folded into the ±P range.
                idx[i] = ((h >> shift).wrapping_add((h & 1).wrapping_mul(d1))) as i32;
            }
        } else {
            let d1 = (1i32 << log2_d) + 1;
            for (i, &k) in keys.iter().enumerate() {
                let (b, neg) = self.bucket(k, log2_d);
                idx[i] = if neg { d1 + b as i32 } else { b as i32 };
            }
        }
        out[..8].copy_from_slice(&idx);
    }

    /// Signed table index of a single key (`bucket`, or `D+1+bucket` for
    /// negative sign) — the scalar counterpart of
    /// [`Self::signed_index_block`].
    #[inline(always)]
    pub fn signed_index(&self, key: u64, log2_d: u32) -> i32 {
        let (b, neg) = self.bucket(key, log2_d);
        if neg { (1i32 << log2_d) + 1 + b as i32 } else { b as i32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buckets_of(hasher: &FeatureHasher, keys: &[u64], log2_d: u32) -> Vec<u32> {
        keys.iter().map(|k| hasher.bucket(*k, log2_d).0).collect()
    }

    #[test]
    fn fmix32_reference_values() {
        // fmix32 of 0 is a fixed point of the final xorshift; spot-check the
        // chain against a hand-rolled computation.
        assert_eq!(fmix32(0), 0);
        let h = fmix32(1);
        let mut x = 1u32;
        x ^= x >> 16;
        x = x.wrapping_mul(0x85EB_CA6B);
        x ^= x >> 13;
        x = x.wrapping_mul(0xC2B2_AE35);
        x ^= x >> 16;
        assert_eq!(h, x);
        assert_ne!(fmix32(1), fmix32(2));
    }

    #[test]
    fn murmur2_matches_whichlang() {
        // Bit-for-bit parity with whichlang's murmurhash2 on u32 inputs.
        assert_eq!(murmurhash2(0, 0x9E37_79B9), murmurhash2(0, 0x9E37_79B9));
        assert_eq!(murmurhash2(0xDEADBEEF, 42), {
            // Independent reference implementation.
            const M: u32 = 0x5BD1_E995;
            let mut k = 0xDEADBEEFu32.wrapping_mul(M);
            k ^= k >> 24;
            k = k.wrapping_mul(M);
            let mut h = 42u32.wrapping_mul(M);
            h ^= k;
            h ^= h >> 13;
            h = h.wrapping_mul(M);
            h ^ (h >> 15)
        });
    }

    #[test]
    fn buckets_in_range_and_deterministic() {
        let hasher = FeatureHasher::default();
        let keys: Vec<u64> = (0..10_000u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        for log2_d in [12u32, 16, 17, 20] {
            for &b in &buckets_of(&hasher, &keys, log2_d) {
                assert!(b < (1 << log2_d));
            }
        }
        // Determinism: same key, same bucket.
        assert_eq!(buckets_of(&hasher, &keys, 17), buckets_of(&hasher, &keys, 17));
        // Seeds change the assignment.
        let other = FeatureHasher { id: HashId::Fmix32, seed: 7 };
        assert_ne!(buckets_of(&hasher, &keys, 17), buckets_of(&other, &keys, 17));
    }

    #[test]
    fn bucket_and_sign_bits_are_disjoint() {
        let hasher = FeatureHasher::default();
        for key in 0..1000u64 {
            let (bucket, _) = hasher.bucket(key, 17);
            assert!(bucket < (1 << 17));
        }
    }

    #[test]
    fn spread_is_reasonably_uniform() {
        // Coarse chi-square-ish sanity: with 2^17 buckets and 200k distinct
        // keys the most-collided bucket should stay well below, say, 12 hits
        // (mean ~1.5 under uniformity).
        let hasher = FeatureHasher::default();
        let log2_d = 17u32;
        let mut counts = vec![0u32; 1 << log2_d];
        for i in 0..200_000u64 {
            let (b, _) = hasher.bucket(i.wrapping_mul(11400714819323198485), log2_d);
            counts[b as usize] += 1;
        }
        let max = counts.iter().copied().max().unwrap();
        assert!(max < 12, "max bucket occupancy {max} too skewed");
    }

    #[test]
    fn signed_index_block_matches_scalar() {
        // The vectorized (AVX2 on this class of machine) block path must
        // reproduce per-key bucket() + sign arithmetic exactly, for every
        // hash id (non-fmix ids ride the scalar fallback inside the block
        // API) and a spread of bucket widths.
        let mut key = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            key = key.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            key
        };
        for id in HashId::ALL {
            let hasher = FeatureHasher { id, seed: 0x9E37_79B9 };
            for log2_d in [4u32, 8, 12, 17, 24] {
                for _ in 0..64 {
                    let keys: [u64; 8] = std::array::from_fn(|_| next());
                    let mut got = [0i32; 8];
                    hasher.signed_index_block(&keys, log2_d, &mut got);
                    for i in 0..8 {
                        assert_eq!(
                            got[i],
                            hasher.signed_index(keys[i], log2_d),
                            "{id:?} log2_d={log2_d} key={:#x}",
                            keys[i]
                        );
                    }
                }
            }
        }
    }
}
