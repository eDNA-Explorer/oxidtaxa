//! R-compatible random number generator.
//!
//! R's `set.seed()` uses a custom initialization for MT19937 that differs
//! from the standard `init_genrand()`. R uses a LCG with multiplier 69069
//! to fill the state, preceded by 50 warmup iterations.
//!
//! The generation algorithm (twist + temper) is standard MT19937.

/// Mix a base seed with an index to produce an independent seed.
/// Uses SplitMix64 finalizer for good avalanche properties.
pub fn mix_seed(base: u32, index: u64) -> u32 {
    let mut z = base as u64 ^ index;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    z as u32
}

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908B0DF;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7FFFFFFF;

pub struct RRng {
    mt: [u32; N],
    mti: usize,
}

impl RRng {
    /// Create a new RNG matching R's `set.seed(seed)`.
    ///
    /// R's initialization (from src/main/RNG.c, RNG_Init):
    /// 1. Warm the seed through 50 LCG iterations: seed = 69069*seed + 1
    /// 2. Fill 625 words (mti + 624 state words) with more LCG iterations
    /// 3. Force mti = 624 (FixupSeeds), so first draw triggers a full twist
    pub fn new(seed_input: u32) -> Self {
        let mut seed = seed_input;

        // Step 1: Warm the seed (50 LCG iterations)
        for _ in 0..50 {
            seed = seed.wrapping_mul(69069).wrapping_add(1);
        }

        // Step 2: Fill 625 words (i_seed[0..625])
        // i_seed[0] = mti index, i_seed[1..625] = mt[0..624]
        let mut i_seed = [0u32; 625];
        for word in i_seed.iter_mut() {
            seed = seed.wrapping_mul(69069).wrapping_add(1);
            *word = seed;
        }

        // Step 3: FixupSeeds — force mti = 624
        // i_seed[0] is the index; force to N so first draw triggers twist
        let mut mt = [0u32; N];
        mt.copy_from_slice(&i_seed[1..625]);

        Self { mt, mti: N }
    }

    /// Standard MT19937 twist (generate_numbers).
    fn twist(&mut self) {
        let mag01 = [0u32, MATRIX_A];

        for kk in 0..(N - M) {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
            self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        }
        for kk in (N - M)..(N - 1) {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
            self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        }
        let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];

        self.mti = 0;
    }

    /// Generate next u32, matching R's MT_genrand().
    fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            self.twist();
        }

        let mut y = self.mt[self.mti];
        self.mti += 1;

        // Tempering
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= y >> 18;

        y
    }

    /// Generate a uniform random double in [0, 1).
    /// Matches R's `unif_rand()` from `src/main/RNG.c`:
    ///   `(double)genrand_int32() * 2.3283064365386963e-10`
    pub fn unif_rand(&mut self) -> f64 {
        let y = self.next_u32();
        y as f64 * 2.3283064365386963e-10
    }

    /// Generate random bits using R's `rbits()` function.
    /// Draws unif_rand() values in 16-bit chunks.
    ///
    /// From R's src/main/RNG.c:
    /// ```c
    /// static double rbits(int bits) {
    ///     uint_least64_t v = 0;
    ///     for (int n = 0; n <= bits; n += 16) {
    ///         int v1 = (int) floor(unif_rand() * 65536);
    ///         v = 65536 * v + v1;
    ///     }
    ///     return (double) (v & ((1uLL << bits) - 1));
    /// }
    /// ```
    fn rbits(&mut self, bits: u32) -> u64 {
        let mut v: u64 = 0;
        let mut n: i32 = 0;
        while n <= bits as i32 {
            let v1 = (self.unif_rand() * 65536.0).floor() as u64;
            v = 65536u64.wrapping_mul(v).wrapping_add(v1);
            n += 16;
        }
        v & ((1u64 << bits) - 1)
    }

    /// R's rejection-based uniform integer sampling (R >= 3.6.0 default).
    /// Returns a value in [0, n) (0-indexed).
    ///
    /// From R's src/main/RNG.c:
    /// ```c
    /// double R_unif_index(double dn) {
    ///     int bits = (int) ceil(log2(dn));
    ///     double dv;
    ///     do { dv = rbits(bits); } while (dn <= dv);
    ///     return dv;
    /// }
    /// ```
    fn r_unif_index(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let dn = n as f64;
        let bits = dn.log2().ceil() as u32;
        loop {
            let dv = self.rbits(bits) as f64;
            if dv < dn {
                return dv as usize;
            }
        }
    }

    /// Sample `size` integers from 0..n with replacement (0-indexed).
    /// Matches R's `sample.int(n, size, replace=TRUE) - 1`.
    ///
    /// R >= 3.6.0 uses rejection sampling by default (sample.kind="Rejection").
    pub fn sample_int_replace(&mut self, n: usize, size: usize) -> Vec<usize> {
        let mut buf = vec![0usize; size];
        self.sample_int_replace_into(n, &mut buf);
        buf
    }

    /// Write `buf.len()` random integers from 0..n into a pre-allocated slice.
    /// Avoids allocation in hot paths where the buffer can be reused.
    pub fn sample_int_replace_into(&mut self, n: usize, buf: &mut [usize]) {
        for slot in buf.iter_mut() {
            *slot = self.r_unif_index(n);
        }
    }

    /// Sample `size` elements from a slice with replacement.
    /// Matches R's `sample(x, size, replace=TRUE)`.
    pub fn sample_replace<T: Clone>(&mut self, x: &[T], size: usize) -> Vec<T> {
        let indices = self.sample_int_replace(x.len(), size);
        indices.iter().map(|&i| x[i].clone()).collect()
    }
}
