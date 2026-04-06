use rayon::prelude::*;

/// R's NA_INTEGER sentinel value (i32::MIN = -2147483648).
pub const NA_INTEGER: i32 = i32::MIN;

/// Convert ASCII DNA base to 0-3 index. Returns -1 for ambiguous bases.
/// Port of enumerate_sequence.c:alphabetFrequency.
#[inline]
fn base_to_index(b: u8) -> i8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => -1,
    }
}

/// Mask repeat regions by setting k-mers to NA_INTEGER.
/// Port of enumerate_sequence.c:maskRepeats (lines 49-104).
fn mask_repeats(x: &mut [i32], word_size: usize) {
    let l = x.len();
    let prob: f64 = 0.7f64.powi(word_size as i32);
    let missed: f64 = -48.35429;
    let max_mismatches = (missed / (1.0 - prob).ln()) as i32;

    let l1 = 1; // min period
    let l2 = 700; // max period
    let l3 = 25; // min length of repeat
    let l4 = max_mismatches; // max positions between matches

    let mut i: usize = 0;
    while i < l {
        if x[i] != NA_INTEGER {
            let mut broke_outer = false;
            for p in l1..=l2 {
                if i + p >= l {
                    break;
                }
                if x[i] == x[i + p] {
                    let mut m: f64 = 1.0;
                    let mut s: f64 = m - prob * p as f64;
                    let mut j = i + 1; // last match
                    let mut k = j; // position
                    let mut c = 0i32; // mismatch count

                    while k < l - p {
                        if x[k] == x[k + p] {
                            m += 1.0;
                            let t = m - prob * (k + p - i) as f64;
                            if t <= 0.0 {
                                break;
                            } else if t > s {
                                s = t;
                                k += 1;
                                j = k;
                            }
                            c = 0;
                        } else {
                            if c >= l4 {
                                break;
                            }
                            c += 1;
                            k += 1;
                        }
                    }

                    if s > 0.0
                        && (j as i64 - i as i64 + word_size as i64) > p as i64
                        && (j + p - i + word_size) > l3 as usize
                    {
                        for item in x.iter_mut().take((j + p).min(l)).skip(i + p) {
                            *item = NA_INTEGER;
                        }
                        i = (j + p).saturating_sub(1);
                        broke_outer = true;
                        break;
                    }
                }
            }
            if broke_outer {
                i += 1;
                continue;
            }
        }
        i += 1;
    }
}

/// Mask low-complexity regions using Pearson's chi-squared test.
/// Port of enumerate_sequence.c:maskSimple (lines 107-198).
fn mask_simple(x: &mut [i32], word_size: usize, n_bins: usize, window_size: usize, threshold: f64) {
    let l = x.len();
    let e: Vec<f64> = vec![0.25; n_bins]; // expected frequency

    let mut freq = vec![0.0f64; n_bins];
    let mut s: usize = 0; // sum of frequencies (window fill level)
    let mut c: usize = 0; // count of consecutive significant positions
    let mut pos = vec![0usize; word_size]; // store positions for masking
    let mut prev = vec![0i32; window_size]; // circular buffer of previous values
    let mut curr: usize = 0; // index in prev

    for j in 0..l {
        if s == window_size {
            let sum_val = prev[curr] as usize;
            freq[sum_val] -= 1.0;
            s -= 1;
        }
        let sum_val = x[j];
        if sum_val >= 0 && sum_val != NA_INTEGER {
            let bin = (sum_val as usize) % n_bins;
            prev[curr] = bin as i32;
            curr += 1;
            if curr == window_size {
                curr = 0;
            }
            freq[bin] += 1.0;
            s += 1;
        }

        let mut score = 0.0f64;
        for k in 0..n_bins {
            let expected = e[k] * s as f64;
            let temp = freq[k] - expected;
            score += temp * temp / expected;
        }

        if score > threshold {
            if c < word_size {
                pos[c] = j.saturating_sub(s / 2);
                c += 1;
            } else if c == word_size {
                for p in &pos[..word_size] {
                    x[*p] = NA_INTEGER;
                }
                let mask_pos = j.saturating_sub(s / 2);
                if mask_pos < l {
                    x[mask_pos] = NA_INTEGER;
                }
                c += 1;
            } else {
                let mask_pos = j.saturating_sub(s / 2);
                if mask_pos < l {
                    x[mask_pos] = NA_INTEGER;
                }
            }
        } else {
            c = 0;
        }
    }

    // Drain remaining window
    let j = l; // maintain j for position calculation
    while s > 1 {
        if curr == 0 {
            curr = window_size;
        }
        curr -= 1;
        let sum_val = prev[curr] as usize;
        freq[sum_val] -= 1.0;

        let mut score = 0.0f64;
        for k in 0..n_bins {
            let expected = e[k] * s as f64;
            let temp = freq[k] - expected;
            score += temp * temp / expected;
        }

        if score > threshold {
            if c < word_size {
                pos[c] = j.saturating_sub(s / 2);
                c += 1;
            } else if c == word_size {
                for p in &pos[..word_size] {
                    if *p < l {
                        x[*p] = NA_INTEGER;
                    }
                }
                let mask_pos = j.saturating_sub(s / 2);
                if mask_pos < l {
                    x[mask_pos] = NA_INTEGER;
                }
                c += 1;
            } else {
                let mask_pos = j.saturating_sub(s / 2);
                if mask_pos < l {
                    x[mask_pos] = NA_INTEGER;
                }
            }
        } else {
            c = 0;
        }
        s -= 1;
    }
    let _ = j; // suppress unused warning
}

/// Mask k-mers that appear more than `max_count` times.
/// Port of enumerate_sequence.c:maskNumerous (lines 201-270).
#[allow(clippy::needless_range_loop)]
fn mask_numerous(x: &mut [i32], max_count: i32, total_possible: usize, word_size: usize) {
    let l = x.len();
    let max_collisions = 100usize;
    let modulus = if l < total_possible { l } else { total_possible };

    if modulus == 0 {
        return;
    }

    let mut counts = vec![0i32; modulus];
    let mut keys = vec![0i32; modulus];

    // Count k-mers using open-addressing hash table
    for i in 0..l {
        if x[i] != NA_INTEGER {
            let mut k = 0u64;
            loop {
                let j = ((x[i] as u64).wrapping_add(k * (k + 1) / 2)) % modulus as u64;
                let j = j as usize;
                if counts[j] == 0 {
                    counts[j] = 1;
                    keys[j] = x[i];
                    break;
                } else if x[i] == keys[j] {
                    counts[j] += 1;
                    break;
                }
                k += 1;
                if k as usize >= max_collisions {
                    break;
                }
            }
        }
    }

    // Mask k-mers that are too numerous
    let mut consecutive = 0usize;
    for i in 0..l {
        if x[i] == NA_INTEGER {
            consecutive += 1;
        } else {
            let mut k = 0u64;
            loop {
                let j = ((x[i] as u64).wrapping_add(k * (k + 1) / 2)) % modulus as u64;
                let j = j as usize;
                if x[i] == keys[j] {
                    if counts[j] > max_count {
                        consecutive += 1;
                        if consecutive == word_size {
                            for back in 0..word_size {
                                x[i - back] = NA_INTEGER;
                            }
                        } else if consecutive > word_size {
                            x[i] = NA_INTEGER;
                        }
                    } else {
                        consecutive = 0;
                    }
                    break;
                }
                k += 1;
                if k as usize >= max_collisions {
                    break;
                }
            }
        }
    }
}

/// Enumerate k-mers for a single sequence.
/// Port of the inner loop of enumerate_sequence.c (lines 341-378).
/// Uses a fixed-size array for the sliding window (max K=15) to avoid
/// Vec::remove(0) overhead in the hot inner loop.
#[allow(clippy::needless_range_loop)]
fn enumerate_single(
    seq: &[u8],
    word_size: usize,
    pwv: &[i32],
    mask_reps: bool,
    mask_lcrs: bool,
    mask_num: Option<i32>,
) -> Vec<i32> {
    let len = seq.len();
    if len < word_size || word_size == 0 {
        return Vec::new();
    }

    let n_kmers = len - word_size + 1;
    let mut result = vec![0i32; n_kmers];

    // Fixed-size array for sliding window (max K=15, matching C implementation)
    let mut bases = [0i8; 16];
    for j in 0..(word_size - 1) {
        bases[j] = base_to_index(seq[j]);
    }

    for j in (word_size - 1)..len {
        bases[word_size - 1] = base_to_index(seq[j]);

        let mut sum = bases[0] as i32 * pwv[0];
        let mut ambiguous = bases[0] < 0;
        for k in 1..word_size {
            sum += bases[k] as i32 * pwv[k];
            if bases[k] < 0 {
                ambiguous = true;
            }
        }

        let pos = j + 1 - word_size;
        result[pos] = if ambiguous { NA_INTEGER } else { sum };

        // Shift left: matches C's bases[k-1] = bases[k] on a stack array
        for k in 0..(word_size - 1) {
            bases[k] = bases[k + 1];
        }
    }

    // Apply masking
    if mask_reps {
        mask_repeats(&mut result, word_size);
    }
    if mask_lcrs {
        // Two passes with different window sizes and thresholds (matching C)
        mask_simple(&mut result, word_size, 4, 20, 12.66667);
        mask_simple(&mut result, word_size, 4, 95, 38.90749);
    }
    if let Some(max_count) = mask_num {
        let tot = 4usize.pow(word_size as u32);
        mask_numerous(&mut result, max_count, tot, word_size);
    }

    result
}

/// Enumerate k-mers for a batch of sequences.
/// Port of enumerate_sequence.c:enumerateSequence.
///
/// Uses rayon for parallelism (replaces OpenMP).
pub fn enumerate_sequences(
    sequences: &[String],
    word_size: usize,
    mask_reps: bool,
    mask_lcrs: bool,
    mask_num: &[i32],
    fast_moving_side: bool,
) -> Vec<Vec<i32>> {
    // Build position weight vector
    let mut pwv = vec![0i32; word_size];
    if word_size > 0 {
        if fast_moving_side {
            pwv[0] = 1;
            for i in 1..word_size {
                pwv[i] = pwv[i - 1] * 4;
            }
        } else {
            pwv[word_size - 1] = 1;
            for i in (0..word_size - 1).rev() {
                pwv[i] = pwv[i + 1] * 4;
            }
        }
    }

    sequences
        .par_iter()
        .enumerate()
        .map(|(i, seq)| {
            let mn = if mask_num.len() == sequences.len() {
                Some(mask_num[i])
            } else {
                None
            };
            enumerate_single(seq.as_bytes(), word_size, &pwv, mask_reps, mask_lcrs, mn)
        })
        .collect()
}
