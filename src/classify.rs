use std::collections::HashMap;

use crate::kmer::{enumerate_sequences, parse_seed_pattern, SpacedSeed, NA_INTEGER};
use crate::matching::{int_match, parallel_match, parallel_match_inverted, vector_sum};
use crate::rng::RRng;
use crate::sequence::reverse_complement;
use crate::types::{ClassificationResult, ClassifyConfig, OutputType, StrandMode, TrainingSet};

/// Pre-computed per-sequence data shared across classification calls.
struct PrecomputedData {
    test_kmers: Vec<Vec<i32>>,
    rev_kmers: Vec<Vec<i32>>,
    s_values: Vec<usize>,
    b_values: Vec<usize>,
    boths: Vec<usize>,
    /// Maps original sequence index → index in rev_kmers for O(1) lookup.
    boths_map: HashMap<usize, usize>,
    ls: Vec<usize>,
    full_length: (f64, f64),
}

/// Classify sequences using a trained IDTAXA model.
///
/// Dual-mode execution:
/// - `deterministic=true`: Sequential outer loop, single shared PRNG.
///   Matches R output exactly. Used for golden tests.
/// - `deterministic=false`: Per-sequence classification parallelized via rayon.
///   Each sequence gets an independent PRNG seeded with `seed ^ index`.
///   Statistically equivalent but not bit-identical to R.
///
/// `config.processors` controls the rayon global thread pool size.
#[allow(clippy::too_many_arguments)]
pub fn id_taxa(
    test_sequences: &[String],
    _test_names: &[String],
    training_set: &TrainingSet,
    config: &ClassifyConfig,
    strand_mode: StrandMode,
    _output_type: OutputType,
    seed: u32,
    deterministic: bool,
) -> Vec<ClassificationResult> {
    // Configure rayon thread pool
    if config.processors > 1 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(config.processors)
            .build_global();
    }

    // De-replicate
    let (unique_seqs, unique_map, unique_strands) = dereplicate(test_sequences, strand_mode);
    let l = unique_seqs.len();
    if l == 0 {
        return Vec::new();
    }

    // Handle bottom strand
    let mut seqs = unique_seqs.clone();
    for (i, &s) in unique_strands.iter().enumerate() {
        if s == 3 {
            seqs[i] = reverse_complement(&seqs[i]);
        }
    }

    // Pre-compute shared data
    let pre = precompute(&seqs, &unique_strands, training_set, config);

    if deterministic {
        let mut rng = RRng::new(seed);
        let results = classify_sequential(
            &pre, &unique_strands, training_set, config, &mut rng,
        );
        unique_map.iter().map(|&i| results[i].clone()).collect()
    } else {
        let results = classify_parallel(
            &pre, &unique_strands, training_set, config, seed,
        );
        unique_map.iter().map(|&i| results[i].clone()).collect()
    }
}

/// Pre-compute k-mers, S, B, and reverse complement k-mers.
fn precompute(
    seqs: &[String],
    strands: &[i32],
    ts: &TrainingSet,
    config: &ClassifyConfig,
) -> PrecomputedData {
    let k = ts.k;
    let n_kmers_total = 4usize.pow(k as u32);
    let ls: Vec<usize> = ts.kmers.iter().map(|km| km.len()).collect();
    let min_s = compute_min_sample_size(&ls, n_kmers_total, ts.kmers.len());

    // Parse spaced seed from model (authoritative source)
    let spaced_seed: Option<SpacedSeed> = ts.seed_pattern.as_ref()
        .map(|pat| parse_seed_pattern(pat).expect("Invalid seed pattern in model"));

    let raw_kmers = enumerate_sequences(seqs, k, false, false, &[], true, spaced_seed.as_ref());
    let not_nas: Vec<usize> = raw_kmers
        .iter()
        .map(|v| v.iter().filter(|&&x| x != NA_INTEGER).count())
        .collect();

    let bootstraps = config.bootstraps;
    let s_values: Vec<usize> = not_nas
        .iter()
        .map(|&nn| (nn as f64).powf(config.sample_exponent).ceil().max(min_s as f64) as usize)
        .collect();

    let test_kmers: Vec<Vec<i32>> = raw_kmers
        .into_iter()
        .map(|v| {
            let mut sorted: Vec<i32> = v.into_iter()
                .filter(|&x| x != NA_INTEGER).map(|x| x + 1).collect();
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        })
        .collect();

    let b_values: Vec<usize> = test_kmers
        .iter()
        .zip(s_values.iter())
        .map(|(km, &s)| {
            if s == 0 { bootstraps }
            else { (5.0 * km.len() as f64 / s as f64).min(bootstraps as f64).max(1.0) as usize }
        })
        .collect();

    let boths: Vec<usize> = strands.iter().enumerate()
        .filter(|(_, &s)| s == 1).map(|(i, _)| i).collect();

    let boths_map: HashMap<usize, usize> = boths.iter().enumerate()
        .map(|(rev_idx, &orig_idx)| (orig_idx, rev_idx))
        .collect();

    let rev_kmers = if !boths.is_empty() {
        let rev_seqs: Vec<String> = boths.iter().map(|&i| reverse_complement(&seqs[i])).collect();
        let raw = enumerate_sequences(&rev_seqs, k, false, false, &[], true, spaced_seed.as_ref());
        raw.into_iter()
            .map(|v| {
                let mut sorted: Vec<i32> = v.into_iter()
                    .filter(|&x| x != NA_INTEGER).map(|x| x + 1).collect();
                sorted.sort_unstable();
                sorted.dedup();
                sorted
            })
            .collect()
    } else {
        Vec::new()
    };

    let full_length = if config.full_length == 0.0 {
        (0.0, f64::INFINITY)
    } else {
        (1.0 / config.full_length, config.full_length)
    };

    PrecomputedData { test_kmers, rev_kmers, s_values, b_values, boths, boths_map, ls, full_length }
}

/// Classify a single pass (one strand) of one sequence.
/// Returns (result, similarity) or None if not enough k-mers.
#[allow(clippy::too_many_arguments)]
fn classify_one_pass(
    my_kmers: &[i32],
    s: usize,
    b: usize,
    ts: &TrainingSet,
    config: &ClassifyConfig,
    full_length: (f64, f64),
    ls: &[usize],
    rng: &mut RRng,
) -> Option<(ClassificationResult, f64)> {
    let children = &ts.children;
    let parents = &ts.parents;
    let fraction = &ts.fraction;
    let sequences = &ts.sequences;
    let train_kmers = &ts.kmers;
    let cross_index = &ts.cross_index;
    let counts = &ts.idf_weights;
    let decision_kmers = &ts.decision_kmers;
    let taxa = &ts.taxa;

    if my_kmers.len() <= s {
        return None;
    }

    // Tree descent
    let mut k_node = 0usize;
    let mut w_indices: Vec<usize>;
    loop {
        let subtrees = &children[k_node];
        let dk = &decision_kmers[k_node];
        if dk.is_none() || fraction[k_node].is_none() {
            w_indices = (0..subtrees.len()).collect();
            break;
        }
        let dk = dk.as_ref().unwrap();
        let n = dk.keep.len();
        if n == 0 {
            w_indices = (0..subtrees.len()).collect();
            break;
        } else if subtrees.len() > 1 {
            let frac = fraction[k_node].unwrap();
            let s_dk = ((n as f64) * frac).ceil() as usize;
            let sampling = rng.sample_int_replace(n, s_dk * b);
            let matches = int_match(&dk.keep, my_kmers);
            // Flat hits: subtrees.len() rows x b columns, single allocation
            let n_sub = subtrees.len();
            let mut hits_flat = vec![0.0f64; n_sub * b];
            for j in 0..n_sub {
                let row = vector_sum(&matches, &dk.profiles[j], &sampling, b);
                hits_flat[j * b..(j + 1) * b].copy_from_slice(&row);
            }
            let mut vote_counts = vec![0usize; n_sub];
            for rep in 0..b {
                let max_val = (0..n_sub).map(|j| hits_flat[j * b + rep]).fold(0.0f64, f64::max);
                if max_val > 0.0 {
                    for j in 0..n_sub {
                        if hits_flat[j * b + rep] == max_val { vote_counts[j] += 1; }
                    }
                }
            }
            let w: Vec<usize> = vote_counts.iter().enumerate()
                .filter(|(_, &c)| c >= (config.min_descend * b as f64) as usize)
                .map(|(i, _)| i).collect();
            if w.len() != 1 {
                let w50: Vec<usize> = vote_counts.iter().enumerate()
                    .filter(|(_, &c)| c >= ((b as f64) * 0.5) as usize)
                    .map(|(i, _)| i).collect();
                if w50.is_empty() {
                    w_indices = (0..vote_counts.len()).collect();
                } else {
                    w_indices = vote_counts.iter().enumerate()
                        .filter(|(_, &c)| c > 0).map(|(i, _)| i).collect();
                    if w_indices.is_empty() { w_indices = (0..vote_counts.len()).collect(); }
                }
                break;
            }
            let winner = w[0];
            if children[subtrees[winner]].is_empty() { w_indices = vec![winner]; break; }
            k_node = subtrees[winner];
        } else {
            if children[subtrees[0]].is_empty() { w_indices = vec![0]; break; }
            k_node = subtrees[0];
        }
    }

    // Gather training sequences
    let subtrees = &children[k_node];
    let mut keep: Vec<usize> = Vec::new();
    for &wi in &w_indices {
        if wi < subtrees.len() {
            if let Some(ref sq) = sequences[subtrees[wi]] { keep.extend(sq); }
        }
    }
    if full_length.0 > 0.0 || full_length.1.is_finite() {
        let my_len = my_kmers.len() as f64;
        keep.retain(|&idx| {
            let tl = ls[idx] as f64;
            tl >= full_length.0 * my_len && tl <= full_length.1 * my_len
        });
        if keep.is_empty() { return None; }
    }

    // Sample query k-mers
    let sampling: Vec<i32> = rng.sample_replace(my_kmers, s * b);

    // Build sorted unique k-mers and position mapping without cloning sampling.
    // Sort indices by k-mer value to group identical k-mers together.
    let sb = s * b;
    let mut sort_idx: Vec<u32> = (0..sb as u32).collect();
    sort_idx.sort_unstable_by_key(|&i| sampling[i as usize]);

    // Build u_sampling, positions, ranges in a single pass over sorted indices
    let mut u_sampling: Vec<i32> = Vec::new();
    let mut positions: Vec<usize> = Vec::with_capacity(sb);
    let mut ranges: Vec<usize> = vec![0];
    {
        let mut i = 0;
        while i < sb {
            let kmer = sampling[sort_idx[i] as usize];
            u_sampling.push(kmer);
            while i < sb && sampling[sort_idx[i] as usize] == kmer {
                positions.push(sort_idx[i] as usize % b);
                i += 1;
            }
            ranges.push(positions.len());
        }
    }

    let u_weights: Vec<f64> = u_sampling.iter()
        .map(|&uk| if uk > 0 && (uk as usize) <= counts.len() { counts[(uk - 1) as usize] } else { 0.0 })
        .collect();

    let (mut hits_flat, mut sum_hits) = if let Some(ref inv_idx) = ts.inverted_index {
        parallel_match_inverted(&u_sampling, inv_idx, &keep, &u_weights, b, &positions, &ranges)
    } else {
        parallel_match(&u_sampling, train_kmers, &keep, &u_weights, b, &positions, &ranges)
    };
    if hits_flat.is_empty() { return None; }

    // Length normalization: scale each training sequence's scores by sqrt(n_unique / avg)
    if config.length_normalize {
        let avg_unique: f64 = keep.iter().map(|&idx| ls[idx] as f64).sum::<f64>() / keep.len() as f64;
        for (k_idx, &seq_idx) in keep.iter().enumerate() {
            let n_unique = ls[seq_idx] as f64;
            if n_unique > 0.0 && avg_unique > 0.0 {
                let norm_factor = (n_unique / avg_unique).sqrt();
                let base = k_idx * b;
                for rep in 0..b {
                    hits_flat[base + rep] /= norm_factor;
                }
                sum_hits[k_idx] /= norm_factor;
            }
        }
    }

    // Find top hit per group — O(n_keep) single-pass matching C's groupMax approach
    let lookup: Vec<usize> = keep.iter().map(|&idx| cross_index[idx]).collect();

    // Sort indices by group for single-pass groupMax
    let mut order: Vec<usize> = (0..lookup.len()).collect();
    order.sort_unstable_by_key(|&i| lookup[i]);

    let mut unique_groups: Vec<usize> = Vec::new();
    let mut top_hits_idx: Vec<usize> = Vec::new();
    {
        let mut i = 0;
        while i < order.len() {
            let group = lookup[order[i]];
            unique_groups.push(group);
            let mut best_idx = order[i];
            let mut best_val = sum_hits[order[i]];
            i += 1;
            while i < order.len() && lookup[order[i]] == group {
                if sum_hits[order[i]] > best_val {
                    best_val = sum_hits[order[i]];
                    best_idx = order[i];
                }
                i += 1;
            }
            top_hits_idx.push(best_idx);
        }
    }

    // Compute confidence using flat hits matrix
    // Compute davg without allocating sampling_weights Vec
    let davg = {
        let mut row_sums = vec![0.0f64; b];
        for (idx, &sk) in sampling.iter().enumerate() {
            let w = if sk > 0 && (sk as usize) <= counts.len() { counts[(sk - 1) as usize] } else { 0.0 };
            row_sums[idx % b] += w;
        }
        row_sums.iter().sum::<f64>() / b as f64
    };

    let n_top = top_hits_idx.len();
    let mut tot_hits = vec![0.0f64; n_top];
    for rep in 0..b {
        let mut max_val = f64::NEG_INFINITY;
        let mut max_col = 0usize;
        for (j, &ti) in top_hits_idx.iter().enumerate() {
            let v = hits_flat[ti * b + rep];
            if v > max_val { max_val = v; max_col = j; }
        }
        if davg != 0.0 { tot_hits[max_col] += hits_flat[top_hits_idx[max_col] * b + rep] / davg; }
    }

    // Choose best group
    let max_tot = tot_hits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let winners: Vec<usize> = tot_hits.iter().enumerate()
        .filter(|(_, &v)| v == max_tot).map(|(i, _)| i).collect();
    let selected = if winners.len() > 1 {
        let idx = rng.sample_int_replace(winners.len(), 1)[0];
        winners[idx]
    } else {
        winners[0]
    };

    let similarity = if davg != 0.0 {
        let base = top_hits_idx[selected] * b;
        hits_flat[base..base + b].iter().sum::<f64>() / davg
    } else { 0.0 };

    // Build prediction
    let predicted_group = unique_groups[selected];
    let mut predicteds: Vec<usize> = Vec::new();
    let mut p = predicted_group;
    loop {
        predicteds.push(p);
        if p == 0 || parents[p] == p { break; }
        p = parents[p];
    }
    predicteds.reverse();

    let base_confidence = tot_hits[selected] / b as f64 * 100.0;
    let mut confidences = vec![base_confidence; predicteds.len()];
    for (j, &th) in tot_hits.iter().enumerate() {
        if th > 0.0 && j != selected {
            let mut p = parents[unique_groups[j]];
            loop {
                if let Some(m) = predicteds.iter().position(|&x| x == p) {
                    confidences[m] += th / b as f64 * 100.0;
                }
                if p == 0 || parents[p] == p { break; }
                p = parents[p];
            }
        }
    }

    let above: Vec<usize> = confidences.iter().enumerate()
        .filter(|(i, &c)| {
            let thresh = match &config.rank_thresholds {
                Some(rt) if *i < rt.len() => rt[*i],
                Some(rt) if !rt.is_empty() => *rt.last().unwrap(),
                _ => config.threshold,
            };
            c >= thresh
        })
        .map(|(i, _)| i).collect();

    let result = if above.len() == predicteds.len() {
        ClassificationResult {
            taxon: predicteds.iter().map(|&p| taxa[p].clone()).collect(),
            confidence: confidences,
        }
    } else {
        let w = if above.is_empty() { vec![0] } else { above };
        let last_w = *w.last().unwrap();
        let mut taxon: Vec<String> = w.iter().map(|&i| taxa[predicteds[i]].clone()).collect();
        taxon.push(format!("unclassified_{}", taxa[predicteds[last_w]]));
        let mut conf: Vec<f64> = w.iter().map(|&i| confidences[i]).collect();
        conf.push(confidences[last_w]);
        ClassificationResult { taxon, confidence: conf }
    };

    Some((result, similarity))
}

/// Sequential classification: single shared PRNG, matches R output.
fn classify_sequential(
    pre: &PrecomputedData,
    _strands: &[i32],
    ts: &TrainingSet,
    config: &ClassifyConfig,
    rng: &mut RRng,
) -> Vec<ClassificationResult> {
    let n = pre.test_kmers.len();
    let mut results = vec![ClassificationResult::unclassified(); n];
    let mut sims = vec![0.0f64; n];

    // Build iteration order: forward then reverse
    let mut iteration: Vec<(usize, Option<usize>)> = Vec::new();
    for i in 0..n { iteration.push((i, None)); }
    for (rev_idx, &orig_idx) in pre.boths.iter().enumerate() {
        iteration.push((orig_idx, Some(rev_idx)));
    }

    for &(seq_idx, rev_idx) in &iteration {
        let my_kmers = match rev_idx {
            Some(ri) => &pre.rev_kmers[ri],
            None => &pre.test_kmers[seq_idx],
        };
        let s = pre.s_values[seq_idx];
        let b = pre.b_values[seq_idx];

        if let Some((result, similarity)) = classify_one_pass(
            my_kmers, s, b, ts, config, pre.full_length, &pre.ls, rng,
        ) {
            if let Some(_ri) = rev_idx {
                // Second pass: only replace if better similarity
                if similarity <= sims[seq_idx] { continue; }
            } else {
                sims[seq_idx] = similarity;
            }
            results[seq_idx] = result;
        }
    }

    results
}

/// Parallel classification: each sequence gets its own PRNG.
/// For strand="both", each thread classifies both strands internally.
fn classify_parallel(
    pre: &PrecomputedData,
    _strands: &[i32], // strand info already encoded in pre.boths
    ts: &TrainingSet,
    config: &ClassifyConfig,
    seed: u32,
) -> Vec<ClassificationResult> {
    use rayon::prelude::*;

    let n = pre.test_kmers.len();

    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = RRng::new(seed ^ (i as u32));
            let s = pre.s_values[i];
            let b = pre.b_values[i];

            // Forward pass
            let fwd = classify_one_pass(
                &pre.test_kmers[i], s, b, ts, config, pre.full_length, &pre.ls, &mut rng,
            );

            // Reverse pass (if "both" strand)
            if let Some(&both_pos) = pre.boths_map.get(&i) {
                let rev = classify_one_pass(
                    &pre.rev_kmers[both_pos], s, b, ts, config, pre.full_length, &pre.ls, &mut rng,
                );
                // Pick the strand with higher similarity
                match (fwd, rev) {
                    (Some((fwd_r, fwd_s)), Some((rev_r, rev_s))) => {
                        if rev_s > fwd_s { rev_r } else { fwd_r }
                    }
                    (Some((r, _)), None) => r,
                    (None, Some((r, _))) => r,
                    (None, None) => ClassificationResult::unclassified(),
                }
            } else {
                fwd.map(|(r, _)| r).unwrap_or_else(ClassificationResult::unclassified)
            }
        })
        .collect()
}

/// De-replicate sequences, returning (unique_seqs, map, strand_per_unique).
fn dereplicate(
    sequences: &[String],
    strand_mode: StrandMode,
) -> (Vec<String>, Vec<usize>, Vec<i32>) {
    let strand_val = match strand_mode {
        StrandMode::Both => 1,
        StrandMode::Top => 2,
        StrandMode::Bottom => 3,
    };
    let mut seen: HashMap<&str, usize> = HashMap::with_capacity(sequences.len());
    let mut unique_seqs: Vec<String> = Vec::new();
    let mut map: Vec<usize> = Vec::with_capacity(sequences.len());
    for seq in sequences.iter() {
        if let Some(&idx) = seen.get(seq.as_str()) {
            map.push(idx);
        } else {
            let idx = unique_seqs.len();
            seen.insert(seq.as_str(), idx);
            map.push(idx);
            unique_seqs.push(seq.clone());
        }
    }
    let strands = vec![strand_val; unique_seqs.len()];
    (unique_seqs, map, strands)
}

/// Compute minimum sample size to avoid false positives.
fn compute_min_sample_size(ls: &[usize], n_kmers: usize, n_seqs: usize) -> usize {
    let l_quantile = {
        let mut sorted_ls: Vec<usize> = ls.to_vec();
        sorted_ls.sort_unstable();
        let idx = ((sorted_ls.len() as f64 - 1.0) * 0.9) as usize;
        sorted_ls[idx.min(sorted_ls.len() - 1)] as f64
    };
    for min_s in (2..=100).step_by(2) {
        let min_s_f = min_s as f64;
        let p_single = 1.0 - pbinom((min_s_f * 0.5 - 1.0).floor() as usize, min_s, l_quantile / n_kmers as f64);
        let p_any = 1.0 - pbinom(0, n_seqs, p_single);
        if p_any < 0.01 { return min_s; }
    }
    100
}

/// Cumulative binomial distribution function P(X <= k).
fn pbinom(k: usize, n: usize, p: f64) -> f64 {
    if p <= 0.0 { return 1.0; }
    if p >= 1.0 { return if k >= n { 1.0 } else { 0.0 }; }
    let mut sum = 0.0f64;
    let mut binom_coeff = 1.0f64;
    let q = 1.0 - p;
    for i in 0..=k.min(n) {
        if i > 0 { binom_coeff *= (n - i + 1) as f64 / i as f64; }
        sum += binom_coeff * p.powi(i as i32) * q.powi((n - i) as i32);
    }
    sum.min(1.0)
}
