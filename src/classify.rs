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
/// `config.processors` controls the rayon thread pool size.
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
    // Build a local thread pool so processors is always respected.
    // build_global() can only succeed once per process — subsequent calls
    // silently fail, leaving Rayon's default (1 thread per core).
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.processors)
        .build()
        .expect("failed to create rayon thread pool");

    pool.install(|| {
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
    })
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
    if config.beam_width > 1 {
        return classify_one_pass_beam(my_kmers, s, b, ts, config, full_length, ls, rng);
    }

    let children = &ts.children;
    let fraction = &ts.fraction;
    let decision_kmers = &ts.decision_kmers;

    if my_kmers.len() <= s {
        return Some((ClassificationResult::unclassified("too_few_kmers"), 0.0));
    }

    // Greedy tree descent (beam_width=1)
    let mut k_node = 0usize;
    let mut w_indices: Vec<usize>;
    // I6: per-descent-step ratio `(top - runner_up) / b`, floored at 0.1, used
    // later in `leaf_phase_score` to discount per-rank confidences. Only
    // populated when `config.confidence_uses_descent_margin` is on.
    let mut descent_margins: Vec<f64> = Vec::new();
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

            if config.confidence_uses_descent_margin {
                let mut sorted = vote_counts.clone();
                sorted.sort_unstable_by(|a, b_| b_.cmp(a));
                let top = *sorted.first().unwrap_or(&0) as f64;
                let runner_up = *sorted.get(1).unwrap_or(&0) as f64;
                let margin = if top > 0.0 {
                    ((top - runner_up) / (b as f64)).max(0.1)
                } else {
                    1.0
                };
                descent_margins.push(margin);
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
            if children[subtrees[winner]].is_empty() {
                // I7: optionally widen to include any sibling with
                // vote_counts[j] >= 0.5 * b (winner always retained).
                w_indices = if config.sibling_aware_leaf {
                    let min_votes = ((b as f64) * 0.5) as usize;
                    vote_counts.iter().enumerate()
                        .filter(|(i, &c)| c >= min_votes || *i == winner)
                        .map(|(i, _)| i).collect()
                } else {
                    vec![winner]
                };
                break;
            }
            k_node = subtrees[winner];
        } else {
            if children[subtrees[0]].is_empty() { w_indices = vec![0]; break; }
            k_node = subtrees[0];
        }
    }

    leaf_phase_score(
        k_node, &w_indices, my_kmers, s, b, ts, config, full_length, ls, rng,
        &descent_margins,
    )
}

/// Beam search variant: maintain multiple candidate paths during tree descent.
#[allow(clippy::too_many_arguments)]
fn classify_one_pass_beam(
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
    let fraction = &ts.fraction;
    let decision_kmers = &ts.decision_kmers;

    if my_kmers.len() <= s {
        return Some((ClassificationResult::unclassified("too_few_kmers"), 0.0));
    }

    struct BeamCandidate {
        node: usize,
        w_indices: Vec<usize>,
        score: f64,
        /// Margin history from root to this candidate's current node. Mirrors
        /// the greedy path's `descent_margins` so I6 works for beam_width > 1.
        descent_margins: Vec<f64>,
    }

    let beam_width = config.beam_width;
    let mut active = vec![BeamCandidate {
        node: 0,
        w_indices: Vec::new(),
        score: 1.0,
        descent_margins: Vec::new(),
    }];

    loop {
        let mut next: Vec<BeamCandidate> = Vec::new();
        let mut any_expanded = false;

        for candidate in &active {
            let k_node = candidate.node;
            let subtrees = &children[k_node];
            let dk = &decision_kmers[k_node];

            if dk.is_none() || fraction[k_node].is_none() {
                next.push(BeamCandidate {
                    node: k_node,
                    w_indices: (0..subtrees.len()).collect(),
                    score: candidate.score,
                    descent_margins: candidate.descent_margins.clone(),
                });
                continue;
            }

            let dk = dk.as_ref().unwrap();
            let n = dk.keep.len();

            if n == 0 || subtrees.len() <= 1 {
                if subtrees.len() == 1 && !children[subtrees[0]].is_empty() {
                    next.push(BeamCandidate {
                        node: subtrees[0],
                        w_indices: Vec::new(),
                        score: candidate.score,
                        descent_margins: candidate.descent_margins.clone(),
                    });
                    any_expanded = true;
                } else {
                    next.push(BeamCandidate {
                        node: k_node,
                        w_indices: if subtrees.is_empty() { vec![] } else { (0..subtrees.len()).collect() },
                        score: candidate.score,
                        descent_margins: candidate.descent_margins.clone(),
                    });
                }
                continue;
            }

            // Vote at this node
            let frac = fraction[k_node].unwrap();
            let s_dk = ((n as f64) * frac).ceil() as usize;
            let sampling = rng.sample_int_replace(n, s_dk * b);
            let matches = int_match(&dk.keep, my_kmers);

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

            // Record the margin for this decision step (mirrors greedy at
            // classify.rs:230-241). All children produced from this candidate
            // inherit the parent's margin history plus this one.
            let next_margins = if config.confidence_uses_descent_margin {
                let mut sorted = vote_counts.clone();
                sorted.sort_unstable_by(|a, b_| b_.cmp(a));
                let top = *sorted.first().unwrap_or(&0) as f64;
                let runner_up = *sorted.get(1).unwrap_or(&0) as f64;
                let margin = if top > 0.0 {
                    ((top - runner_up) / (b as f64)).max(0.1)
                } else {
                    1.0
                };
                let mut m = candidate.descent_margins.clone();
                m.push(margin);
                m
            } else {
                candidate.descent_margins.clone()
            };

            // Collect children with votes, sorted by vote count descending
            let mut children_by_votes: Vec<(usize, usize)> = vote_counts.iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(j, &c)| (j, c))
                .collect();
            children_by_votes.sort_by(|a, b_| b_.1.cmp(&a.1));

            if children_by_votes.is_empty() {
                // No votes at all — terminal
                next.push(BeamCandidate {
                    node: k_node,
                    w_indices: (0..n_sub).collect(),
                    score: candidate.score,
                    descent_margins: next_margins,
                });
                continue;
            }

            let top_vote_frac = children_by_votes[0].1 as f64 / b as f64;

            if top_vote_frac >= config.min_descend {
                // Top child is confident — descend it
                let winner_idx = children_by_votes[0].0;
                let winner_child = subtrees[winner_idx];
                if children[winner_child].is_empty() {
                    next.push(BeamCandidate {
                        node: k_node,
                        w_indices: vec![winner_idx],
                        score: candidate.score * top_vote_frac,
                        descent_margins: next_margins.clone(),
                    });
                } else {
                    next.push(BeamCandidate {
                        node: winner_child,
                        w_indices: Vec::new(),
                        score: candidate.score * top_vote_frac,
                        descent_margins: next_margins.clone(),
                    });
                    any_expanded = true;
                }

                // Keep runner-ups for beam
                for &(j, votes) in children_by_votes.iter().skip(1) {
                    let vf = votes as f64 / b as f64;
                    let child = subtrees[j];
                    if children[child].is_empty() {
                        next.push(BeamCandidate {
                            node: k_node,
                            w_indices: vec![j],
                            score: candidate.score * vf,
                            descent_margins: next_margins.clone(),
                        });
                    } else {
                        next.push(BeamCandidate {
                            node: child,
                            w_indices: Vec::new(),
                            score: candidate.score * vf,
                            descent_margins: next_margins.clone(),
                        });
                        any_expanded = true;
                    }
                }
            } else {
                // No confident winner — terminal (same fallback as greedy)
                let w50: Vec<usize> = vote_counts.iter().enumerate()
                    .filter(|(_, &c)| c >= ((b as f64) * 0.5) as usize)
                    .map(|(i, _)| i).collect();
                let w_indices = if w50.is_empty() {
                    (0..vote_counts.len()).collect()
                } else {
                    let w_pos: Vec<usize> = vote_counts.iter().enumerate()
                        .filter(|(_, &c)| c > 0).map(|(i, _)| i).collect();
                    if w_pos.is_empty() { (0..vote_counts.len()).collect() } else { w_pos }
                };
                next.push(BeamCandidate {
                    node: k_node,
                    w_indices,
                    score: candidate.score * top_vote_frac,
                    descent_margins: next_margins,
                });
            }
        }

        // Prune to beam_width
        next.sort_by(|a, b_| b_.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        next.truncate(beam_width);
        active = next;

        if !any_expanded { break; }
    }

    // Score each candidate via leaf phase, pick best
    let mut best: Option<(ClassificationResult, f64)> = None;
    for candidate in &active {
        if let Some((result, sim)) = leaf_phase_score(
            candidate.node, &candidate.w_indices,
            my_kmers, s, b, ts, config, full_length, ls, rng,
            &candidate.descent_margins,
        ) {
            match &best {
                None => best = Some((result, sim)),
                Some((_, best_sim)) if sim > *best_sim => best = Some((result, sim)),
                _ => {}
            }
        }
    }
    best
}

/// Leaf-phase scoring: gather training sequences, bootstrap match, compute confidence.
/// Shared by both greedy and beam-search descent.
#[allow(clippy::too_many_arguments)]
fn leaf_phase_score(
    k_node: usize,
    w_indices: &[usize],
    my_kmers: &[i32],
    s: usize,
    b: usize,
    ts: &TrainingSet,
    config: &ClassifyConfig,
    full_length: (f64, f64),
    ls: &[usize],
    rng: &mut RRng,
    descent_margins: &[f64],
) -> Option<(ClassificationResult, f64)> {
    let children = &ts.children;
    let parents = &ts.parents;
    let sequences = &ts.sequences;
    let train_kmers = &ts.kmers;
    let cross_index = &ts.cross_index;
    let taxa = &ts.taxa;

    // Per-rank IDF: pick the row at the descent node's depth.
    let descent_depth = (ts.levels.get(k_node).copied().unwrap_or(1) - 1).max(0) as usize;
    let rank_idx = descent_depth.min(ts.idf_weights_by_rank.len().saturating_sub(1));
    let counts: &[f64] = &ts.idf_weights_by_rank[rank_idx];

    // Gather training sequences
    let subtrees = &children[k_node];
    let mut keep: Vec<usize> = Vec::new();
    for &wi in w_indices {
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
        if keep.is_empty() {
            return Some((ClassificationResult::unclassified("no_training_match"), 0.0));
        }
    }

    // Sample query k-mers
    let sampling: Vec<i32> = rng.sample_replace(my_kmers, s * b);

    // Group sampled k-mers by value. Use counting sort when sb is large enough
    // to justify the O(4^k) auxiliary allocation; fall back to comparison sort
    // for small inputs where the overhead dominates.
    let sb = s * b;
    let n_possible = 4usize.pow(ts.k as u32);

    let (u_sampling, positions, ranges) = if sb >= n_possible / 4 {
        // Counting sort: O(sb + 4^k). Wins when sb is comparable to 4^k.
        let mut kmer_counts = vec![0u32; n_possible + 1];
        for &km in &sampling {
            kmer_counts[km as usize] += 1;
        }

        let mut offsets = vec![0usize; n_possible + 1];
        for i in 1..=n_possible {
            offsets[i] = offsets[i - 1] + kmer_counts[i - 1] as usize;
        }

        let mut sorted_positions = vec![0usize; sb];
        let mut cursors = offsets.clone();
        for (idx, &km) in sampling.iter().enumerate() {
            let k = km as usize;
            sorted_positions[cursors[k]] = idx % b;
            cursors[k] += 1;
        }

        let mut us: Vec<i32> = Vec::new();
        let mut rg: Vec<usize> = vec![0];
        for km in 1..=n_possible {
            let c = kmer_counts[km] as usize;
            if c > 0 {
                us.push(km as i32);
                rg.push(rg.last().unwrap() + c);
            }
        }
        (us, sorted_positions, rg)
    } else {
        // Comparison sort: O(sb·log(sb)). Better for small inputs.
        let mut sort_idx: Vec<u32> = (0..sb as u32).collect();
        sort_idx.sort_unstable_by_key(|&i| sampling[i as usize]);

        let mut us: Vec<i32> = Vec::new();
        let mut pos: Vec<usize> = Vec::with_capacity(sb);
        let mut rg: Vec<usize> = vec![0];
        let mut i = 0;
        while i < sb {
            let kmer = sampling[sort_idx[i] as usize];
            us.push(kmer);
            while i < sb && sampling[sort_idx[i] as usize] == kmer {
                pos.push(sort_idx[i] as usize % b);
                i += 1;
            }
            rg.push(pos.len());
        }
        (us, pos, rg)
    };

    let u_weights: Vec<f64> = u_sampling.iter()
        .map(|&uk| if uk > 0 && (uk as usize) <= counts.len() { counts[(uk - 1) as usize] } else { 0.0 })
        .collect();

    let (mut hits_flat, mut sum_hits) = if let Some(ref inv_idx) = ts.inverted_index {
        parallel_match_inverted(&u_sampling, inv_idx, &keep, &u_weights, b, &positions, &ranges)
    } else {
        parallel_match(&u_sampling, train_kmers, &keep, &u_weights, b, &positions, &ranges)
    };
    if hits_flat.is_empty() {
        return Some((ClassificationResult::unclassified("no_training_match"), 0.0));
    }

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
        for &ti in top_hits_idx.iter() {
            let v = hits_flat[ti * b + rep];
            if v > max_val { max_val = v; }
        }
        if davg == 0.0 { continue; }
        // Per-replicate winner-take-all with tie splitting. R's IdTaxa uses
        // `max.col(..., ties.method = "random")` which randomly picks one of the
        // tied columns per replicate; in expectation each tied column receives
        // credit proportional to its share of ties. We deterministically split
        // credit equally among tied max columns, which matches that expected
        // value and ensures groups with bit-identical training sequences
        // surface as ties in the downstream `winners` filter at line 373.
        //
        // Note: `hits_flat` values can be negative when the IDF weights
        // (`counts`) contain negative entries, so we do not guard on
        // `max_val > 0` — doing so would drop real per-replicate contributions
        // whenever every group's score is negative (the non-tied branch here
        // matches the legacy accumulator bit-for-bit in that case).
        let mut n_tied: usize = 0;
        for &ti in top_hits_idx.iter() {
            if hits_flat[ti * b + rep] == max_val { n_tied += 1; }
        }
        let share = 1.0 / n_tied as f64;
        for (j, &ti) in top_hits_idx.iter().enumerate() {
            if hits_flat[ti * b + rep] == max_val {
                tot_hits[j] += hits_flat[ti * b + rep] / davg * share;
            }
        }
    }

    // Choose best group
    let max_tot = tot_hits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // I5: with `tie_margin > 0`, any group scoring within `(1 - tie_margin)`
    // of `max_tot` joins the tied set, feeding LCA-cap and `alternatives`.
    // Default `tie_margin = 0.0` preserves exact-equality semantics.
    let winners: Vec<usize> = if config.tie_margin > 0.0 && max_tot > 0.0 {
        let cutoff = max_tot * (1.0 - config.tie_margin);
        tot_hits.iter().enumerate()
            .filter(|(_, &v)| v >= cutoff).map(|(i, _)| i).collect()
    } else {
        tot_hits.iter().enumerate()
            .filter(|(_, &v)| v == max_tot).map(|(i, _)| i).collect()
    };
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

    // I6: discount each rank's confidence by the margin of the single
    // decision that selected it — no compounding across ranks.
    // `descent_margins[i]` is the decisiveness of the split at node i, which
    // picks rank i+1, so it discounts `confidences[i+1]` only. Root stays
    // untouched. Non-cumulative on purpose: margin is a decisiveness score,
    // not a probability, and raw per-rank confidence already encodes path
    // dependence via the bootstrap vote fraction — compounding margins on
    // top double-counts uncertainty and collapses deep ranks to zero.
    if config.confidence_uses_descent_margin && !descent_margins.is_empty() {
        for i in 1..confidences.len() {
            if let Some(&m) = descent_margins.get(i - 1) {
                confidences[i] *= m;
            }
        }
    }

    // When multiple groups are tied at `max_tot`, the classifier cannot honestly
    // resolve below the LCA of the tied set. Compute that LCA's position in
    // `predicteds` so the `above` filter can cap the reportable lineage there.
    //
    // Every non-selected winner's pairwise LCA with `selected` is the deepest
    // ancestor of `unique_groups[j]` that lives in `predicteds`. The group-wise
    // LCA is the shallowest (smallest index) of those pairwise LCAs, since it
    // must be an ancestor of every winner.
    let (lca_cap, alternatives): (Option<usize>, Vec<String>) = if winners.len() > 1 {
        let mut deepest_allowed = predicteds.len() - 1;
        for &j in &winners {
            if j == selected { continue; }
            let mut p = parents[unique_groups[j]];
            loop {
                if let Some(pos) = predicteds.iter().position(|&x| x == p) {
                    if pos < deepest_allowed { deepest_allowed = pos; }
                    break;
                }
                if p == 0 || parents[p] == p { break; }
                p = parents[p];
            }
        }
        let mut alts: Vec<String> = winners
            .iter()
            .map(|&w| taxa[unique_groups[w]].clone())
            .collect();
        alts.sort();
        (Some(deepest_allowed), alts)
    } else {
        (None, Vec::new())
    };

    // Build `above` as a contiguous prefix by breaking at the first rank that
    // fails its threshold. This guarantees the reported lineage is always a
    // valid rooted path (e.g., K;P;C), never skipping an intermediate rank when
    // `rank_thresholds` uses different values per depth. Confidences are
    // monotonically non-decreasing toward Root, so the default (global
    // threshold) case already produces a prefix — the loop form just makes the
    // contiguity invariant explicit.
    let mut above: Vec<usize> = Vec::with_capacity(confidences.len());
    for (i, &c) in confidences.iter().enumerate() {
        if let Some(cap) = lca_cap {
            if i > cap { break; }
        }
        let thresh = match &config.rank_thresholds {
            Some(rt) if i < rt.len() => rt[i],
            Some(rt) if !rt.is_empty() => *rt.last().unwrap(),
            _ => config.threshold,
        };
        if c >= thresh {
            above.push(i);
        } else {
            break;
        }
    }

    let result = if above.len() == predicteds.len() {
        ClassificationResult {
            taxon: predicteds.iter().map(|&p| taxa[p].clone()).collect(),
            confidence: confidences,
            alternatives: alternatives.clone(),
            reject_reason: None,
            similarity,
        }
    } else {
        let w = if above.is_empty() { vec![0] } else { above };
        let taxon: Vec<String> = w.iter().map(|&i| taxa[predicteds[i]].clone()).collect();
        let conf: Vec<f64> = w.iter().map(|&i| confidences[i]).collect();
        ClassificationResult {
            taxon,
            confidence: conf,
            alternatives,
            reject_reason: Some("below_threshold".to_string()),
            similarity,
        }
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
    let mut results = vec![ClassificationResult::unclassified("no_result"); n];
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
                    (None, None) => ClassificationResult::unclassified("no_result"),
                }
            } else {
                fwd.map(|(r, _)| r)
                    .unwrap_or_else(|| ClassificationResult::unclassified("no_result"))
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
