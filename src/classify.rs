use std::collections::HashMap;

use crate::kmer::{enumerate_sequences, parse_seed_pattern, SpacedSeed, NA_INTEGER};
use crate::matching::{
    int_match, match_selected_rows, match_selected_rows_inverted, match_sums, match_sums_inverted,
    vector_sum,
};
use crate::rng::RRng;
use crate::sequence::reverse_complement;
use crate::types::{
    ClassificationResult, ClassifyConfig, DeepestRankTopK, LeafAggregationMode, OutputType,
    RankTraceRow, StrandMode, TrainingSet,
};

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
            let results =
                classify_sequential(&pre, &unique_strands, training_set, config, &mut rng);
            unique_map.iter().map(|&i| results[i].clone()).collect()
        } else {
            let results = classify_parallel(&pre, &unique_strands, training_set, config, seed);
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
    let spaced_seed: Option<SpacedSeed> = ts
        .seed_pattern
        .as_ref()
        .map(|pat| parse_seed_pattern(pat).expect("Invalid seed pattern in model"));

    let raw_kmers = enumerate_sequences(seqs, k, false, false, &[], true, spaced_seed.as_ref());
    let not_nas: Vec<usize> = raw_kmers
        .iter()
        .map(|v| v.iter().filter(|&&x| x != NA_INTEGER).count())
        .collect();

    let bootstraps = config.bootstraps;
    let s_values: Vec<usize> = not_nas
        .iter()
        .map(|&nn| {
            (nn as f64)
                .powf(config.sample_exponent)
                .ceil()
                .max(min_s as f64) as usize
        })
        .collect();

    let test_kmers: Vec<Vec<i32>> = raw_kmers
        .into_iter()
        .map(|v| {
            let mut sorted: Vec<i32> = v
                .into_iter()
                .filter(|&x| x != NA_INTEGER)
                .map(|x| x + 1)
                .collect();
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        })
        .collect();

    let b_values: Vec<usize> = test_kmers
        .iter()
        .zip(s_values.iter())
        .map(|(km, &s)| {
            effective_bootstrap_replicates(km.len(), s, bootstraps, config.bootstrap_coverage_cap)
        })
        .collect();

    let boths: Vec<usize> = strands
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == 1)
        .map(|(i, _)| i)
        .collect();

    let boths_map: HashMap<usize, usize> = boths
        .iter()
        .enumerate()
        .map(|(rev_idx, &orig_idx)| (orig_idx, rev_idx))
        .collect();

    let rev_kmers = if !boths.is_empty() {
        let rev_seqs: Vec<String> = boths
            .iter()
            .map(|&i| reverse_complement(&seqs[i]))
            .collect();
        let raw = enumerate_sequences(&rev_seqs, k, false, false, &[], true, spaced_seed.as_ref());
        raw.into_iter()
            .map(|v| {
                let mut sorted: Vec<i32> = v
                    .into_iter()
                    .filter(|&x| x != NA_INTEGER)
                    .map(|x| x + 1)
                    .collect();
                sorted.sort_unstable();
                sorted.dedup();
                sorted
            })
            .collect()
    } else {
        Vec::new()
    };

    PrecomputedData {
        test_kmers,
        rev_kmers,
        s_values,
        b_values,
        boths,
        boths_map,
        ls,
    }
}

fn effective_bootstrap_replicates(
    unique_kmer_count: usize,
    sample_size: usize,
    bootstraps: usize,
    bootstrap_coverage_cap: bool,
) -> usize {
    if !bootstrap_coverage_cap || sample_size == 0 {
        bootstraps
    } else {
        (5.0 * unique_kmer_count as f64 / sample_size as f64)
            .min(bootstraps as f64)
            .max(1.0) as usize
    }
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
    ls: &[usize],
    rng: &mut RRng,
) -> Option<(ClassificationResult, f64)> {
    if config.beam_width > 1 {
        return classify_one_pass_beam(my_kmers, s, b, ts, config, ls, rng);
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
    // Phase 1b telemetry: count near-tied descent splits and track the max
    // runner-up vote fraction. Always-on (no flag) — overhead is two scalars
    // per split. `CLOSE_RUNNER_UP_RATIO` defines what counts as "close": a
    // runner-up clearing 90% of the winner's votes. Used by Phase 1c E.0
    // frequency probe to gauge whether beam widening (Phase 2) has substrate.
    const CLOSE_RUNNER_UP_RATIO: f64 = 0.90;
    let mut beam_close_runner_up_count: u32 = 0;
    let mut beam_max_runner_up_fraction: f64 = 0.0;
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
            let profiles = if ts.use_idf_in_descent {
                &dk.weighted_profiles
            } else {
                &dk.profiles
            };
            let mut hits_flat = vec![0.0f64; n_sub * b];
            for j in 0..n_sub {
                let row = vector_sum(&matches, &profiles[j], &sampling, b);
                hits_flat[j * b..(j + 1) * b].copy_from_slice(&row);
            }
            let mut vote_counts = vec![0usize; n_sub];
            for rep in 0..b {
                let max_val = (0..n_sub)
                    .map(|j| hits_flat[j * b + rep])
                    .fold(0.0f64, f64::max);
                if max_val > 0.0 {
                    for j in 0..n_sub {
                        if hits_flat[j * b + rep] == max_val {
                            vote_counts[j] += 1;
                        }
                    }
                }
            }

            // Phase 1b telemetry: track runner-up closeness. `vote_counts`
            // is votes per child; we need the top-2 to compute the ratio.
            // When n_sub < 2 there's no runner-up, so the metric stays at 0.
            if n_sub >= 2 {
                let mut sorted = vote_counts.clone();
                sorted.sort_unstable_by(|a, b_| b_.cmp(a));
                let top = sorted[0];
                let runner_up = sorted[1];
                if top > 0 {
                    let frac = runner_up as f64 / top as f64;
                    if frac > beam_max_runner_up_fraction {
                        beam_max_runner_up_fraction = frac;
                    }
                    if frac >= CLOSE_RUNNER_UP_RATIO {
                        beam_close_runner_up_count += 1;
                    }
                }
            }

            let w: Vec<usize> = vote_counts
                .iter()
                .enumerate()
                .filter(|(_, &c)| c >= (config.min_descend * b as f64) as usize)
                .map(|(i, _)| i)
                .collect();
            if w.len() != 1 {
                // Mech 1 — halt-in-the-middle fallback. If at least one child
                // cleared the 0.5*b vote-fraction threshold, hand all children
                // with > 0 votes to leaf-phase. Otherwise (scattered vote
                // pattern) fall through to all children.
                let min_votes = ((b as f64) * 0.5) as usize;
                let any_strong = vote_counts.iter().any(|&c| c >= min_votes);
                if !any_strong {
                    w_indices = (0..vote_counts.len()).collect();
                } else {
                    w_indices = vote_counts
                        .iter()
                        .enumerate()
                        .filter(|(_, &c)| c > 0)
                        .map(|(i, _)| i)
                        .collect();
                    if w_indices.is_empty() {
                        w_indices = (0..vote_counts.len()).collect();
                    }
                }
                break;
            }
            let winner = w[0];
            if children[subtrees[winner]].is_empty() {
                // Mech 2 — terminal: only the winner enters leaf-phase.
                w_indices = vec![winner];
                break;
            }
            k_node = subtrees[winner];
        } else {
            if children[subtrees[0]].is_empty() {
                w_indices = vec![0];
                break;
            }
            k_node = subtrees[0];
        }
    }

    let scored = leaf_phase_score(k_node, &w_indices, my_kmers, s, b, ts, config, ls, rng);
    // Attach descent telemetry to the result. Greedy is single-candidate by
    // construction; `beam_widened_runner_ups` stays 0 (Phase 2 territory).
    // Skip the overwrite on abstention paths — `unclassified()` and Path-C
    // produce fully-zeroed diagnostic fields, and the abstention contract
    // requires they stay 0.
    scored.map(|(mut result, sim)| {
        if result.reject_reason.is_none() {
            result.beam_close_runner_up_count = beam_close_runner_up_count;
            result.beam_max_runner_up_fraction = beam_max_runner_up_fraction;
            result.beam_candidate_count = 1;
            result.beam_widened_runner_ups = 0;
        }
        (result, sim)
    })
}

/// True when a runner-up's vote count clears the relaxed-equality cutoff
/// relative to the winner. With `descent_tie_margin = 0.0`, only exact ties
/// pass this helper; the full beam admission rule also admits runner-ups that
/// independently clear `min_descend`.
///
/// Used at beam descent to add below-threshold near-runner-ups beside the
/// normal top-`beam_width` children that clear `min_descend`.
#[inline]
pub fn passes_descent_runner_up_gate(
    votes: usize,
    winner_votes: usize,
    descent_tie_margin: f64,
) -> bool {
    (votes as f64) >= (winner_votes as f64) * (1.0 - descent_tie_margin)
}

/// Aggregate per-replicate scores across the top-M refs of a group.
///
/// `refs` are indices into `hits_flat` rows (each row has `b` per-replicate
/// values). Caller is responsible for ensuring `refs` is non-empty. At
/// `refs.len() == 1` every mode reduces to identity on the single value, so
/// the leaf-phase M=1 path is bit-identical regardless of which mode is
/// selected.
///
/// `TrimmedMean` falls back to `Mean` when `refs.len() < 3` — too few refs to
/// drop both ends.
#[inline]
pub fn aggregate_top_m_at_rep(
    refs: &[usize],
    rep: usize,
    b: usize,
    hits_flat: &[f64],
    mode: LeafAggregationMode,
) -> f64 {
    debug_assert!(!refs.is_empty(), "aggregate_top_m_at_rep on empty refs");
    let n = refs.len();
    match mode {
        LeafAggregationMode::Mean => {
            let mut sum = 0.0;
            for &r in refs {
                sum += hits_flat[r * b + rep];
            }
            sum / n as f64
        }
        LeafAggregationMode::PerRepBest => {
            let mut best = f64::NEG_INFINITY;
            for &r in refs {
                let v = hits_flat[r * b + rep];
                if v > best {
                    best = v;
                }
            }
            best
        }
        LeafAggregationMode::TrimmedMean => {
            if n < 3 {
                let mut sum = 0.0;
                for &r in refs {
                    sum += hits_flat[r * b + rep];
                }
                sum / n as f64
            } else {
                let mut vals: Vec<f64> = refs.iter().map(|&r| hits_flat[r * b + rep]).collect();
                vals.sort_by(|a, b_| a.partial_cmp(b_).unwrap_or(std::cmp::Ordering::Equal));
                let trimmed = &vals[1..n - 1];
                trimmed.iter().sum::<f64>() / trimmed.len() as f64
            }
        }
    }
}

/// Beam search variant: maintain multiple candidate paths during tree descent.
#[allow(clippy::too_many_arguments)]
fn classify_one_pass_beam(
    my_kmers: &[i32],
    s: usize,
    b: usize,
    ts: &TrainingSet,
    config: &ClassifyConfig,
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
        /// Phase 1b telemetry — count of descent steps along this candidate's
        /// path where the runner-up cleared `CLOSE_RUNNER_UP_RATIO` × winner.
        beam_close_runner_up_count: u32,
        /// Max `runner_up_votes / winner_votes` ratio seen along this path.
        beam_max_runner_up_fraction: f64,
        /// Number of runner-up admissions on this candidate's path that were
        /// admitted by `descent_tie_margin` below the normal
        /// integer-truncated `min_descend` floor. Stays 0 at default
        /// `descent_tie_margin = 0.0`.
        beam_widened_runner_ups: u32,
        /// True when this candidate has reached a leaf-parent (its `node`
        /// has no further children to descend into) or otherwise halted in
        /// the middle. Terminal candidates pass through subsequent descent
        /// iterations unchanged — without this flag they would re-enter the
        /// vote loop at the same node, double-counting telemetry and
        /// possibly clobbering `w_indices`.
        terminal: bool,
    }

    const CLOSE_RUNNER_UP_RATIO: f64 = 0.90;
    let beam_width = config.beam_width;
    let mut active = vec![BeamCandidate {
        node: 0,
        w_indices: Vec::new(),
        score: 1.0,
        beam_close_runner_up_count: 0,
        beam_max_runner_up_fraction: 0.0,
        beam_widened_runner_ups: 0,
        terminal: false,
    }];

    loop {
        let mut next: Vec<BeamCandidate> = Vec::new();
        let mut any_expanded = false;

        for candidate in &active {
            // Terminal candidates have already halted at a leaf-parent or
            // mid-tree dead-end; pass them through unchanged so we don't
            // re-run vote computation and double-count widening telemetry.
            if candidate.terminal {
                next.push(BeamCandidate {
                    node: candidate.node,
                    w_indices: candidate.w_indices.clone(),
                    score: candidate.score,
                    beam_close_runner_up_count: candidate.beam_close_runner_up_count,
                    beam_max_runner_up_fraction: candidate.beam_max_runner_up_fraction,
                    beam_widened_runner_ups: candidate.beam_widened_runner_ups,
                    terminal: true,
                });
                continue;
            }

            let k_node = candidate.node;
            let subtrees = &children[k_node];
            let dk = &decision_kmers[k_node];

            if dk.is_none() || fraction[k_node].is_none() {
                next.push(BeamCandidate {
                    node: k_node,
                    w_indices: (0..subtrees.len()).collect(),
                    score: candidate.score,
                    beam_close_runner_up_count: candidate.beam_close_runner_up_count,
                    beam_max_runner_up_fraction: candidate.beam_max_runner_up_fraction,
                    beam_widened_runner_ups: candidate.beam_widened_runner_ups,
                    terminal: true,
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
                        beam_close_runner_up_count: candidate.beam_close_runner_up_count,
                        beam_max_runner_up_fraction: candidate.beam_max_runner_up_fraction,
                        beam_widened_runner_ups: candidate.beam_widened_runner_ups,
                        terminal: false,
                    });
                    any_expanded = true;
                } else {
                    next.push(BeamCandidate {
                        node: k_node,
                        w_indices: if subtrees.is_empty() {
                            vec![]
                        } else {
                            (0..subtrees.len()).collect()
                        },
                        score: candidate.score,
                        beam_close_runner_up_count: candidate.beam_close_runner_up_count,
                        beam_max_runner_up_fraction: candidate.beam_max_runner_up_fraction,
                        beam_widened_runner_ups: candidate.beam_widened_runner_ups,
                        terminal: true,
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
            let profiles = if ts.use_idf_in_descent {
                &dk.weighted_profiles
            } else {
                &dk.profiles
            };
            let mut hits_flat = vec![0.0f64; n_sub * b];
            for j in 0..n_sub {
                let row = vector_sum(&matches, &profiles[j], &sampling, b);
                hits_flat[j * b..(j + 1) * b].copy_from_slice(&row);
            }
            let mut vote_counts = vec![0usize; n_sub];
            for rep in 0..b {
                let max_val = (0..n_sub)
                    .map(|j| hits_flat[j * b + rep])
                    .fold(0.0f64, f64::max);
                if max_val > 0.0 {
                    for j in 0..n_sub {
                        if hits_flat[j * b + rep] == max_val {
                            vote_counts[j] += 1;
                        }
                    }
                }
            }

            // Collect children with votes, sorted by vote count descending
            let mut children_by_votes: Vec<(usize, usize)> = vote_counts
                .iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(j, &c)| (j, c))
                .collect();
            children_by_votes.sort_by(|a, b_| b_.1.cmp(&a.1));

            // Phase 1b telemetry: same close-runner-up logic as greedy. Use
            // the full `vote_counts` (including 0-vote children) so the ratio
            // matches the greedy form exactly when both fire.
            let (next_close_count, next_max_fraction) = if n_sub >= 2 {
                let mut sorted = vote_counts.clone();
                sorted.sort_unstable_by(|a, b_| b_.cmp(a));
                let top = sorted[0];
                let runner_up = sorted[1];
                if top > 0 {
                    let frac = runner_up as f64 / top as f64;
                    let new_max = candidate.beam_max_runner_up_fraction.max(frac);
                    let new_close = candidate.beam_close_runner_up_count
                        + if frac >= CLOSE_RUNNER_UP_RATIO { 1 } else { 0 };
                    (new_close, new_max)
                } else {
                    (
                        candidate.beam_close_runner_up_count,
                        candidate.beam_max_runner_up_fraction,
                    )
                }
            } else {
                (
                    candidate.beam_close_runner_up_count,
                    candidate.beam_max_runner_up_fraction,
                )
            };

            if children_by_votes.is_empty() {
                // No votes at all — terminal
                next.push(BeamCandidate {
                    node: k_node,
                    w_indices: (0..n_sub).collect(),
                    score: candidate.score,
                    beam_close_runner_up_count: next_close_count,
                    beam_max_runner_up_fraction: next_max_fraction,
                    beam_widened_runner_ups: candidate.beam_widened_runner_ups,
                    terminal: true,
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
                        beam_close_runner_up_count: next_close_count,
                        beam_max_runner_up_fraction: next_max_fraction,
                        beam_widened_runner_ups: candidate.beam_widened_runner_ups,
                        terminal: true,
                    });
                } else {
                    next.push(BeamCandidate {
                        node: winner_child,
                        w_indices: Vec::new(),
                        score: candidate.score * top_vote_frac,
                        beam_close_runner_up_count: next_close_count,
                        beam_max_runner_up_fraction: next_max_fraction,
                        beam_widened_runner_ups: candidate.beam_widened_runner_ups,
                        terminal: false,
                    });
                    any_expanded = true;
                }

                // Runner-ups enter the beam if they independently clear the
                // same `min_descend` floor as the winner, or if the relaxed
                // `descent_tie_margin` cutoff admits them as near-runner-ups.
                // `beam_width` is the capacity; this gate decides which
                // siblings are eligible for that capacity.
                //
                // `children_by_votes` is vote-sorted descending, so once one
                // runner-up fails both gates, all subsequent runner-ups fail
                // too — break is safe.
                //
                // `beam_widened_runner_ups` increments only for admissions
                // supplied by `descent_tie_margin` below the normal
                // integer-truncated `min_descend` floor.
                let winner_votes = children_by_votes[0].1;
                let min_descend_floor = (config.min_descend * b as f64) as usize;
                for &(j, votes) in children_by_votes.iter().skip(1) {
                    let clears_min_descend = votes >= min_descend_floor;
                    let clears_margin = passes_descent_runner_up_gate(
                        votes,
                        winner_votes,
                        config.descent_tie_margin,
                    );
                    if !clears_min_descend && !clears_margin {
                        break;
                    }
                    let vf = votes as f64 / b as f64;
                    let widened_inc = if !clears_min_descend && clears_margin {
                        1
                    } else {
                        0
                    };
                    let new_widened = candidate.beam_widened_runner_ups + widened_inc;
                    let child = subtrees[j];
                    if children[child].is_empty() {
                        next.push(BeamCandidate {
                            node: k_node,
                            w_indices: vec![j],
                            score: candidate.score * vf,
                            beam_close_runner_up_count: next_close_count,
                            beam_max_runner_up_fraction: next_max_fraction,
                            beam_widened_runner_ups: new_widened,
                            terminal: true,
                        });
                    } else {
                        next.push(BeamCandidate {
                            node: child,
                            w_indices: Vec::new(),
                            score: candidate.score * vf,
                            beam_close_runner_up_count: next_close_count,
                            beam_max_runner_up_fraction: next_max_fraction,
                            beam_widened_runner_ups: new_widened,
                            terminal: false,
                        });
                        any_expanded = true;
                    }
                }
            } else {
                // No confident winner — terminal (same fallback as greedy
                // Mech 1). If at least one child cleared the 0.5*b
                // vote-fraction threshold, hand all children with > 0 votes
                // to leaf-phase. Otherwise fall through to all children.
                let min_votes = ((b as f64) * 0.5) as usize;
                let any_strong = vote_counts.iter().any(|&c| c >= min_votes);
                let w_indices: Vec<usize> = if !any_strong {
                    (0..vote_counts.len()).collect()
                } else {
                    let w_pos: Vec<usize> = vote_counts
                        .iter()
                        .enumerate()
                        .filter(|(_, &c)| c > 0)
                        .map(|(i, _)| i)
                        .collect();
                    if w_pos.is_empty() {
                        (0..vote_counts.len()).collect()
                    } else {
                        w_pos
                    }
                };
                next.push(BeamCandidate {
                    node: k_node,
                    w_indices,
                    score: candidate.score * top_vote_frac,
                    beam_close_runner_up_count: next_close_count,
                    beam_max_runner_up_fraction: next_max_fraction,
                    beam_widened_runner_ups: candidate.beam_widened_runner_ups,
                    terminal: true,
                });
            }
        }

        // Prune to beam_width
        next.sort_by(|a, b_| {
            b_.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        next.truncate(beam_width);
        active = next;

        if !any_expanded {
            break;
        }
    }

    // Score each candidate via leaf phase, pick best. Telemetry comes from
    // the winning candidate's path; `beam_candidate_count` is the number of
    // candidates that survived to leaf-phase scoring (= `active.len()`).
    let beam_candidate_count = active.len() as u32;
    let mut best: Option<(ClassificationResult, f64, u32, f64, u32)> = None;
    for candidate in &active {
        if let Some((result, sim)) = leaf_phase_score(
            candidate.node,
            &candidate.w_indices,
            my_kmers,
            s,
            b,
            ts,
            config,
            ls,
            rng,
        ) {
            let entry = (
                result,
                sim,
                candidate.beam_close_runner_up_count,
                candidate.beam_max_runner_up_fraction,
                candidate.beam_widened_runner_ups,
            );
            match &best {
                None => best = Some(entry),
                Some((_, best_sim, _, _, _)) if sim > *best_sim => best = Some(entry),
                _ => {}
            }
        }
    }
    best.map(|(mut result, sim, close, max_frac, widened)| {
        // Skip the overwrite on abstention paths so descent telemetry doesn't
        // leak through `unclassified()` and Path-C results that contracted
        // for all-zero diagnostic fields.
        if result.reject_reason.is_none() {
            result.beam_close_runner_up_count = close;
            result.beam_max_runner_up_fraction = max_frac;
            result.beam_candidate_count = beam_candidate_count;
            result.beam_widened_runner_ups = widened;
        }
        (result, sim)
    })
}

const ZERO_CONF_EPSILON: f64 = 1e-12;
const DEEPEST_MODE_THRESHOLD: &str = "threshold";
const DEEPEST_MODE_MARGIN_RESCUE: &str = "margin_rescue";
const DEEPEST_MODE_REJECTED: &str = "rejected";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RescueRejection {
    RescueDisabled,
    ShallowerRankFailed,
    LcaCap,
    ChallengeNoCandidatesAfterSuppression,
    ChallengeScoringEmpty,
    CandidateCountLt2,
    ChallengeCandidateCapExceeded,
    NonfiniteChallengeScore,
    NegativeChallengeRunner,
    NonpositiveChallengeTop,
    SelectedNotChallengeTop,
    SelectedMissingFromChallenge,
    ChallengeExactTie,
    ChallengeNearTie,
    FloorFailed,
    DeltaFailed,
    RatioFailed,
}

impl RescueRejection {
    fn as_str(self) -> &'static str {
        match self {
            RescueRejection::RescueDisabled => "rescue_disabled",
            RescueRejection::ShallowerRankFailed => "shallower_rank_failed",
            RescueRejection::LcaCap => "lca_cap",
            RescueRejection::ChallengeNoCandidatesAfterSuppression => {
                "challenge_no_candidates_after_suppression"
            }
            RescueRejection::ChallengeScoringEmpty => "challenge_scoring_empty",
            RescueRejection::CandidateCountLt2 => "candidate_count_lt_2",
            RescueRejection::ChallengeCandidateCapExceeded => "challenge_candidate_cap_exceeded",
            RescueRejection::NonfiniteChallengeScore => "nonfinite_challenge_score",
            RescueRejection::NegativeChallengeRunner => "negative_challenge_runner",
            RescueRejection::NonpositiveChallengeTop => "nonpositive_challenge_top",
            RescueRejection::SelectedNotChallengeTop => "selected_not_challenge_top",
            RescueRejection::SelectedMissingFromChallenge => "selected_missing_from_challenge",
            RescueRejection::ChallengeExactTie => "challenge_exact_tie",
            RescueRejection::ChallengeNearTie => "challenge_near_tie",
            RescueRejection::FloorFailed => "floor_failed",
            RescueRejection::DeltaFailed => "delta_failed",
            RescueRejection::RatioFailed => "ratio_failed",
        }
    }
}

fn deepest_rank_stop_reason_for_rejection(reason: RescueRejection) -> &'static str {
    match reason {
        RescueRejection::ChallengeNoCandidatesAfterSuppression => {
            "challenge_no_candidates_after_suppression"
        }
        RescueRejection::ChallengeScoringEmpty => "challenge_scoring_empty",
        RescueRejection::CandidateCountLt2 => "challenge_candidate_count_lt2",
        RescueRejection::ChallengeCandidateCapExceeded => "challenge_candidate_cap_exceeded",
        RescueRejection::SelectedMissingFromChallenge => "selected_missing_from_challenge",
        RescueRejection::ChallengeExactTie => "challenge_exact_tie",
        RescueRejection::ChallengeNearTie => "challenge_near_tie",
        RescueRejection::FloorFailed => "margin_floor_failed",
        RescueRejection::DeltaFailed => "margin_delta_failed",
        RescueRejection::RatioFailed => "margin_ratio_failed",
        _ => reason.as_str(),
    }
}

pub struct DeepestRankRescueDecisionInput {
    pub challenge_original_selection: f64,
    pub challenge_runner: f64,
    pub selected_is_unique_challenge_top: bool,
    pub challenge_exact_tie: bool,
    pub challenge_near_tie: bool,
    pub candidate_cap_exceeded: bool,
    pub floor: Option<f64>,
    pub min_delta: f64,
    pub min_ratio: f64,
    pub candidate_count: u32,
}

pub fn passes_deepest_rank_margin_rescue(
    input: &DeepestRankRescueDecisionInput,
) -> Result<(), RescueRejection> {
    let floor = input.floor.ok_or(RescueRejection::RescueDisabled)?;
    if input.candidate_count < 2 {
        return Err(RescueRejection::CandidateCountLt2);
    }
    if input.candidate_cap_exceeded {
        return Err(RescueRejection::ChallengeCandidateCapExceeded);
    }
    let top = input.challenge_original_selection;
    let runner = input.challenge_runner;
    if !top.is_finite() || !runner.is_finite() {
        return Err(RescueRejection::NonfiniteChallengeScore);
    }
    if runner < -ZERO_CONF_EPSILON {
        return Err(RescueRejection::NegativeChallengeRunner);
    }
    if top <= ZERO_CONF_EPSILON {
        return Err(RescueRejection::NonpositiveChallengeTop);
    }
    if !input.selected_is_unique_challenge_top {
        return Err(RescueRejection::SelectedNotChallengeTop);
    }
    if input.challenge_exact_tie {
        return Err(RescueRejection::ChallengeExactTie);
    }
    if input.challenge_near_tie {
        return Err(RescueRejection::ChallengeNearTie);
    }
    if top < floor {
        return Err(RescueRejection::FloorFailed);
    }
    let runner_for_math = if runner.abs() <= ZERO_CONF_EPSILON {
        0.0
    } else {
        runner
    };
    let delta = top - runner_for_math;
    if delta < input.min_delta {
        return Err(RescueRejection::DeltaFailed);
    }
    if runner_for_math > 0.0 {
        let ratio = top / runner_for_math;
        if ratio < input.min_ratio {
            return Err(RescueRejection::RatioFailed);
        }
    }
    Ok(())
}

struct LeafBootstrapContext<'a> {
    u_sampling: &'a [i32],
    positions: &'a [usize],
    ranges: &'a [usize],
    u_weights: &'a [f64],
    b: usize,
    davg: f64,
    avg_unique_for_stop_node: f64,
}

struct LeafCandidateScores {
    lookup: Vec<usize>,
    unique_groups: Vec<usize>,
    top_hits_idx: Vec<usize>,
    top_m_refs_per_group: Vec<Vec<usize>>,
    selected_hits_flat: Vec<f64>,
    sum_hits: Vec<f64>,
    tot_hits: Vec<f64>,
    winners: Vec<usize>,
    leaf_widened_winners: u32,
    top_confidence: Option<f64>,
    runner_up_confidence: Option<f64>,
    margin_delta: Option<f64>,
    margin_ratio: Option<f64>,
    candidate_count: u32,
}

#[allow(clippy::too_many_arguments)]
fn score_leaf_candidate_set(
    keep_seq_indices: &[usize],
    context: &LeafBootstrapContext<'_>,
    ts: &TrainingSet,
    config: &ClassifyConfig,
    ls: &[usize],
) -> Option<LeafCandidateScores> {
    let train_kmers = &ts.kmers;
    let cross_index = &ts.cross_index;
    let mut sum_hits = if let Some(ref inv_idx) = ts.inverted_index {
        match_sums_inverted(
            context.u_sampling,
            inv_idx,
            keep_seq_indices,
            context.u_weights,
            context.positions,
            context.ranges,
        )
    } else {
        match_sums(
            context.u_sampling,
            train_kmers,
            keep_seq_indices,
            context.u_weights,
            context.positions,
            context.ranges,
        )
    };
    let b = context.b;
    if sum_hits.is_empty() || b == 0 {
        return None;
    }

    if config.length_normalize {
        let avg_unique = context.avg_unique_for_stop_node;
        for (k_idx, &seq_idx) in keep_seq_indices.iter().enumerate() {
            let n_unique = ls[seq_idx] as f64;
            if n_unique > 0.0 && avg_unique > 0.0 {
                let norm_factor = (n_unique / avg_unique).sqrt();
                sum_hits[k_idx] /= norm_factor;
            }
        }
    }

    let lookup: Vec<usize> = keep_seq_indices
        .iter()
        .map(|&idx| cross_index[idx])
        .collect();
    let mut order: Vec<usize> = (0..lookup.len()).collect();
    order.sort_unstable_by_key(|&i| lookup[i]);

    let mut unique_groups: Vec<usize> = Vec::new();
    let mut top_hits_idx: Vec<usize> = Vec::new();
    let mut top_m_refs_per_group: Vec<Vec<usize>> = Vec::new();
    let mut selected_ref_positions: Vec<usize> = Vec::new();
    let leaf_top_m = config.leaf_top_m.max(1);
    let mut i = 0;
    while i < order.len() {
        let group = lookup[order[i]];
        unique_groups.push(group);
        let mut best_idx = order[i];
        let mut best_val = sum_hits[order[i]];
        let mut group_refs: Vec<usize> = if leaf_top_m > 1 {
            vec![order[i]]
        } else {
            Vec::new()
        };
        i += 1;
        while i < order.len() && lookup[order[i]] == group {
            if sum_hits[order[i]] > best_val {
                best_val = sum_hits[order[i]];
                best_idx = order[i];
            }
            if leaf_top_m > 1 {
                group_refs.push(order[i]);
            }
            i += 1;
        }
        if leaf_top_m > 1 {
            group_refs.sort_by(|&a, &b_| {
                sum_hits[b_]
                    .partial_cmp(&sum_hits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            group_refs.truncate(leaf_top_m.min(group_refs.len()));
            let mut selected_group_refs = Vec::with_capacity(group_refs.len());
            for keep_pos in group_refs {
                selected_group_refs.push(selected_ref_positions.len());
                selected_ref_positions.push(keep_pos);
            }
            top_hits_idx.push(selected_group_refs[0]);
            top_m_refs_per_group.push(selected_group_refs);
        } else {
            top_hits_idx.push(selected_ref_positions.len());
            selected_ref_positions.push(best_idx);
        }
    }

    let n_top = top_hits_idx.len();
    if n_top == 0 {
        return None;
    }

    let mut selected_hits_flat = if let Some(ref inv_idx) = ts.inverted_index {
        match_selected_rows_inverted(
            context.u_sampling,
            inv_idx,
            keep_seq_indices,
            &selected_ref_positions,
            context.u_weights,
            b,
            context.positions,
            context.ranges,
        )
    } else {
        match_selected_rows(
            context.u_sampling,
            train_kmers,
            keep_seq_indices,
            &selected_ref_positions,
            context.u_weights,
            b,
            context.positions,
            context.ranges,
        )
    };

    if selected_hits_flat.is_empty() {
        return None;
    }

    if config.length_normalize {
        let avg_unique = context.avg_unique_for_stop_node;
        for (selected_row, &keep_pos) in selected_ref_positions.iter().enumerate() {
            let seq_idx = keep_seq_indices[keep_pos];
            let n_unique = ls[seq_idx] as f64;
            if n_unique > 0.0 && avg_unique > 0.0 {
                let norm_factor = (n_unique / avg_unique).sqrt();
                let base = selected_row * b;
                for rep in 0..b {
                    selected_hits_flat[base + rep] /= norm_factor;
                }
            }
        }
    }

    let descendants_of: Vec<Vec<u32>> = if config.suppress_ancestor_only_groups {
        let group_paths: Vec<&str> = unique_groups
            .iter()
            .map(|&g| ts.taxonomy[g].as_str())
            .collect();
        (0..n_top)
            .map(|j| {
                let j_path = group_paths[j];
                (0..n_top)
                    .filter(|&k| k != j && group_paths[k].starts_with(j_path))
                    .map(|k| k as u32)
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };

    let m_active = leaf_top_m > 1;
    let leaf_mode = config.leaf_aggregation_mode;
    let mut tot_hits = vec![0.0f64; n_top];
    let mut is_tied = vec![false; n_top];
    let mut drop_mask = vec![false; n_top];
    let mut group_vals_at_rep = vec![0.0_f64; n_top];

    for rep in 0..b {
        for j in 0..n_top {
            group_vals_at_rep[j] = if m_active {
                aggregate_top_m_at_rep(
                    &top_m_refs_per_group[j],
                    rep,
                    b,
                    &selected_hits_flat,
                    leaf_mode,
                )
            } else {
                selected_hits_flat[top_hits_idx[j] * b + rep]
            };
        }
        let mut max_val = f64::NEG_INFINITY;
        for &v in &group_vals_at_rep {
            if v > max_val {
                max_val = v;
            }
        }
        if context.davg == 0.0 {
            continue;
        }
        let tie_cutoff = if config.leaf_tie_margin > 0.0 {
            max_val - max_val.abs() * config.leaf_tie_margin
        } else {
            max_val
        };
        is_tied.iter_mut().for_each(|x| *x = false);
        let mut n_tied: usize = 0;
        for j in 0..n_top {
            if group_vals_at_rep[j] >= tie_cutoff {
                is_tied[j] = true;
                n_tied += 1;
            }
        }
        if config.suppress_ancestor_only_groups && n_tied > 1 {
            drop_mask.iter_mut().for_each(|x| *x = false);
            let mut to_drop_count = 0usize;
            for j in 0..n_top {
                if !is_tied[j] {
                    continue;
                }
                for &k in &descendants_of[j] {
                    if is_tied[k as usize] {
                        drop_mask[j] = true;
                        to_drop_count += 1;
                        break;
                    }
                }
            }
            if to_drop_count > 0 && to_drop_count < n_tied {
                for j in 0..n_top {
                    if drop_mask[j] {
                        is_tied[j] = false;
                        n_tied -= 1;
                    }
                }
            }
        }
        if n_tied == 0 {
            continue;
        }
        let share = 1.0 / n_tied as f64;
        for j in 0..n_top {
            if is_tied[j] {
                tot_hits[j] += group_vals_at_rep[j] / context.davg * share;
            }
        }
    }

    let max_tot = tot_hits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut leaf_widened_winners: u32 = 0;
    let mut winners: Vec<usize> = if config.leaf_tie_margin > 0.0 {
        let cutoff = max_tot - max_tot.abs() * config.leaf_tie_margin;
        let mut out = Vec::new();
        for (idx, &v) in tot_hits.iter().enumerate() {
            if v >= cutoff {
                out.push(idx);
                if v != max_tot {
                    leaf_widened_winners += 1;
                }
            }
        }
        out
    } else {
        tot_hits
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == max_tot)
            .map(|(idx, _)| idx)
            .collect()
    };

    if config.suppress_ancestor_only_groups && winners.len() > 1 {
        let winner_paths: Vec<&str> = winners
            .iter()
            .map(|&w| ts.taxonomy[unique_groups[w]].as_str())
            .collect();
        let mut keep_winner = vec![true; winners.len()];
        for i in 0..winners.len() {
            let i_path = winner_paths[i];
            for (j, j_path) in winner_paths.iter().enumerate() {
                if i != j && j_path.starts_with(i_path) {
                    keep_winner[i] = false;
                    break;
                }
            }
        }
        let filtered: Vec<usize> = winners
            .iter()
            .zip(keep_winner.iter())
            .filter_map(|(&w, &k)| k.then_some(w))
            .collect();
        if !filtered.is_empty() {
            winners = filtered;
        }
    }

    let (top_confidence, runner_up_confidence, margin_delta, margin_ratio) =
        confidence_margin_from_tot_hits(&tot_hits, b);

    Some(LeafCandidateScores {
        lookup,
        unique_groups,
        top_hits_idx,
        top_m_refs_per_group,
        selected_hits_flat,
        sum_hits,
        tot_hits,
        winners,
        leaf_widened_winners,
        top_confidence,
        runner_up_confidence,
        margin_delta,
        margin_ratio,
        candidate_count: n_top as u32,
    })
}

fn confidence_margin_from_tot_hits(
    tot_hits: &[f64],
    b: usize,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    if tot_hits.is_empty() || b == 0 {
        return (None, None, None, None);
    }
    let mut vals: Vec<f64> = tot_hits.iter().map(|v| *v / b as f64 * 100.0).collect();
    vals.sort_by(|a, b_| b_.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top = vals[0];
    let runner = vals.get(1).copied().unwrap_or(0.0);
    if !top.is_finite() || !runner.is_finite() {
        return (Some(top), Some(runner), None, None);
    }
    let normalized_runner = if runner.abs() <= ZERO_CONF_EPSILON {
        0.0
    } else {
        runner
    };
    let delta = top - normalized_runner;
    let ratio = if normalized_runner > 0.0 {
        Some(top / normalized_runner)
    } else {
        None
    };
    (Some(top), Some(normalized_runner), Some(delta), ratio)
}

#[derive(Debug, Clone, Copy)]
enum ThresholdStopReason {
    FullPass,
    FailedThreshold {
        rank_index: usize,
        threshold: f64,
        confidence: f64,
    },
    LcaCapped {
        cap_index: usize,
    },
}

fn threshold_for_rank(config: &ClassifyConfig, rank_index: usize) -> f64 {
    match &config.rank_thresholds {
        Some(rt) if rank_index < rt.len() => rt[rank_index],
        Some(rt) if !rt.is_empty() => *rt.last().unwrap(),
        _ => config.threshold,
    }
}

fn is_deepest_rank_margin_eligible(
    predicted_group: usize,
    ts: &TrainingSet,
    max_level: i32,
) -> bool {
    ts.children
        .get(predicted_group)
        .map(|c| c.is_empty())
        .unwrap_or(false)
        && ts
            .sequences
            .get(predicted_group)
            .and_then(|s| s.as_ref())
            .map(|s| !s.is_empty())
            .unwrap_or(false)
        && ts.levels.get(predicted_group).copied() == Some(max_level)
}

fn terminal_sibling_endpoint_groups(
    predicted_group: usize,
    ts: &TrainingSet,
    max_level: i32,
    suppress_ancestor_only_groups: bool,
) -> Vec<usize> {
    let parent = ts
        .parents
        .get(predicted_group)
        .copied()
        .unwrap_or(predicted_group);
    let mut groups: Vec<usize> = ts
        .children
        .get(parent)
        .into_iter()
        .flat_map(|children| children.iter().copied())
        .filter(|&g| is_deepest_rank_margin_eligible(g, ts, max_level))
        .collect();
    groups.sort_unstable();
    groups.dedup();
    if !groups.contains(&predicted_group)
        && is_deepest_rank_margin_eligible(predicted_group, ts, max_level)
    {
        groups.push(predicted_group);
        groups.sort_unstable();
    }
    if suppress_ancestor_only_groups && groups.len() > 1 {
        groups = groups
            .iter()
            .copied()
            .filter(|&g| {
                let gp = ts.taxonomy[g].as_str();
                !groups
                    .iter()
                    .any(|&other| other != g && ts.taxonomy[other].starts_with(gp))
            })
            .collect();
    }
    groups
}

fn finite_opt(v: Option<f64>) -> Option<f64> {
    v.filter(|x| x.is_finite())
}

fn build_deepest_rank_top_k(
    scores: &LeafCandidateScores,
    ts: &TrainingSet,
    b: usize,
    k: usize,
) -> Option<DeepestRankTopK> {
    if k == 0 || b == 0 || scores.tot_hits.is_empty() {
        return None;
    }
    let mut rows: Vec<(String, Option<f64>, f64)> = scores
        .unique_groups
        .iter()
        .zip(scores.tot_hits.iter())
        .map(|(&group, &hits)| {
            let confidence = hits / b as f64 * 100.0;
            (
                ts.taxonomy[group].clone(),
                finite_opt(Some(confidence)),
                confidence,
            )
        })
        .collect();
    rows.sort_by(|a, b_| {
        b_.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b_.0))
    });
    rows.truncate(k.min(rows.len()));

    let positive: Vec<f64> = rows
        .iter()
        .filter_map(|(_, conf, _)| conf.filter(|v| *v > 0.0))
        .collect();
    let total_positive: f64 = positive.iter().sum();
    let entropy = if total_positive > 0.0 {
        Some(
            positive
                .iter()
                .map(|v| {
                    let p = *v / total_positive;
                    -p * p.ln()
                })
                .sum(),
        )
    } else {
        None
    };
    let effective_n = entropy.map(f64::exp);

    Some(DeepestRankTopK {
        paths: rows.iter().map(|(path, _, _)| path.clone()).collect(),
        confidences: rows.iter().map(|(_, conf, _)| *conf).collect(),
        entropy: finite_opt(entropy),
        effective_n: finite_opt(effective_n),
    })
}

fn ancestor_at_level(group: usize, target_level: i32, ts: &TrainingSet) -> usize {
    let mut p = group;
    loop {
        let level = ts.levels.get(p).copied().unwrap_or(target_level);
        if level <= target_level || p == 0 || ts.parents[p] == p {
            return p;
        }
        p = ts.parents[p];
    }
}

fn build_rank_candidate_top_k(
    scores: &LeafCandidateScores,
    ts: &TrainingSet,
    b: usize,
    rank_index: usize,
    predicteds: &[usize],
    k: usize,
) -> Option<DeepestRankTopK> {
    if k == 0 || b == 0 || scores.tot_hits.is_empty() {
        return None;
    }
    let target_node = *predicteds.get(rank_index)?;
    let target_level = ts
        .levels
        .get(target_node)
        .copied()
        .unwrap_or(rank_index as i32);
    let mut hits_by_node: HashMap<usize, f64> = HashMap::new();
    for (&group, &hits) in scores.unique_groups.iter().zip(scores.tot_hits.iter()) {
        let node = ancestor_at_level(group, target_level, ts);
        *hits_by_node.entry(node).or_insert(0.0) += hits;
    }

    let mut rows: Vec<(String, Option<f64>, f64)> = hits_by_node
        .into_iter()
        .map(|(node, hits)| {
            let confidence = hits / b as f64 * 100.0;
            (
                ts.taxonomy[node].clone(),
                finite_opt(Some(confidence)),
                confidence,
            )
        })
        .collect();
    rows.sort_by(|a, b_| {
        b_.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b_.0))
    });
    rows.truncate(k.min(rows.len()));

    let positive: Vec<f64> = rows
        .iter()
        .filter_map(|(_, conf, _)| conf.filter(|v| *v > 0.0))
        .collect();
    let total_positive: f64 = positive.iter().sum();
    let entropy = if total_positive > 0.0 {
        Some(
            positive
                .iter()
                .map(|v| {
                    let p = *v / total_positive;
                    -p * p.ln()
                })
                .sum(),
        )
    } else {
        None
    };
    let effective_n = entropy.map(f64::exp);

    Some(DeepestRankTopK {
        paths: rows.iter().map(|(path, _, _)| path.clone()).collect(),
        confidences: rows.iter().map(|(_, conf, _)| *conf).collect(),
        entropy: finite_opt(entropy),
        effective_n: finite_opt(effective_n),
    })
}

fn rank_label(ts: &TrainingSet, rank_index: usize) -> String {
    ts.ranks
        .as_ref()
        .and_then(|ranks| ranks.get(rank_index))
        .cloned()
        .unwrap_or_else(|| format!("rank_{}", rank_index))
}

#[allow(clippy::too_many_arguments)]
fn build_rank_trace(
    predicteds: &[usize],
    confidences: &[f64],
    ts: &TrainingSet,
    config: &ClassifyConfig,
    stop_reason: ThresholdStopReason,
    challenge_runner_up_path: Option<&String>,
    challenge_runner_up_confidence: Option<f64>,
) -> Vec<RankTraceRow> {
    if !config.rank_trace_diagnostics {
        return Vec::new();
    }

    let stop_rank_index = match stop_reason {
        ThresholdStopReason::FullPass => None,
        ThresholdStopReason::FailedThreshold { rank_index, .. } => Some(rank_index),
        ThresholdStopReason::LcaCapped { cap_index } => Some(cap_index.saturating_add(1)),
    };

    predicteds
        .iter()
        .enumerate()
        .map(|(i, &node)| {
            let threshold = threshold_for_rank(config, i);
            let confidence = confidences.get(i).copied().unwrap_or(0.0);
            let stop_reason_label = match stop_reason {
                ThresholdStopReason::FullPass => None,
                ThresholdStopReason::FailedThreshold { rank_index, .. } if rank_index == i => {
                    Some("failed_threshold".to_string())
                }
                ThresholdStopReason::LcaCapped { cap_index } if i == cap_index + 1 => {
                    Some("lca_capped".to_string())
                }
                _ => None,
            };
            let stopped_before_or_here = stop_rank_index.is_some_and(|stop_i| i >= stop_i);
            let runner_up_path = if i == predicteds.len().saturating_sub(1) {
                challenge_runner_up_path.cloned()
            } else {
                None
            };
            let runner_up_confidence = if i == predicteds.len().saturating_sub(1) {
                challenge_runner_up_confidence
            } else {
                None
            };
            RankTraceRow {
                rank: rank_label(ts, i),
                selected_path: ts.taxonomy[node].clone(),
                selected_confidence: finite_opt(Some(confidence)),
                threshold: finite_opt(Some(threshold)),
                passed_threshold: !stopped_before_or_here && confidence >= threshold,
                runner_up_path,
                runner_up_confidence: finite_opt(runner_up_confidence),
                stop_reason: stop_reason_label,
            }
        })
        .collect()
}

struct ChallengeDecision {
    original_selection_confidence: Option<f64>,
    runner_up_confidence: Option<f64>,
    margin_delta: Option<f64>,
    margin_ratio: Option<f64>,
    candidate_count: Option<u32>,
    selected_path: Option<String>,
    runner_up_path: Option<String>,
    stop_reason: Option<String>,
    top_k: Option<DeepestRankTopK>,
    decision: Result<(), RescueRejection>,
}

fn evaluate_deepest_rank_challenge(
    predicted_group: usize,
    context: &LeafBootstrapContext<'_>,
    ts: &TrainingSet,
    config: &ClassifyConfig,
    ls: &[usize],
    max_level: i32,
) -> ChallengeDecision {
    let groups = terminal_sibling_endpoint_groups(
        predicted_group,
        ts,
        max_level,
        config.suppress_ancestor_only_groups,
    );
    if groups.is_empty() {
        return ChallengeDecision {
            original_selection_confidence: None,
            runner_up_confidence: None,
            margin_delta: None,
            margin_ratio: None,
            candidate_count: Some(0),
            selected_path: None,
            runner_up_path: None,
            stop_reason: Some(
                deepest_rank_stop_reason_for_rejection(
                    RescueRejection::ChallengeNoCandidatesAfterSuppression,
                )
                .to_string(),
            ),
            top_k: None,
            decision: Err(RescueRejection::ChallengeNoCandidatesAfterSuppression),
        };
    }
    let candidate_count = groups.len() as u32;
    if candidate_count < 2 {
        return ChallengeDecision {
            original_selection_confidence: None,
            runner_up_confidence: None,
            margin_delta: None,
            margin_ratio: None,
            candidate_count: Some(candidate_count),
            selected_path: None,
            runner_up_path: None,
            stop_reason: Some(
                deepest_rank_stop_reason_for_rejection(RescueRejection::CandidateCountLt2)
                    .to_string(),
            ),
            top_k: None,
            decision: Err(RescueRejection::CandidateCountLt2),
        };
    }
    if let Some(cap) = config.max_deepest_rank_challenge_candidates {
        if groups.len() > cap {
            return ChallengeDecision {
                original_selection_confidence: None,
                runner_up_confidence: None,
                margin_delta: None,
                margin_ratio: None,
                candidate_count: Some(candidate_count),
                selected_path: None,
                runner_up_path: None,
                stop_reason: Some(
                    deepest_rank_stop_reason_for_rejection(
                        RescueRejection::ChallengeCandidateCapExceeded,
                    )
                    .to_string(),
                ),
                top_k: None,
                decision: Err(RescueRejection::ChallengeCandidateCapExceeded),
            };
        }
    }

    let mut keep = Vec::new();
    for &group in &groups {
        if let Some(seq_indices) = ts.sequences[group].as_ref() {
            keep.extend(seq_indices);
        }
    }
    let Some(challenge_scores) = score_leaf_candidate_set(&keep, context, ts, config, ls) else {
        return ChallengeDecision {
            original_selection_confidence: None,
            runner_up_confidence: None,
            margin_delta: None,
            margin_ratio: None,
            candidate_count: Some(candidate_count),
            selected_path: None,
            runner_up_path: None,
            stop_reason: Some(
                deepest_rank_stop_reason_for_rejection(RescueRejection::ChallengeScoringEmpty)
                    .to_string(),
            ),
            top_k: None,
            decision: Err(RescueRejection::ChallengeScoringEmpty),
        };
    };
    let top_k = build_deepest_rank_top_k(
        &challenge_scores,
        ts,
        context.b,
        config.deepest_rank_diagnostic_top_k,
    );
    let Some(selected_idx) = challenge_scores
        .unique_groups
        .iter()
        .position(|&g| g == predicted_group)
    else {
        return ChallengeDecision {
            original_selection_confidence: None,
            runner_up_confidence: None,
            margin_delta: None,
            margin_ratio: None,
            candidate_count: Some(candidate_count),
            selected_path: None,
            runner_up_path: None,
            stop_reason: Some(
                deepest_rank_stop_reason_for_rejection(
                    RescueRejection::SelectedMissingFromChallenge,
                )
                .to_string(),
            ),
            top_k,
            decision: Err(RescueRejection::SelectedMissingFromChallenge),
        };
    };

    let selected_conf = challenge_scores.tot_hits[selected_idx] / context.b as f64 * 100.0;
    let selected_path = Some(ts.taxonomy[challenge_scores.unique_groups[selected_idx]].clone());
    let mut runner = f64::NEG_INFINITY;
    let mut runner_idx: Option<usize> = None;
    let mut top = f64::NEG_INFINITY;
    for (idx, &th) in challenge_scores.tot_hits.iter().enumerate() {
        let conf = th / context.b as f64 * 100.0;
        if conf > top {
            top = conf;
        }
        if idx != selected_idx && conf > runner {
            runner = conf;
            runner_idx = Some(idx);
        }
    }
    if runner == f64::NEG_INFINITY {
        runner = 0.0;
    }
    let normalized_runner = if runner.abs() <= ZERO_CONF_EPSILON {
        0.0
    } else {
        runner
    };
    let delta = if selected_conf.is_finite() && normalized_runner.is_finite() {
        Some(selected_conf - normalized_runner)
    } else {
        None
    };
    let ratio = if selected_conf.is_finite() && normalized_runner > 0.0 {
        Some(selected_conf / normalized_runner)
    } else {
        None
    };
    let runner_up_path =
        runner_idx.map(|idx| ts.taxonomy[challenge_scores.unique_groups[idx]].clone());

    let selected_is_top = selected_conf == top;
    let exact_tie = selected_is_top
        && challenge_scores
            .tot_hits
            .iter()
            .enumerate()
            .any(|(idx, &th)| idx != selected_idx && th / context.b as f64 * 100.0 == top);
    let near_tie = selected_is_top
        && config.leaf_tie_margin > 0.0
        && challenge_scores
            .tot_hits
            .iter()
            .enumerate()
            .any(|(idx, &th)| {
                if idx == selected_idx {
                    return false;
                }
                let conf = th / context.b as f64 * 100.0;
                conf != top && conf >= top - top.abs() * config.leaf_tie_margin
            });

    let decision_input = DeepestRankRescueDecisionInput {
        challenge_original_selection: selected_conf,
        challenge_runner: normalized_runner,
        selected_is_unique_challenge_top: selected_is_top && !exact_tie,
        challenge_exact_tie: exact_tie,
        challenge_near_tie: near_tie,
        candidate_cap_exceeded: false,
        floor: config.deepest_rank_margin_floor,
        min_delta: config.deepest_rank_margin_min_delta,
        min_ratio: config.deepest_rank_margin_min_ratio,
        candidate_count,
    };
    let decision = passes_deepest_rank_margin_rescue(&decision_input);
    let stop_reason = match decision {
        Err(RescueRejection::RescueDisabled) => None,
        Err(reason) => Some(deepest_rank_stop_reason_for_rejection(reason).to_string()),
        Ok(()) => None,
    };
    ChallengeDecision {
        original_selection_confidence: finite_opt(Some(selected_conf)),
        runner_up_confidence: finite_opt(Some(normalized_runner)),
        margin_delta: finite_opt(delta),
        margin_ratio: finite_opt(ratio),
        candidate_count: Some(candidate_count),
        selected_path,
        runner_up_path,
        stop_reason,
        top_k,
        decision,
    }
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
    ls: &[usize],
    rng: &mut RRng,
) -> Option<(ClassificationResult, f64)> {
    let children = &ts.children;
    let parents = &ts.parents;
    let sequences = &ts.sequences;
    let taxa = &ts.taxa;

    // Per-rank IDF: pick the row at the descent node's depth.
    let counts: &[f64] = crate::training::idf_row_for_depth(
        &ts.idf_weights_by_rank,
        ts.levels.get(k_node).copied().unwrap_or(1),
    );

    // Gather training sequences
    let subtrees = &children[k_node];
    let mut keep: Vec<usize> = Vec::new();
    for &wi in w_indices {
        if wi < subtrees.len() {
            if let Some(ref sq) = sequences[subtrees[wi]] {
                keep.extend(sq);
            }
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
        for (km, &count) in kmer_counts.iter().enumerate().take(n_possible + 1).skip(1) {
            let c = count as usize;
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

    let u_weights: Vec<f64> = u_sampling
        .iter()
        .map(|&uk| {
            if uk > 0 && (uk as usize) <= counts.len() {
                counts[(uk - 1) as usize]
            } else {
                0.0
            }
        })
        .collect();

    // Compute davg without allocating sampling_weights Vec. The candidate-set
    // scorer reuses this exact sampled-query denominator for both the original
    // pruned contest and any expanded terminal-sibling challenge.
    let davg = {
        let mut row_sums = vec![0.0f64; b];
        for (idx, &sk) in sampling.iter().enumerate() {
            let w = if sk > 0 && (sk as usize) <= counts.len() {
                counts[(sk - 1) as usize]
            } else {
                0.0
            };
            row_sums[idx % b] += w;
        }
        row_sums.iter().sum::<f64>() / b as f64
    };
    let cached = ts.avg_n_unique_per_node[k_node];
    let avg_unique_for_stop_node = if cached > 0.0 {
        cached
    } else {
        ts.avg_n_unique_global
    };
    let context = LeafBootstrapContext {
        u_sampling: &u_sampling,
        positions: &positions,
        ranges: &ranges,
        u_weights: &u_weights,
        b,
        davg,
        avg_unique_for_stop_node,
    };
    let scores = match score_leaf_candidate_set(&keep, &context, ts, config, ls) {
        Some(scores) => scores,
        None => return Some((ClassificationResult::unclassified("no_training_match"), 0.0)),
    };

    let selected = if scores.winners.len() > 1 {
        let idx = rng.sample_int_replace(scores.winners.len(), 1)[0];
        scores.winners[idx]
    } else {
        scores.winners[0]
    };

    // At M=1, similarity is the single-best ref's per-rep sum normalized by
    // davg — bit-identical to the pre-Phase-3 path. At M>1, sum the
    // aggregated per-rep values for the selected group's top M_eff refs.
    let similarity = if davg != 0.0 {
        if config.leaf_top_m.max(1) > 1 {
            let mut s = 0.0;
            for rep in 0..b {
                s += aggregate_top_m_at_rep(
                    &scores.top_m_refs_per_group[selected],
                    rep,
                    b,
                    &scores.selected_hits_flat,
                    config.leaf_aggregation_mode,
                );
            }
            s / davg
        } else {
            let base = scores.top_hits_idx[selected] * b;
            scores.selected_hits_flat[base..base + b]
                .iter()
                .sum::<f64>()
                / davg
        }
    } else {
        0.0
    };

    // Build prediction
    let predicted_group = scores.unique_groups[selected];
    let mut predicteds: Vec<usize> = Vec::new();
    let mut p = predicted_group;
    loop {
        predicteds.push(p);
        if p == 0 || parents[p] == p {
            break;
        }
        p = parents[p];
    }
    predicteds.reverse();

    let base_confidence = scores.tot_hits[selected] / b as f64 * 100.0;
    let mut confidences = vec![base_confidence; predicteds.len()];
    for (j, &th) in scores.tot_hits.iter().enumerate() {
        if th > 0.0 && j != selected {
            // Start at the group's own node (not its parent) so a group whose
            // own node lies on `predicteds` — the case for an ancestor-only
            // training entry whose descendant is `selected` — credits its
            // tot_hits at its OWN rank instead of skipping past it. For
            // sibling/descendant groups, `unique_groups[j]` is not in
            // `predicteds` and the first iteration is a harmless no-match.
            let mut p = scores.unique_groups[j];
            loop {
                if let Some(m) = predicteds.iter().position(|&x| x == p) {
                    confidences[m] += th / b as f64 * 100.0;
                }
                if p == 0 || parents[p] == p {
                    break;
                }
                p = parents[p];
            }
        }
    }

    // When multiple groups are tied at `max_tot`, the classifier cannot honestly
    // resolve below the LCA of the tied set. Compute that LCA's position in
    // `predicteds` so the `above` filter can cap the reportable lineage there.
    //
    // Every non-selected winner's pairwise LCA with `selected` is the deepest
    // ancestor-or-equal of `unique_groups[j]` that lives in `predicteds`. The
    // group-wise LCA is the shallowest (smallest index) of those pairwise
    // LCAs, since it must be an ancestor of every winner. The walk below
    // starts at `unique_groups[j]` itself so an ancestor-only winner's own
    // node is accepted as the cap (rather than its parent's position).
    let (lca_cap, alternatives): (Option<usize>, Vec<String>) = if scores.winners.len() > 1 {
        let mut deepest_allowed = predicteds.len() - 1;
        for &j in &scores.winners {
            if j == selected {
                continue;
            }
            // Start at the winner's own node so the LCA-cap accepts
            // `unique_groups[j]` itself as the cap when this winner is a
            // strict ancestor of `selected` (i.e., the LCA is the winner's
            // own node). The pre-fix start at parents[X] capped one rank
            // too shallow in that case.
            let mut p = scores.unique_groups[j];
            loop {
                if let Some(pos) = predicteds.iter().position(|&x| x == p) {
                    if pos < deepest_allowed {
                        deepest_allowed = pos;
                    }
                    break;
                }
                if p == 0 || parents[p] == p {
                    break;
                }
                p = parents[p];
            }
        }
        let mut alts: Vec<String> = scores
            .winners
            .iter()
            .map(|&w| taxa[scores.unique_groups[w]].clone())
            .collect();
        alts.sort();
        (Some(deepest_allowed), alts)
    } else {
        (None, Vec::new())
    };

    // Build `above` as a contiguous prefix, carrying an explicit stop reason
    // so deepest-rank rescue does not have to infer threshold vs LCA failure
    // from prefix length alone.
    let mut above: Vec<usize> = Vec::with_capacity(confidences.len());
    let mut stop_reason = ThresholdStopReason::FullPass;
    for (i, &c) in confidences.iter().enumerate() {
        if let Some(cap) = lca_cap {
            if i > cap {
                stop_reason = ThresholdStopReason::LcaCapped { cap_index: cap };
                break;
            }
        }
        let thresh = threshold_for_rank(config, i);
        if c >= thresh {
            above.push(i);
        } else {
            stop_reason = ThresholdStopReason::FailedThreshold {
                rank_index: i,
                threshold: thresh,
                confidence: c,
            };
            break;
        }
    }

    let deepest_idx = predicteds.len() - 1;
    let max_level = ts.levels.iter().copied().max().unwrap_or(0);
    let deepest_eligible = is_deepest_rank_margin_eligible(predicted_group, ts, max_level);
    let deepest_threshold = threshold_for_rank(config, deepest_idx);
    let mut deepest_rank_acceptance_mode: Option<String> = None;
    let mut deepest_rank_effective_confidence: Option<f64> = None;
    let mut deepest_rank_effective_threshold: Option<f64> = None;
    let mut deepest_rank_margin_rejection_reason: Option<String> = None;
    let mut challenge_original_selection_confidence: Option<f64> = None;
    let mut challenge_runner_up_confidence: Option<f64> = None;
    let mut challenge_margin_delta: Option<f64> = None;
    let mut challenge_margin_ratio: Option<f64> = None;
    let mut challenge_candidate_count: Option<u32> = None;
    let mut challenge_selected_path: Option<String> = None;
    let mut challenge_runner_up_path: Option<String> = None;
    let mut deepest_rank_stop_reason: Option<String> = None;
    let mut deepest_rank_top_k: Option<DeepestRankTopK> = None;

    let deepest_rank_diagnostics_enabled = config.deepest_rank_diagnostics
        || config.deepest_rank_margin_floor.is_some()
        || config.deepest_rank_diagnostic_top_k > 0;
    let rescue_enabled = config.deepest_rank_margin_floor.is_some();
    if deepest_eligible && deepest_rank_diagnostics_enabled {
        deepest_rank_effective_confidence = finite_opt(confidences.get(deepest_idx).copied());
        deepest_rank_effective_threshold = finite_opt(Some(deepest_threshold));
        challenge_selected_path = Some(ts.taxonomy[predicted_group].clone());
        match stop_reason {
            ThresholdStopReason::FullPass => {
                deepest_rank_acceptance_mode = Some(DEEPEST_MODE_THRESHOLD.to_string());
                deepest_rank_stop_reason = Some("full_pass".to_string());
            }
            ThresholdStopReason::LcaCapped { cap_index } => {
                if cap_index < deepest_idx {
                    deepest_rank_acceptance_mode = Some(DEEPEST_MODE_REJECTED.to_string());
                    deepest_rank_margin_rejection_reason =
                        Some(RescueRejection::LcaCap.as_str().to_string());
                    deepest_rank_stop_reason = Some("lca_capped".to_string());
                }
            }
            ThresholdStopReason::FailedThreshold {
                rank_index,
                threshold,
                confidence,
            } => {
                if rank_index < deepest_idx {
                    deepest_rank_acceptance_mode = Some(DEEPEST_MODE_REJECTED.to_string());
                    deepest_rank_margin_rejection_reason =
                        Some(RescueRejection::ShallowerRankFailed.as_str().to_string());
                    deepest_rank_stop_reason = Some("shallower_threshold_failed".to_string());
                } else if rank_index == deepest_idx {
                    deepest_rank_effective_confidence = finite_opt(Some(confidence));
                    deepest_rank_effective_threshold = finite_opt(Some(threshold));
                    deepest_rank_acceptance_mode = Some(DEEPEST_MODE_REJECTED.to_string());
                    deepest_rank_stop_reason = Some("deepest_threshold_failed".to_string());
                    let challenge = evaluate_deepest_rank_challenge(
                        predicted_group,
                        &context,
                        ts,
                        config,
                        ls,
                        max_level,
                    );
                    challenge_original_selection_confidence =
                        challenge.original_selection_confidence;
                    challenge_runner_up_confidence = challenge.runner_up_confidence;
                    challenge_margin_delta = challenge.margin_delta;
                    challenge_margin_ratio = challenge.margin_ratio;
                    challenge_candidate_count = challenge.candidate_count;
                    challenge_selected_path = challenge.selected_path;
                    challenge_runner_up_path = challenge.runner_up_path;
                    deepest_rank_top_k = challenge.top_k;
                    if let Some(reason) = challenge.stop_reason {
                        deepest_rank_stop_reason = Some(reason);
                    }
                    match challenge.decision {
                        Ok(()) if rescue_enabled => {
                            above.push(deepest_idx);
                            deepest_rank_acceptance_mode =
                                Some(DEEPEST_MODE_MARGIN_RESCUE.to_string());
                            deepest_rank_margin_rejection_reason = None;
                        }
                        Ok(()) => {}
                        Err(reason) => {
                            deepest_rank_margin_rejection_reason =
                                Some(reason.as_str().to_string());
                        }
                    }
                }
            }
        }
    }

    // After `winners` and `selected` are settled but before the LCA-cap is
    // applied, compute leaf-phase telemetry for the F.0 frequency probe and
    // F-stratification analysis. Telemetry is keyed off the
    // **pre-LCA-cap selected leaf group**: the group whose
    // `unique_groups[selected]` won the post-bootstrap winners cutoff in
    // `leaf_phase_score`. The reported `taxon` Vec may be collapsed to a
    // higher rank by LCA-cap (when winners.len() > 1) or by the per-rank
    // threshold walk; `reference_depth_of_predicted` and the other group-
    // level diagnostics still report the leaf-level values. Use
    // `taxon.len()` to detect that the lineage has been truncated.
    //
    // We compute here (before the threshold-walk truncation builds
    // `result.taxon`) so values are available for both classified and
    // partial-truncation branches; Path-C abstention zeroes them downstream.
    //
    // `lookup` was set above as `keep[i] -> cross_index[keep[i]]`, so refs
    // belonging to the selected group are exactly those `keep[i]` with
    // `lookup[i] == predicted_group`. We pull their `sum_hits` values to
    // compute n_refs, n_close_representatives, dispersion, and the
    // within-vs-between margin against the rest of `keep`.
    let reference_depth_of_predicted: u32 = sequences
        .get(predicted_group)
        .and_then(|s| s.as_ref().map(|v| v.len() as u32))
        .unwrap_or(0);

    let (
        leaf_winning_group_n_refs,
        leaf_n_close_representatives,
        leaf_top_m_score_dispersion,
        leaf_within_vs_between_margin,
    ) = {
        let mut in_group: Vec<f64> = Vec::new();
        let mut out_group_max = f64::NEG_INFINITY;
        for (i, &lk) in scores.lookup.iter().enumerate() {
            let v = scores.sum_hits[i];
            if lk == predicted_group {
                in_group.push(v);
            } else if v > out_group_max {
                out_group_max = v;
            }
        }
        let n_refs = in_group.len() as u32;
        let mut n_close: u32 = 0;
        let mut dispersion: f64 = 0.0;
        let mut margin: f64 = 0.0;
        if !in_group.is_empty() {
            let in_max = in_group.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            const LEAF_CLOSE_REP_RATIO: f64 = 0.95;
            // Negative-safe form matching the post-bootstrap winner cutoff
            // above (`max - |max| * margin`). Equivalent to `0.95 * max` for
            // positive max; correct under negative IDF-induced sums.
            let cutoff = in_max - in_max.abs() * (1.0 - LEAF_CLOSE_REP_RATIO);
            n_close = in_group.iter().filter(|&&v| v >= cutoff).count() as u32;

            if in_group.len() >= 5 {
                let mut sorted = in_group.clone();
                sorted.sort_by(|a, b_| b_.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let top5 = &sorted[..5];
                let mean = top5.iter().sum::<f64>() / 5.0;
                if mean > 0.0 {
                    // Sample variance (n-1 denominator) for an unbiased
                    // estimator on the small top-5 sample.
                    let var = top5.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 4.0;
                    dispersion = var.sqrt() / mean;
                }
                if out_group_max > f64::NEG_INFINITY && in_max > 0.0 {
                    margin = (in_max - out_group_max) / in_max;
                }
            }
        }
        (n_refs, n_close, dispersion, margin)
    };
    let common_leaf_top_confidence = deepest_rank_diagnostics_enabled
        .then(|| finite_opt(scores.top_confidence))
        .flatten();
    let common_leaf_runner_up_confidence = deepest_rank_diagnostics_enabled
        .then(|| finite_opt(scores.runner_up_confidence))
        .flatten();
    let common_leaf_margin_delta = deepest_rank_diagnostics_enabled
        .then(|| finite_opt(scores.margin_delta))
        .flatten();
    let common_leaf_margin_ratio = deepest_rank_diagnostics_enabled
        .then(|| finite_opt(scores.margin_ratio))
        .flatten();
    let common_leaf_margin_candidate_count =
        deepest_rank_diagnostics_enabled.then_some(scores.candidate_count);
    let failed_rank_top_k = match stop_reason {
        ThresholdStopReason::FailedThreshold { rank_index, .. }
            if config.deepest_rank_diagnostic_top_k > 0 =>
        {
            build_rank_candidate_top_k(
                &scores,
                ts,
                b,
                rank_index,
                &predicteds,
                config.deepest_rank_diagnostic_top_k,
            )
        }
        _ => None,
    };
    let rank_trace = build_rank_trace(
        &predicteds,
        &confidences,
        ts,
        config,
        stop_reason,
        challenge_runner_up_path.as_ref(),
        challenge_runner_up_confidence,
    );

    let result = if above.len() == predicteds.len() {
        ClassificationResult {
            taxon: predicteds.iter().map(|&p| taxa[p].clone()).collect(),
            confidence: confidences,
            alternatives: alternatives.clone(),
            reject_reason: None,
            similarity,
            beam_candidate_count: 1,
            leaf_widened_winners: scores.leaf_widened_winners,
            leaf_winning_group_n_refs,
            leaf_n_close_representatives,
            leaf_top_m_score_dispersion,
            leaf_within_vs_between_margin,
            reference_depth_of_predicted,
            deepest_rank_acceptance_mode: deepest_rank_acceptance_mode.clone(),
            deepest_rank_effective_confidence,
            deepest_rank_effective_threshold,
            deepest_rank_margin_rejection_reason: deepest_rank_margin_rejection_reason.clone(),
            leaf_top_confidence: common_leaf_top_confidence,
            leaf_runner_up_confidence: common_leaf_runner_up_confidence,
            leaf_margin_delta: common_leaf_margin_delta,
            leaf_margin_ratio: common_leaf_margin_ratio,
            leaf_margin_candidate_count: common_leaf_margin_candidate_count,
            deepest_rank_challenge_original_selection_confidence:
                challenge_original_selection_confidence,
            deepest_rank_challenge_runner_up_confidence: challenge_runner_up_confidence,
            deepest_rank_challenge_margin_delta: challenge_margin_delta,
            deepest_rank_challenge_margin_ratio: challenge_margin_ratio,
            deepest_rank_challenge_candidate_count: challenge_candidate_count,
            deepest_rank_challenge_selected_path: challenge_selected_path.clone(),
            deepest_rank_challenge_runner_up_path: challenge_runner_up_path.clone(),
            deepest_rank_stop_reason: deepest_rank_stop_reason.clone(),
            deepest_rank_top_k: deepest_rank_top_k.clone(),
            failed_rank_top_k: failed_rank_top_k.clone(),
            rank_trace: rank_trace.clone(),
            ..Default::default()
        }
    } else if above.is_empty() {
        // Path C — abstention: every rank (including Root) fell below its
        // threshold. Collapse to ["Root"], but keep new deepest-rank rescue
        // audit diagnostics from the pre-truncation selected group.
        ClassificationResult {
            taxon: vec![taxa[predicteds[0]].clone()],
            confidence: vec![confidences[0]],
            alternatives: Vec::new(),
            reject_reason: Some("below_threshold".to_string()),
            similarity,
            deepest_rank_acceptance_mode: deepest_rank_acceptance_mode.clone(),
            deepest_rank_effective_confidence,
            deepest_rank_effective_threshold,
            deepest_rank_margin_rejection_reason: deepest_rank_margin_rejection_reason.clone(),
            leaf_top_confidence: common_leaf_top_confidence,
            leaf_runner_up_confidence: common_leaf_runner_up_confidence,
            leaf_margin_delta: common_leaf_margin_delta,
            leaf_margin_ratio: common_leaf_margin_ratio,
            leaf_margin_candidate_count: common_leaf_margin_candidate_count,
            deepest_rank_challenge_original_selection_confidence:
                challenge_original_selection_confidence,
            deepest_rank_challenge_runner_up_confidence: challenge_runner_up_confidence,
            deepest_rank_challenge_margin_delta: challenge_margin_delta,
            deepest_rank_challenge_margin_ratio: challenge_margin_ratio,
            deepest_rank_challenge_candidate_count: challenge_candidate_count,
            deepest_rank_challenge_selected_path: challenge_selected_path.clone(),
            deepest_rank_challenge_runner_up_path: challenge_runner_up_path.clone(),
            deepest_rank_stop_reason: deepest_rank_stop_reason.clone(),
            deepest_rank_top_k: deepest_rank_top_k.clone(),
            failed_rank_top_k: failed_rank_top_k.clone(),
            rank_trace: rank_trace.clone(),
            ..Default::default()
        }
    } else {
        // Partial truncation — some ranks classified, some fell below
        // threshold. Diagnostics remain valid (they describe the leaf-phase
        // selected group, which still exists), but the reportable lineage
        // is the contiguous prefix in `above`.
        let taxon: Vec<String> = above.iter().map(|&i| taxa[predicteds[i]].clone()).collect();
        let conf: Vec<f64> = above.iter().map(|&i| confidences[i]).collect();
        ClassificationResult {
            taxon,
            confidence: conf,
            alternatives,
            reject_reason: Some("below_threshold".to_string()),
            similarity,
            beam_candidate_count: 1,
            leaf_widened_winners: scores.leaf_widened_winners,
            leaf_winning_group_n_refs,
            leaf_n_close_representatives,
            leaf_top_m_score_dispersion,
            leaf_within_vs_between_margin,
            reference_depth_of_predicted,
            deepest_rank_acceptance_mode,
            deepest_rank_effective_confidence,
            deepest_rank_effective_threshold,
            deepest_rank_margin_rejection_reason,
            leaf_top_confidence: common_leaf_top_confidence,
            leaf_runner_up_confidence: common_leaf_runner_up_confidence,
            leaf_margin_delta: common_leaf_margin_delta,
            leaf_margin_ratio: common_leaf_margin_ratio,
            leaf_margin_candidate_count: common_leaf_margin_candidate_count,
            deepest_rank_challenge_original_selection_confidence:
                challenge_original_selection_confidence,
            deepest_rank_challenge_runner_up_confidence: challenge_runner_up_confidence,
            deepest_rank_challenge_margin_delta: challenge_margin_delta,
            deepest_rank_challenge_margin_ratio: challenge_margin_ratio,
            deepest_rank_challenge_candidate_count: challenge_candidate_count,
            deepest_rank_challenge_selected_path: challenge_selected_path,
            deepest_rank_challenge_runner_up_path: challenge_runner_up_path,
            deepest_rank_stop_reason,
            deepest_rank_top_k,
            failed_rank_top_k,
            rank_trace,
            ..Default::default()
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
    for i in 0..n {
        iteration.push((i, None));
    }
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

        if let Some((result, similarity)) =
            classify_one_pass(my_kmers, s, b, ts, config, &pre.ls, rng)
        {
            if let Some(_ri) = rev_idx {
                // Second pass: only replace if better similarity
                if similarity <= sims[seq_idx] {
                    continue;
                }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    let n = pre.test_kmers.len();
    // Diagnostic gate: when OXIDTAXA_DIAG is set in the env (any value), emit
    // thread-saturation telemetry around the outer parallel iter. Used to
    // disambiguate splitter under-division vs pool/registry confusion vs
    // per-query work variance.
    let diag = std::env::var_os("OXIDTAXA_DIAG").is_some();
    let pool_threads = rayon::current_num_threads();

    if diag {
        eprintln!(
            "[oxidtaxa-diag] classify_parallel n={} current_num_threads={} caller_idx={:?}",
            n,
            pool_threads,
            rayon::current_thread_index()
        );
    }

    let per_thread: Vec<AtomicUsize> = if diag {
        (0..pool_threads).map(|_| AtomicUsize::new(0)).collect()
    } else {
        Vec::new()
    };

    // Optionally cap chunk size to force rayon's splitter to subdivide more
    // aggressively. Set OXIDTAXA_MAX_CHUNK=<n> to enable. Useful for diagnosing
    // whether under-saturation in the outer iter is splitter-related.
    let max_chunk: Option<usize> = std::env::var("OXIDTAXA_MAX_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    let map_body = |i: usize| -> ClassificationResult {
        if diag {
            if let Some(idx) = rayon::current_thread_index() {
                if idx < per_thread.len() {
                    per_thread[idx].fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        let mut rng = RRng::new(seed ^ (i as u32));
        let s = pre.s_values[i];
        let b = pre.b_values[i];

        // Forward pass
        let fwd = classify_one_pass(&pre.test_kmers[i], s, b, ts, config, &pre.ls, &mut rng);

        // Reverse pass (if "both" strand)
        if let Some(&both_pos) = pre.boths_map.get(&i) {
            let rev = classify_one_pass(
                &pre.rev_kmers[both_pos],
                s,
                b,
                ts,
                config,
                &pre.ls,
                &mut rng,
            );
            match (fwd, rev) {
                (Some((fwd_r, fwd_s)), Some((rev_r, rev_s))) => {
                    if rev_s > fwd_s {
                        rev_r
                    } else {
                        fwd_r
                    }
                }
                (Some((r, _)), None) => r,
                (None, Some((r, _))) => r,
                (None, None) => ClassificationResult::unclassified("no_result"),
            }
        } else {
            fwd.map(|(r, _)| r)
                .unwrap_or_else(|| ClassificationResult::unclassified("no_result"))
        }
    };

    let results: Vec<ClassificationResult> = if let Some(c) = max_chunk {
        (0..n)
            .into_par_iter()
            .with_max_len(c)
            .map(map_body)
            .collect()
    } else {
        (0..n).into_par_iter().map(map_body).collect()
    };

    if diag {
        let counts: Vec<usize> = per_thread
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let active = counts.iter().filter(|&&c| c > 0).count();
        let max_count = counts.iter().max().copied().unwrap_or(0);
        let min_active = counts
            .iter()
            .filter(|&&c| c > 0)
            .min()
            .copied()
            .unwrap_or(0);
        eprintln!(
            "[oxidtaxa-diag] distinct workers used: {}/{} (active item-count range: {}..={})",
            active, pool_threads, min_active, max_count
        );
        eprintln!("[oxidtaxa-diag] per-thread items: {:?}", counts);
    }

    results
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
        let p_single = 1.0
            - pbinom(
                (min_s_f * 0.5 - 1.0).floor() as usize,
                min_s,
                l_quantile / n_kmers as f64,
            );
        let p_any = 1.0 - pbinom(0, n_seqs, p_single);
        if p_any < 0.01 {
            return min_s;
        }
    }
    100
}

/// Cumulative binomial distribution function P(X <= k).
fn pbinom(k: usize, n: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
    }
    let mut sum = 0.0f64;
    let mut binom_coeff = 1.0f64;
    let q = 1.0 - p;
    for i in 0..=k.min(n) {
        if i > 0 {
            binom_coeff *= (n - i + 1) as f64 / i as f64;
        }
        sum += binom_coeff * p.powi(i as i32) * q.powi((n - i) as i32);
    }
    sum.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::effective_bootstrap_replicates;

    #[test]
    fn bootstrap_coverage_cap_can_be_disabled() {
        assert_eq!(effective_bootstrap_replicates(100, 20, 100, true), 25);
        assert_eq!(effective_bootstrap_replicates(100, 20, 100, false), 100);
    }

    #[test]
    fn bootstrap_coverage_cap_still_respects_configured_ceiling() {
        assert_eq!(effective_bootstrap_replicates(1_000, 20, 100, true), 100);
    }
}
