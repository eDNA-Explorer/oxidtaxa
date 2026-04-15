use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use rayon::prelude::*;

use crate::alphabet::alphabet_size;
use crate::kmer::{enumerate_sequences, parse_seed_pattern, SpacedSeed, NA_INTEGER};
use crate::matching::{int_match, vector_sum};
use crate::rng::{mix_seed, RRng};
use crate::types::{DecisionNode, DescendantWeighting, ProblemSequence, TrainConfig, TrainingSet};

/// Train an IDTAXA classifier.
/// Port of R/LearnTaxa.R (487 lines).
#[allow(clippy::too_many_arguments)]
pub fn learn_taxa(
    sequences: &[String],
    taxonomy_strings: &[String],
    config: &TrainConfig,
    seed: u32,
    _verbose: bool,
) -> Result<TrainingSet, String> {
    let l = sequences.len();
    if l < 2 {
        return Err("At least two training sequences are required.".to_string());
    }
    if taxonomy_strings.len() != l {
        return Err("taxonomy must be the same length as train.".to_string());
    }

    // Use a local thread pool so processors is always respected.
    // Without this, par_chunks/par_iter hit the global Rayon pool
    // which defaults to 1 thread per core.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.processors)
        .build()
        .map_err(|e| format!("failed to create rayon thread pool: {e}"))?;

    pool.install(|| _learn_taxa_inner(sequences, taxonomy_strings, config, seed))
}

fn _learn_taxa_inner(
    sequences: &[String],
    taxonomy_strings: &[String],
    config: &TrainConfig,
    seed: u32,
) -> Result<TrainingSet, String> {
    let l = sequences.len();

    // DNA-only: fixed alphabet size
    let size: usize = 4;

    // Parse spaced seed if provided
    let spaced_seed: Option<SpacedSeed> = match &config.seed_pattern {
        Some(pat) => Some(parse_seed_pattern(pat)?),
        None => None,
    };

    // Compute K if not specified (lines 47-75)
    // When using a spaced seed, k = seed.weight (effective k-mer size)
    let k = match (&spaced_seed, config.k) {
        (Some(seed), _) => seed.weight,
        (None, Some(k)) => k,
        (None, None) => {
            let quant = percentile_nchar(sequences, 0.99);
            let as_val = alphabet_size(sequences);
            let computed = ((config.n * quant as f64).ln() / as_val.ln()).floor() as usize;
            let max_k = 13;
            if computed < 1 {
                1
            } else if computed > max_k {
                max_k
            } else {
                computed
            }
        }
    };

    let n_kmers = size.pow(k as u32);
    let b: usize = 100;

    // Parse taxonomy: strip "Root;" prefix, normalize (lines 122-131)
    let classes: Vec<String> = taxonomy_strings
        .iter()
        .map(|t| {
            let t = t.replace(" ; ", ";").replace("; ", ";");
            let after_root = t
                .split("Root;")
                .nth(1)
                .unwrap_or("")
                .trim_end()
                .to_string();
            if after_root.ends_with(';') {
                after_root
            } else {
                format!("{};", after_root)
            }
        })
        .collect();

    // Enumerate k-mers (lines 134-146)
    let raw_kmers = enumerate_sequences(sequences, k, false, false, &[], true, spaced_seed.as_ref());
    let kmers: Vec<Vec<i32>> = raw_kmers
        .into_iter()
        .map(|v| {
            let mut sorted: Vec<i32> = v
                .into_iter()
                .filter(|&x| x != NA_INTEGER)
                .map(|x| x + 1) // 1-index like R
                .collect();
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        })
        .collect();

    // Build inverted index: for each possible k-mer, which sequences have it
    let mut inverted_index: Vec<Vec<u32>> = vec![Vec::new(); n_kmers];
    for (seq_idx, seq_kmers) in kmers.iter().enumerate() {
        for &km in seq_kmers {
            if km > 0 && (km as usize) <= n_kmers {
                inverted_index[(km - 1) as usize].push(seq_idx as u32);
            }
        }
    }

    // Build taxonomy tree (lines 155-198)
    let u_classes: Vec<String> = {
        let mut uc_set: HashSet<String> = HashSet::new();
        let mut uc: Vec<String> = Vec::new();
        for c in &classes {
            if uc_set.insert(c.clone()) {
                uc.push(c.clone());
            }
        }
        uc.sort();
        uc
    };

    // Build all taxonomy prefixes
    let mut taxa_set: HashSet<String> = HashSet::new();
    let mut all_taxa: Vec<String> = Vec::new();
    for uc in &u_classes {
        let parts: Vec<&str> = uc.split(';').filter(|s| !s.is_empty()).collect();
        for n in 1..=parts.len() {
            let prefix: String = parts[..n].iter().map(|s| format!("{};", s)).collect();
            if taxa_set.insert(prefix.clone()) {
                all_taxa.push(prefix);
            }
        }
    }

    // taxonomy[0] = "Root;", taxonomy[1..] = "Root;" + each prefix
    let mut taxonomy: Vec<String> = vec!["Root;".to_string()];
    for t in &all_taxa {
        taxonomy.push(format!("Root;{}", t));
    }

    // crossIndex: which taxonomy node each sequence belongs to (lines 169)
    // R uses 1-indexed: match(classes, taxonomy) + 1 offset for Root
    // Our taxonomy[0] = "Root;", taxonomy[1..] = "Root;" + prefix
    // So crossIndex[i] = position in all_taxa + 1 (for Root offset)
    // This should match R's 1-indexed crossIndex
    let taxa_to_idx: HashMap<&str, usize> = all_taxa
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i + 1)) // +1 for Root offset
        .collect();
    let cross_index: Vec<usize> = classes
        .iter()
        .map(|c| *taxa_to_idx.get(c.as_str()).unwrap_or(&0))
        .collect();

    // Extract taxa names and levels (lines 172-174)
    let taxa: Vec<String> = taxonomy
        .iter()
        .map(|t| {
            let parts: Vec<&str> = t.split(';').filter(|s| !s.is_empty()).collect();
            parts.last().unwrap_or(&"Root").to_string()
        })
        .collect();

    let levels: Vec<i32> = taxonomy
        .iter()
        .map(|t| t.split(';').filter(|s| !s.is_empty()).count() as i32)
        .collect();

    // Build children/parents (lines 176-202)
    let max_level = *levels.iter().max().unwrap_or(&1) - 1;
    let mut levs: Vec<Vec<usize>> = vec![Vec::new(); max_level as usize];
    for (i, &lev) in levels.iter().enumerate() {
        let idx = lev - 2; // level (i+1) maps to levs[i-1], but we want level=lev at levs[lev-2]
        if idx >= 0 && (idx as usize) < levs.len() {
            levs[idx as usize].push(i);
        }
    }

    let mut starts = vec![0usize; max_level as usize];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); taxonomy.len()];
    for i in 0..taxonomy.len() {
        let j = levels[i] - 1; // 0-indexed level
        if j >= 0 && (j as usize) < levs.len() {
            let j = j as usize;
            while starts[j] < levs[j].len() && levs[j][starts[j]] <= i {
                starts[j] += 1;
            }
            if starts[j] < levs[j].len() {
                let w: Vec<usize> = levs[j][starts[j]..]
                    .iter()
                    .filter(|&&idx| taxonomy[idx].starts_with(&taxonomy[i]))
                    .copied()
                    .collect();
                children[i] = w;
            }
        }
    }

    let mut parents = vec![0usize; taxonomy.len()];
    for (i, ch) in children.iter().enumerate() {
        for &c in ch {
            parents[c] = i;
        }
    }

    // Find sequences for each taxonomy node (lines 252-264)
    // O(n) approach: for each sequence, walk its taxonomy prefix chain and assign to all ancestors.
    let end_taxonomy: Vec<String> = taxonomy
        .iter()
        .map(|t| {
            if t.len() > 5 {
                t[5..].to_string() // strip "Root;"
            } else {
                String::new()
            }
        })
        .collect();

    // Build end_taxonomy → taxonomy index for O(1) lookup
    let mut et_to_idx: HashMap<&str, usize> = HashMap::with_capacity(end_taxonomy.len());
    for (i, et) in end_taxonomy.iter().enumerate() {
        et_to_idx.insert(et.as_str(), i);
    }

    let mut sequences_per_node: Vec<Option<Vec<usize>>> = vec![None; taxonomy.len()];
    // Root node (index 0, end_taxonomy="") matches all sequences via starts_with("")
    sequences_per_node[0] = Some((0..classes.len()).collect());
    for (seq_idx, class) in classes.iter().enumerate() {
        // Walk all prefixes of this class string to assign to ancestor nodes
        let parts: Vec<&str> = class.split(';').filter(|s| !s.is_empty()).collect();
        for n in 1..=parts.len() {
            let prefix: String = parts[..n].iter().map(|s| format!("{};", s)).collect();
            if let Some(&tax_idx) = et_to_idx.get(prefix.as_str()) {
                sequences_per_node[tax_idx]
                    .get_or_insert_with(Vec::new)
                    .push(seq_idx);
            }
        }
    }

    let n_seqs: Vec<usize> = sequences_per_node
        .iter()
        .map(|s| s.as_ref().map_or(0, |v| v.len()))
        .collect();

    // Build decision tree (lines 267-321)
    // Sibling subtrees are processed in parallel via rayon; each returns
    // its collected (node_index, DecisionNode) pairs which we scatter here.
    let mut decision_kmers: Vec<Option<DecisionNode>> = vec![None; taxonomy.len()];
    let (_root_profile, _root_desc, nodes) = create_tree(
        0,
        &children,
        &sequences_per_node,
        &kmers,
        n_kmers,
        config,
    );
    for (idx, dk) in nodes {
        decision_kmers[idx] = Some(dk);
    }
    // Compute IDF weights early (needed by fraction loop when use_idf_in_training=true,
    // and always stored in the final model). IDF doesn't depend on fractions.
    let class_counts = {
        let mut counts = HashMap::new();
        for c in &classes {
            *counts.entry(c.clone()).or_insert(0usize) += 1;
        }
        counts
    };
    let n_classes = class_counts.len();
    let idf_seq_weights: Vec<f64> = classes
        .iter()
        .map(|c| 1.0 / *class_counts.get(c).unwrap() as f64)
        .collect();

    let chunk_size = 256;
    let partial_counts: Vec<Vec<f64>> = kmers
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk_kmers)| {
            let mut local = vec![0.0f64; n_kmers];
            let base = chunk_idx * chunk_size;
            for (local_i, class_kmers) in chunk_kmers.iter().enumerate() {
                let w = idf_seq_weights[base + local_i];
                for &km in class_kmers {
                    if km > 0 && (km as usize) <= n_kmers {
                        local[(km - 1) as usize] += w;
                    }
                }
            }
            local
        })
        .collect();

    let mut idf_counts = vec![0.0f64; n_kmers];
    for partial in &partial_counts {
        for (i, &v) in partial.iter().enumerate() {
            idf_counts[i] += v;
        }
    }
    let idf_weights: Vec<f64> = idf_counts
        .iter()
        .map(|&c| (n_classes as f64 / (1.0 + c)).ln())
        .collect();

    // Learn fractions (lines 323-420) — batch, order-independent
    // Pre-compute per-sequence identity hashes for deterministic PRNG seeding
    // regardless of input order. Hash(sequence content + taxonomy) → unique identity.
    let seq_hashes: Vec<u64> = sequences
        .iter()
        .zip(taxonomy_strings.iter())
        .map(|(seq, tax)| {
            let mut hasher = DefaultHasher::new();
            seq.hash(&mut hasher);
            tax.hash(&mut hasher);
            hasher.finish()
        })
        .collect();

    let mut fraction: Vec<Option<f64>> = vec![Some(config.max_fraction); taxonomy.len()];
    let mut incorrect: Vec<Option<bool>> = vec![Some(true); l]; // Some(true)=incorrect, Some(false)=correct, None=gave up
    let mut predicted = vec![String::new(); l];
    let delta = (config.max_fraction - config.min_fraction) * config.multiplier;

    for _it in 0..config.max_iterations {
        let remaining: Vec<usize> = incorrect
            .iter()
            .enumerate()
            .filter(|(_, v)| *v == &Some(true))
            .map(|(i, _)| i)
            .collect();

        if remaining.is_empty() {
            break;
        }

        // Snapshot fractions so all sequences see the same values this iteration
        let fraction_snapshot = fraction.clone();

        // Classify all remaining sequences (order-independent, parallel)
        let results: Vec<(usize, bool, usize, String)> = remaining
            .par_iter()
            .map(|&i| {
                if kmers[i].is_empty() {
                    return (i, true, 0, String::new());
                }

                let mut seq_rng =
                    RRng::new(mix_seed(seed, (_it as u64) * 1_000_000 + seq_hashes[i]));
                let mut k_node = 0usize;
                let mut correct = true;
                let mut pred = String::new();

                loop {
                    let subtrees = &children[k_node];
                    let dk = &decision_kmers[k_node];

                    if dk.is_none() || dk.as_ref().unwrap().keep.is_empty() {
                        break;
                    }

                    let dk = dk.as_ref().unwrap();
                    let n = dk.keep.len();

                    if subtrees.len() > 1 {
                        let s = match fraction_snapshot[k_node] {
                            None => ((n as f64) * config.min_fraction).ceil() as usize,
                            Some(f) => ((n as f64) * f).ceil() as usize,
                        };

                        let sampling = seq_rng.sample_int_replace(n, s * b);

                        let matches = int_match(&dk.keep, &kmers[i]);

                        // LOO: find which subtree this sequence belongs to
                        // and the group size for its subtree
                        let loo_child_idx = if config.leave_one_out {
                            subtrees.iter().enumerate().find(|(_, &st)| {
                                classes[i].starts_with(&end_taxonomy[st])
                            }).map(|(j, &st)| {
                                let group_size = n_seqs[st];
                                (j, group_size)
                            })
                        } else {
                            None
                        };

                        let mut hits = vec![vec![0.0f64; b]; subtrees.len()];
                        for (j, _subtree) in subtrees.iter().enumerate() {
                            let mut weights_j: Vec<f64> = if config.use_idf_in_training {
                                dk.profiles[j].iter().zip(dk.keep.iter())
                                    .map(|(&prof, &km)| {
                                        let idf = if km > 0 && (km as usize) <= idf_weights.len() {
                                            idf_weights[(km - 1) as usize]
                                        } else { 0.0 };
                                        prof * idf
                                    })
                                    .collect()
                            } else {
                                dk.profiles[j].clone()
                            };

                            // LOO adjustment: reduce weights for this sequence's own subtree
                            if let Some((loo_j, group_size)) = loo_child_idx {
                                if j == loo_j && group_size <= 5 {
                                    if group_size <= 1 {
                                        // Singleton: zero out weights (can't self-classify)
                                        for w in &mut weights_j { *w = 0.0; }
                                    } else {
                                        // Scale by (n-1)/n to approximate removing this sequence
                                        let scale = (group_size - 1) as f64 / group_size as f64;
                                        for w in &mut weights_j { *w *= scale; }
                                    }
                                }
                            }

                            hits[j] =
                                vector_sum(&matches, &weights_j, &sampling, b);
                        }

                        let mut vote_counts = vec![0usize; subtrees.len()];
                        for rep in 0..b {
                            let max_val =
                                hits.iter().map(|h| h[rep]).fold(0.0f64, f64::max);
                            if max_val > 0.0 {
                                for (j, h) in hits.iter().enumerate() {
                                    if h[rep] == max_val {
                                        vote_counts[j] += 1;
                                    }
                                }
                            }
                        }

                        let w = vote_counts
                            .iter()
                            .enumerate()
                            .max_by_key(|(_, &c)| c)
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);

                        if vote_counts[w] < ((b as f64) * config.training_threshold) as usize {
                            break;
                        }

                        if !classes[i].starts_with(&end_taxonomy[subtrees[w]]) {
                            correct = false;
                            pred = taxonomy[subtrees[w]].clone();
                            break;
                        }

                        if children[subtrees[w]].is_empty() {
                            break;
                        }
                        k_node = subtrees[w];
                    } else {
                        if children[subtrees[0]].is_empty() {
                            break;
                        }
                        k_node = subtrees[0];
                    }
                }

                (i, correct, k_node, pred)
            })
            .collect();

        // Apply batch updates: collect failures per node, then update fractions
        let mut node_failures: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(seq_idx, correct, fail_node, ref pred) in &results {
            if correct {
                incorrect[seq_idx] = Some(false);
            } else {
                predicted[seq_idx] = pred.clone();
                node_failures.entry(fail_node).or_default().push(seq_idx);
            }
        }

        // Apply capped decrements per node: each failure contributes delta/n_seqs,
        // but the total per-iteration decrement is capped at half the remaining
        // headroom to prevent a single batch from cratering the fraction.
        for (&node, seq_indices) in &node_failures {
            if let Some(f) = fraction[node] {
                let per_failure = delta / n_seqs[node] as f64;
                let raw_decrement = per_failure * seq_indices.len() as f64;
                let headroom = f - config.min_fraction;
                let capped_decrement = raw_decrement.min(headroom * 0.5);
                let new_f = f - capped_decrement;
                if new_f <= config.min_fraction {
                    fraction[node] = None;
                    for &si in seq_indices {
                        incorrect[si] = None;
                    }
                } else {
                    fraction[node] = Some(new_f);
                }
            } else {
                // Node already at None — mark sequences as gave-up
                for &si in seq_indices {
                    incorrect[si] = None;
                }
            }
        }
    }

    // Record problem sequences and groups (lines 422-441)
    let mut problem_sequences: Vec<ProblemSequence> = Vec::new();
    for (i, inc) in incorrect.iter().enumerate() {
        if inc != &Some(false) {
            problem_sequences.push(ProblemSequence {
                index: i + 1, // 1-indexed like R
                expected: format!("Root;{}", classes[i]),
                predicted: predicted[i].clone(),
            });
        }
    }

    let problem_groups: Vec<String> = fraction
        .iter()
        .enumerate()
        .filter(|(_, f)| f.is_none())
        .map(|(i, _)| taxonomy[i].clone())
        .collect();

    // Assemble result (lines 455-471)
    Ok(TrainingSet {
        taxonomy,
        taxa,
        ranks: None,
        levels,
        children,
        parents,
        fraction,
        sequences: sequences_per_node,
        kmers,
        cross_index,
        k,
        idf_weights,
        decision_kmers,
        problem_sequences,
        problem_groups,
        seed_pattern: config.seed_pattern.clone(),
        inverted_index: Some(inverted_index),
    })
}

/// Compute the p-th percentile of nchar values.
fn percentile_nchar(sequences: &[String], p: f64) -> usize {
    let mut lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
    lengths.sort_unstable();
    if lengths.is_empty() {
        return 0;
    }
    let idx = ((lengths.len() as f64 - 1.0) * p) as usize;
    lengths[idx.min(lengths.len() - 1)]
}

/// Sparse profile: sorted by k-mer index (0-based), only non-zero entries.
type SparseProfile = Vec<(usize, f64)>;

/// Merge multiple weighted sparse profiles into a single sparse profile (weighted average).
fn merge_sparse_profiles(profiles: &[SparseProfile], weights: &[f64], total_weight: f64) -> SparseProfile {
    // k-way merge of sorted sparse profiles
    let mut cursors = vec![0usize; profiles.len()];
    let mut result = Vec::new();

    loop {
        // Find the minimum k-mer index across all profiles at their current cursor
        let mut min_key = usize::MAX;
        for (i, profile) in profiles.iter().enumerate() {
            if cursors[i] < profile.len() && profile[cursors[i]].0 < min_key {
                min_key = profile[cursors[i]].0;
            }
        }
        if min_key == usize::MAX {
            break;
        }

        // Sum weighted values for this k-mer across all profiles that have it
        let mut val = 0.0f64;
        for (i, profile) in profiles.iter().enumerate() {
            if cursors[i] < profile.len() && profile[cursors[i]].0 == min_key {
                val += profile[cursors[i]].1 * weights[i];
                cursors[i] += 1;
            }
        }
        result.push((min_key, val / total_weight));
    }
    result
}

/// Precomputed statistics for a profile vector, used to avoid redundant
/// computation in the correlation-aware feature selection hot loop.
#[derive(Clone)]
struct ProfileStats {
    sum: f64,
    /// Precomputed denominator component: sqrt(n * sum_sq - sum * sum)
    denom: f64,
}

impl ProfileStats {
    fn new(v: &[f64]) -> Self {
        let n_f = v.len() as f64;
        let sum: f64 = v.iter().sum();
        let sum_sq: f64 = v.iter().map(|x| x * x).sum();
        let denom = (n_f * sum_sq - sum * sum).sqrt();
        Self { sum, denom }
    }
}

/// Pearson correlation using precomputed statistics for both vectors.
/// Only computes the cross-product sum_ab, which is the only term that
/// changes between different (a, b) pairings.
#[inline]
fn pearson_with_stats(a: &[f64], a_stats: &ProfileStats,
                      b: &[f64], b_stats: &ProfileStats) -> f64 {
    let n_f = a.len() as f64;
    if a.len() < 2 { return 0.0; }
    if a_stats.denom < 1e-15 || b_stats.denom < 1e-15 { return 0.0; }
    let sum_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let num = n_f * sum_ab - a_stats.sum * b_stats.sum;
    (num / (a_stats.denom * b_stats.denom)).abs()
}

/// Absolute Pearson correlation between two profile vectors.
/// Returns 0.0 for constant or zero-length vectors.
#[allow(dead_code)]
fn pearson_abs(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 { return 0.0; }
    let n_f = n as f64;
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    let sum_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let sum_a2: f64 = a.iter().map(|x| x * x).sum();
    let sum_b2: f64 = b.iter().map(|x| x * x).sum();
    let num = n_f * sum_ab - sum_a * sum_b;
    let den_a = (n_f * sum_a2 - sum_a * sum_a).sqrt();
    let den_b = (n_f * sum_b2 - sum_b * sum_b).sqrt();
    if den_a < 1e-15 || den_b < 1e-15 { return 0.0; }
    (num / (den_a * den_b)).abs()
}

/// Recursive tree construction for decision k-mers.
/// Uses sparse profiles to avoid processing 65K dense vectors.
/// Sibling subtrees are processed in parallel via rayon.
/// Port of R/LearnTaxa.R:.createTree (lines 267-319).
fn create_tree(
    node: usize,
    children: &[Vec<usize>],
    sequences: &[Option<Vec<usize>>],
    kmers: &[Vec<i32>],
    n_kmers: usize,
    config: &TrainConfig,
) -> (SparseProfile, usize, Vec<(usize, DecisionNode)>) {
    let child_nodes = &children[node];
    let n_children = child_nodes.len();

    if n_children > 0 && n_children <= config.max_children {
        let mut profiles: Vec<SparseProfile> = Vec::with_capacity(n_children);
        let mut descendants: Vec<usize> = Vec::with_capacity(n_children);
        let mut collected_nodes: Vec<(usize, DecisionNode)> = Vec::new();

        if config.processors > 1 {
            // Process sibling subtrees in parallel (they access disjoint
            // data and all shared params are immutable borrows).
            let child_results: Vec<(SparseProfile, usize, Vec<(usize, DecisionNode)>)> =
                if n_children == 2 {
                    let (r0, r1) = rayon::join(
                        || create_tree(child_nodes[0], children, sequences, kmers, n_kmers, config),
                        || create_tree(child_nodes[1], children, sequences, kmers, n_kmers, config),
                    );
                    vec![r0, r1]
                } else {
                    let mut results: Vec<Option<(SparseProfile, usize, Vec<(usize, DecisionNode)>)>> =
                        (0..n_children).map(|_| None).collect();
                    rayon::scope(|s| {
                        for (i, result_slot) in results.iter_mut().enumerate() {
                            let child = child_nodes[i];
                            s.spawn(move |_| {
                                *result_slot = Some(create_tree(
                                    child, children, sequences, kmers, n_kmers, config,
                                ));
                            });
                        }
                    });
                    results.into_iter().map(|r| r.unwrap()).collect()
                };

            for (profile, desc, nodes) in child_results {
                profiles.push(profile);
                descendants.push(desc);
                collected_nodes.extend(nodes);
            }
        } else {
            // Sequential path: no rayon overhead when processors == 1
            for &child in child_nodes {
                let (profile, desc, nodes) =
                    create_tree(child, children, sequences, kmers, n_kmers, config);
                profiles.push(profile);
                descendants.push(desc);
                collected_nodes.extend(nodes);
            }
        }

        let total_desc: usize = descendants.iter().sum();
        let desc_weights: Vec<f64> = match config.descendant_weighting {
            DescendantWeighting::Count => descendants.iter().map(|&d| d as f64).collect(),
            DescendantWeighting::Equal => vec![1.0; descendants.len()],
            DescendantWeighting::Log => descendants.iter().map(|&d| (1.0 + d as f64).ln()).collect(),
        };

        // Compute weighted average profile q (sparse merge)
        let total_weight: f64 = desc_weights.iter().sum();
        let q = merge_sparse_profiles(&profiles, &desc_weights, total_weight);

        // Build a lookup from k-mer index to q value for fast cross-entropy computation
        let q_map: HashMap<usize, f64> =
            q.iter().map(|&(k, v)| (k, v)).collect();

        // Compute cross-entropy H = -p_i * log(q_i) for each child (only non-zero entries)
        // Each H entry is (kmer_index, entropy_value)
        let h_matrix: Vec<Vec<(usize, f64)>> = profiles
            .iter()
            .map(|p| {
                p.iter()
                    .filter_map(|&(k, pi)| {
                        if let Some(&qi) = q_map.get(&k) {
                            if qi > 0.0 && pi > 0.0 {
                                Some((k, -pi * qi.ln()))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        // Sort each child's H entries by entropy descending
        let mut sorted_h: Vec<Vec<(usize, f64)>> = h_matrix;
        for h in &mut sorted_h {
            h.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Feature selection: choose top k-mers for this decision node
        let record_kmers = {
            let max_nonzero = profiles
                .iter()
                .map(|p| p.len())
                .max()
                .unwrap_or(0);
            ((max_nonzero as f64) * config.record_kmers_fraction).ceil() as usize
        };

        let keep_set: HashSet<usize> = if config.correlation_aware_features {
            // Greedy forward selection: pick k-mers that maximize
            // conditional information gain, penalizing redundancy.
            //
            // For each candidate, gain = base_entropy * (1 - max_corr_with_selected).
            // Profile vectors across children serve as the feature representation.
            //
            // Uses struct-of-arrays with a flat contiguous matrix for cache-friendly
            // access and precomputed Pearson statistics to avoid redundant work.

            // Collect top candidates from each child's entropy ranking.
            let per_child_limit = record_kmers * 2;
            let mut cand_set: HashSet<usize> = HashSet::new();
            for child_h in &sorted_h {
                for &(kmer_idx, _) in child_h.iter().take(per_child_limit) {
                    cand_set.insert(kmer_idx);
                }
            }

            // Build struct-of-arrays: flat row-major profile matrix for cache locality.
            let mut kmer_indices = Vec::with_capacity(cand_set.len());
            let mut entropies = Vec::with_capacity(cand_set.len());
            let mut profiles_flat = Vec::with_capacity(cand_set.len() * n_children);

            for &kmer_idx in &cand_set {
                let mut prof_vec = Vec::with_capacity(n_children);
                for p in profiles.iter() {
                    let val = match p.binary_search_by_key(&kmer_idx, |&(k, _)| k) {
                        Ok(pos) => p[pos].1,
                        Err(_) => 0.0,
                    };
                    prof_vec.push(val);
                }
                let mut max_h = 0.0f64;
                for child_h in &sorted_h {
                    for &(ki, h) in child_h.iter() {
                        if ki == kmer_idx && h > max_h { max_h = h; break; }
                    }
                }
                kmer_indices.push(kmer_idx);
                entropies.push(max_h);
                profiles_flat.extend_from_slice(&prof_vec);
            }

            // Sort all arrays by entropy descending (permutation sort)
            let mut order: Vec<usize> = (0..kmer_indices.len()).collect();
            order.sort_by(|&a, &b| entropies[b].partial_cmp(&entropies[a])
                .unwrap_or(std::cmp::Ordering::Equal));

            let sorted_kmer_indices: Vec<usize> = order.iter().map(|&i| kmer_indices[i]).collect();
            let sorted_entropies: Vec<f64> = order.iter().map(|&i| entropies[i]).collect();
            let mut sorted_profiles_flat = vec![0.0f64; profiles_flat.len()];
            for (new_idx, &old_idx) in order.iter().enumerate() {
                let src = old_idx * n_children;
                let dst = new_idx * n_children;
                sorted_profiles_flat[dst..dst + n_children]
                    .copy_from_slice(&profiles_flat[src..src + n_children]);
            }

            // Precompute Pearson statistics for all candidates
            let cand_stats: Vec<ProfileStats> = (0..order.len()).map(|ci| {
                ProfileStats::new(&sorted_profiles_flat[ci * n_children..(ci + 1) * n_children])
            }).collect();

            let n_cand = sorted_kmer_indices.len();
            let mut is_selected = vec![false; n_cand];
            let mut result_set = HashSet::new();

            // Selected features: contiguous buffer for cache-friendly correlation
            let mut sel_profiles_flat: Vec<f64> = Vec::with_capacity(record_kmers * n_children);
            let mut sel_stats: Vec<ProfileStats> = Vec::with_capacity(record_kmers);
            let mut n_selected: usize = 0;

            for _ in 0..record_kmers {
                let mut best_ci = None;
                let mut best_gain = f64::NEG_INFINITY;

                for ci in 0..n_cand {
                    if is_selected[ci] { continue; }
                    let base_h = sorted_entropies[ci];
                    // Early exit: candidates sorted by entropy descending,
                    // so max possible gain <= base_h. If that can't beat best, stop.
                    if base_h <= best_gain { break; }

                    let cand_prof = &sorted_profiles_flat[ci * n_children..(ci + 1) * n_children];
                    let cand_st = &cand_stats[ci];

                    let mut max_corr: f64 = 0.0;
                    for si in 0..n_selected {
                        let sel_prof = &sel_profiles_flat[si * n_children..(si + 1) * n_children];
                        let sel_st = &sel_stats[si];
                        let corr = pearson_with_stats(cand_prof, cand_st, sel_prof, sel_st);
                        if corr > max_corr { max_corr = corr; }
                        if max_corr >= 1.0 { break; }
                    }

                    let gain = base_h * (1.0 - max_corr);
                    if gain > best_gain {
                        best_gain = gain;
                        best_ci = Some(ci);
                    }
                }

                match best_ci {
                    Some(ci) => {
                        is_selected[ci] = true;
                        result_set.insert(sorted_kmer_indices[ci]);
                        // Append to selected buffers
                        let src = ci * n_children;
                        sel_profiles_flat.extend_from_slice(
                            &sorted_profiles_flat[src..src + n_children]
                        );
                        sel_stats.push(cand_stats[ci].clone());
                        n_selected += 1;
                    }
                    None => break,
                }
            }
            result_set
        } else {
            // Original round-robin selection (lines 290-306)
            let mut keep_set = HashSet::new();
            let mut count = 0usize;
            let mut kmer_idx = 0usize;
            let mut group_idx = 0usize;

            while count < record_kmers {
                group_idx += 1;
                if group_idx > n_children {
                    group_idx = 1;
                    kmer_idx += 1;
                }
                if kmer_idx >= sorted_h[group_idx - 1].len() {
                    if sorted_h.iter().all(|h| kmer_idx >= h.len()) {
                        break;
                    }
                    continue;
                }

                let selected_kmer = sorted_h[group_idx - 1][kmer_idx].0;
                if keep_set.insert(selected_kmer) {
                    count += 1;
                }
            }
            keep_set
        };

        // Convert to sorted keep indices (1-indexed like R)
        let mut keep_vec: Vec<usize> = keep_set.into_iter().collect();
        keep_vec.sort_unstable();
        let keep_indices: Vec<i32> = keep_vec.iter().map(|&k| (k + 1) as i32).collect();

        // Build profile matrix for kept k-mers: rows=subtrees, cols=kept kmers.
        // Both profiles[i] and keep_vec are sorted by k-mer index, so use merge-join
        // instead of building a HashMap per child.
        let selected_profiles: Vec<Vec<f64>> = profiles
            .iter()
            .map(|p| {
                let mut result = Vec::with_capacity(keep_vec.len());
                let mut pi = 0usize;
                for &k in &keep_vec {
                    while pi < p.len() && p[pi].0 < k {
                        pi += 1;
                    }
                    if pi < p.len() && p[pi].0 == k {
                        result.push(p[pi].1);
                    } else {
                        result.push(0.0);
                    }
                }
                result
            })
            .collect();

        collected_nodes.push((node, DecisionNode {
            keep: keep_indices,
            profiles: selected_profiles,
        }));

        (q, total_desc, collected_nodes)
    } else {
        // Leaf node: tabulate k-mers as sparse profile
        let mut counts: HashMap<usize, f64> = HashMap::new();
        if let Some(seq_indices) = &sequences[node] {
            for &si in seq_indices {
                for &km in &kmers[si] {
                    if km > 0 && (km as usize) <= n_kmers {
                        *counts.entry((km - 1) as usize).or_insert(0.0) += 1.0;
                    }
                }
            }
        }
        let total: f64 = counts.values().sum();
        let mut profile: SparseProfile = if total > 0.0 {
            counts.into_iter().map(|(k, v)| (k, v / total)).collect()
        } else {
            Vec::new()
        };
        profile.sort_by_key(|&(k, _)| k);
        (profile, 1, Vec::new())
    }
}
