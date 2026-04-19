use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use rayon::prelude::*;

use crate::alphabet::alphabet_size;
use crate::kmer::{enumerate_sequences, parse_seed_pattern, SpacedSeed, NA_INTEGER};
use crate::matching::{int_match, vector_sum};
use crate::rng::{mix_seed, RRng};
use crate::types::{
    BuildTreeConfig, BuiltTree, DecisionNode, DescendantWeighting, LearnFractionsConfig,
    PreparedData, ProblemSequence, TrainConfig, TrainingSet,
};

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

/// Phase 1: Enumerate k-mers, build taxonomy tree, compute IDF weights.
///
/// Depends only on (sequences, taxonomy, k, n, seed_pattern).
pub fn prepare_data(
    sequences: &[String],
    taxonomy_strings: &[String],
    k: Option<usize>,
    n: f64,
    seed_pattern: Option<String>,
    processors: usize,
) -> Result<PreparedData, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processors)
        .build()
        .map_err(|e| format!("failed to create rayon thread pool: {e}"))?;
    pool.install(|| _prepare_data_inner(sequences, taxonomy_strings, k, n, seed_pattern))
}

/// Phase 2: Build decision tree with feature selection at each node.
///
/// This is the most expensive phase when correlation_aware_features=true.
pub fn build_tree(
    prepared: &PreparedData,
    config: &BuildTreeConfig,
) -> Result<BuiltTree, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.processors)
        .build()
        .map_err(|e| format!("failed to create rayon thread pool: {e}"))?;
    pool.install(|| _build_tree_inner(prepared, config))
}

/// Phase 3: Iterative fraction-learning loop + model assembly.
///
/// The cheapest phase — re-run this when only training_threshold,
/// use_idf_in_training, or leave_one_out changes.
pub fn learn_fractions(
    prepared: &PreparedData,
    built_tree: &BuiltTree,
    config: &LearnFractionsConfig,
    seed: u32,
) -> Result<TrainingSet, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.processors)
        .build()
        .map_err(|e| format!("failed to create rayon thread pool: {e}"))?;
    pool.install(|| _learn_fractions_inner(prepared, built_tree, config, seed))
}

fn _prepare_data_inner(
    sequences: &[String],
    taxonomy_strings: &[String],
    k_param: Option<usize>,
    n: f64,
    seed_pattern: Option<String>,
) -> Result<PreparedData, String> {
    let l = sequences.len();
    if l < 2 {
        return Err("At least two training sequences are required.".to_string());
    }
    if taxonomy_strings.len() != l {
        return Err("taxonomy must be the same length as train.".to_string());
    }

    // DNA-only: fixed alphabet size
    let size: usize = 4;

    // Parse spaced seed if provided
    let spaced_seed: Option<SpacedSeed> = match &seed_pattern {
        Some(pat) => Some(parse_seed_pattern(pat)?),
        None => None,
    };

    // Compute K if not specified
    let k = match (&spaced_seed, k_param) {
        (Some(seed), _) => seed.weight,
        (None, Some(k)) => k,
        (None, None) => {
            let quant = percentile_nchar(sequences, 0.99);
            let as_val = alphabet_size(sequences);
            let computed = ((n * quant as f64).ln() / as_val.ln()).floor() as usize;
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

    // Parse taxonomy: strip "Root;" prefix, normalize
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

    // Enumerate k-mers
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

    // Build taxonomy tree
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

    let mut taxonomy: Vec<String> = vec!["Root;".to_string()];
    for t in &all_taxa {
        taxonomy.push(format!("Root;{}", t));
    }

    let taxa_to_idx: HashMap<&str, usize> = all_taxa
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i + 1))
        .collect();
    let cross_index: Vec<usize> = classes
        .iter()
        .map(|c| *taxa_to_idx.get(c.as_str()).unwrap_or(&0))
        .collect();

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

    let max_level = *levels.iter().max().unwrap_or(&1) - 1;
    let mut levs: Vec<Vec<usize>> = vec![Vec::new(); max_level as usize];
    for (i, &lev) in levels.iter().enumerate() {
        let idx = lev - 2;
        if idx >= 0 && (idx as usize) < levs.len() {
            levs[idx as usize].push(i);
        }
    }

    let mut starts = vec![0usize; max_level as usize];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); taxonomy.len()];
    for i in 0..taxonomy.len() {
        let j = levels[i] - 1;
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

    let end_taxonomy: Vec<String> = taxonomy
        .iter()
        .map(|t| {
            if t.len() > 5 {
                t[5..].to_string()
            } else {
                String::new()
            }
        })
        .collect();

    let mut et_to_idx: HashMap<&str, usize> = HashMap::with_capacity(end_taxonomy.len());
    for (i, et) in end_taxonomy.iter().enumerate() {
        et_to_idx.insert(et.as_str(), i);
    }

    let mut sequences_per_node: Vec<Option<Vec<usize>>> = vec![None; taxonomy.len()];
    sequences_per_node[0] = Some((0..classes.len()).collect());
    for (seq_idx, class) in classes.iter().enumerate() {
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

    // Per-rank IDF: one row per taxonomic depth. Used by fraction-learning
    // descent (when `use_idf_in_training = true`) AND at classify-time leaf
    // phase so the two score against the same IDF. The deepest row is the
    // species-level equivalent of the single IDF vector R IDTAXA produced.
    let idf_weights_by_rank = compute_idf_by_rank(&classes, &kmers, n_kmers);

    // Seq hash precompute (moved before create_tree — independent of tree output)
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

    Ok(PreparedData {
        k,
        n_kmers,
        kmers,
        inverted_index,
        classes,
        taxonomy,
        taxa,
        levels,
        children,
        parents,
        end_taxonomy,
        sequences_per_node,
        n_seqs,
        cross_index,
        idf_weights_by_rank,
        seq_hashes,
        seed_pattern,
    })
}

fn _build_tree_inner(
    prepared: &PreparedData,
    config: &BuildTreeConfig,
) -> Result<BuiltTree, String> {
    let train_config = TrainConfig {
        record_kmers_fraction: config.record_kmers_fraction,
        descendant_weighting: config.descendant_weighting,
        correlation_aware_features: config.correlation_aware_features,
        max_children: config.max_children,
        processors: config.processors,
        ..Default::default()
    };

    let mut decision_kmers: Vec<Option<DecisionNode>> = vec![None; prepared.taxonomy.len()];
    let (_root_profile, _root_desc, nodes) = create_tree(
        0,
        &prepared.children,
        &prepared.sequences_per_node,
        &prepared.kmers,
        prepared.n_kmers,
        &train_config,
    );
    for (idx, dk) in nodes {
        decision_kmers[idx] = Some(dk);
    }

    Ok(BuiltTree { decision_kmers })
}

fn _learn_fractions_inner(
    prepared: &PreparedData,
    built_tree: &BuiltTree,
    config: &LearnFractionsConfig,
    seed: u32,
) -> Result<TrainingSet, String> {
    let l = prepared.kmers.len();
    let b: usize = 100;

    let mut fraction: Vec<Option<f64>> = vec![Some(config.max_fraction); prepared.taxonomy.len()];
    let mut incorrect: Vec<Option<bool>> = vec![Some(true); l];
    let mut predicted = vec![String::new(); l];
    let delta = (config.max_fraction - config.min_fraction) * config.multiplier;

    let idf_by_rank: &[Vec<f64>] = &prepared.idf_weights_by_rank;

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

        let fraction_snapshot = fraction.clone();

        let results: Vec<(usize, bool, usize, String)> = remaining
            .par_iter()
            .map(|&i| {
                if prepared.kmers[i].is_empty() {
                    return (i, true, 0, String::new());
                }

                let mut seq_rng =
                    RRng::new(mix_seed(seed, (_it as u64) * 1_000_000 + prepared.seq_hashes[i]));
                let mut k_node = 0usize;
                let mut correct = true;
                let mut pred = String::new();

                loop {
                    let subtrees = &prepared.children[k_node];
                    let dk = &built_tree.decision_kmers[k_node];

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

                        let matches = int_match(&dk.keep, &prepared.kmers[i]);

                        let loo_child_idx = if config.leave_one_out {
                            subtrees.iter().enumerate().find(|(_, &st)| {
                                prepared.classes[i].starts_with(&prepared.end_taxonomy[st])
                            }).map(|(j, &st)| {
                                let group_size = prepared.n_seqs[st];
                                (j, group_size)
                            })
                        } else {
                            None
                        };

                        // Pick the per-rank IDF row matching the descent
                        // node's depth. Matches classify-time semantics so
                        // `use_idf_in_training = true` scores training and
                        // classification against the same IDF.
                        let idf_row: &[f64] = {
                            let depth = (prepared.levels[k_node] - 1).max(0) as usize;
                            let row_idx = depth.min(idf_by_rank.len().saturating_sub(1));
                            &idf_by_rank[row_idx]
                        };

                        let mut hits = vec![vec![0.0f64; b]; subtrees.len()];
                        for (j, _subtree) in subtrees.iter().enumerate() {
                            let mut weights_j: Vec<f64> = if config.use_idf_in_training {
                                dk.profiles[j].iter().zip(dk.keep.iter())
                                    .map(|(&prof, &km)| {
                                        let idf = if km > 0 && (km as usize) <= idf_row.len() {
                                            idf_row[(km - 1) as usize]
                                        } else { 0.0 };
                                        prof * idf
                                    })
                                    .collect()
                            } else {
                                dk.profiles[j].clone()
                            };

                            if let Some((loo_j, group_size)) = loo_child_idx {
                                if j == loo_j && group_size > 1 && group_size <= 5 {
                                    let scale = (group_size - 1) as f64 / group_size as f64;
                                    for w in &mut weights_j { *w *= scale; }
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

                        if !prepared.classes[i].starts_with(&prepared.end_taxonomy[subtrees[w]]) {
                            correct = false;
                            pred = prepared.taxonomy[subtrees[w]].clone();
                            break;
                        }

                        if prepared.children[subtrees[w]].is_empty() {
                            break;
                        }
                        k_node = subtrees[w];
                    } else {
                        if prepared.children[subtrees[0]].is_empty() {
                            break;
                        }
                        k_node = subtrees[0];
                    }
                }

                (i, correct, k_node, pred)
            })
            .collect();

        let mut node_failures: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(seq_idx, correct, fail_node, ref pred) in &results {
            if correct {
                incorrect[seq_idx] = Some(false);
            } else {
                predicted[seq_idx] = pred.clone();
                node_failures.entry(fail_node).or_default().push(seq_idx);
            }
        }

        for (&node, seq_indices) in &node_failures {
            if let Some(f) = fraction[node] {
                let per_failure = delta / prepared.n_seqs[node] as f64;
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
                for &si in seq_indices {
                    incorrect[si] = None;
                }
            }
        }
    }

    let mut problem_sequences: Vec<ProblemSequence> = Vec::new();
    for (i, inc) in incorrect.iter().enumerate() {
        if inc != &Some(false) {
            problem_sequences.push(ProblemSequence {
                index: i + 1,
                expected: format!("Root;{}", prepared.classes[i]),
                predicted: predicted[i].clone(),
            });
        }
    }

    let problem_groups: Vec<String> = fraction
        .iter()
        .enumerate()
        .filter(|(_, f)| f.is_none())
        .map(|(i, _)| prepared.taxonomy[i].clone())
        .collect();

    Ok(TrainingSet {
        taxonomy: prepared.taxonomy.clone(),
        taxa: prepared.taxa.clone(),
        ranks: None,
        levels: prepared.levels.clone(),
        children: prepared.children.clone(),
        parents: prepared.parents.clone(),
        fraction,
        sequences: prepared.sequences_per_node.clone(),
        kmers: prepared.kmers.clone(),
        cross_index: prepared.cross_index.clone(),
        k: prepared.k,
        decision_kmers: built_tree.decision_kmers.clone(),
        problem_sequences,
        problem_groups,
        seed_pattern: prepared.seed_pattern.clone(),
        inverted_index: Some(prepared.inverted_index.clone()),
        idf_weights_by_rank: prepared.idf_weights_by_rank.clone(),
    })
}

fn _learn_taxa_inner(
    sequences: &[String],
    taxonomy_strings: &[String],
    config: &TrainConfig,
    seed: u32,
) -> Result<TrainingSet, String> {
    let prepared = _prepare_data_inner(
        sequences, taxonomy_strings,
        config.k, config.n, config.seed_pattern.clone(),
    )?;
    let built_tree = _build_tree_inner(&prepared, &BuildTreeConfig::from(config))?;
    _learn_fractions_inner(&prepared, &built_tree, &LearnFractionsConfig::from(config), seed)
}

/// Compute per-seq prefix strings at depth `rank` (1 = Kingdom-level group,
/// 2 = Phylum-level, ...). If a class has fewer than `rank` components, the
/// full class string is used (no truncation).
fn prefixes_at_rank(classes: &[String], rank: usize) -> Vec<String> {
    classes
        .iter()
        .map(|c| {
            let parts: Vec<&str> = c.split(';').filter(|s| !s.is_empty()).collect();
            let r = rank.min(parts.len());
            parts[..r]
                .iter()
                .map(|s| format!("{};", s))
                .collect::<String>()
        })
        .collect()
}

/// IDF vector computed against groupings at a given rank depth.
/// Same formula as the global IDF but with `n_classes` replaced by the number
/// of distinct prefixes at that rank, and per-seq weights normalized by
/// rank-group size.
fn compute_idf_at_rank(
    classes: &[String],
    kmers: &[Vec<i32>],
    n_kmers: usize,
    rank: usize,
) -> Vec<f64> {
    let prefixes = prefixes_at_rank(classes, rank);
    let mut prefix_counts: HashMap<String, usize> = HashMap::new();
    for p in &prefixes {
        *prefix_counts.entry(p.clone()).or_insert(0) += 1;
    }
    let n_classes = prefix_counts.len();
    let idf_seq_weights: Vec<f64> = prefixes
        .iter()
        .map(|p| 1.0 / *prefix_counts.get(p).unwrap_or(&1) as f64)
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
    idf_counts
        .iter()
        .map(|&c| (n_classes as f64 / (1.0 + c)).ln())
        .collect()
}

/// Compute a per-rank IDF matrix: row `r-1` corresponds to depth `r`
/// (Kingdom-level = row 0, Phylum-level = row 1, ..., Species-level = last).
fn compute_idf_by_rank(
    classes: &[String],
    kmers: &[Vec<i32>],
    n_kmers: usize,
) -> Vec<Vec<f64>> {
    let max_rank = classes
        .iter()
        .map(|c| c.split(';').filter(|s| !s.is_empty()).count())
        .max()
        .unwrap_or(0);
    (1..=max_rank)
        .into_par_iter()
        .map(|r| compute_idf_at_rank(classes, kmers, n_kmers, r))
        .collect()
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

/// Precomputed √(p̃) values for a profile, where p̃ = p / Σp is the L1
/// normalization. Pairwise Bhattacharyya coefficient is then a simple dot
/// product of two √(p̃) vectors.
///
/// Bhattacharyya is the only redundancy metric oxidtaxa supports for
/// correlation-aware feature selection. Absolute Pearson correlation was
/// considered but rejected: it degenerates at `n_children = 2` (|r| = 1
/// for any two 2D points, collapsing the selection back to round-robin)
/// and isn't well-defined on probability-distribution-like profiles.
#[derive(Clone)]
struct BhattacharyyaStats {
    sqrt_profile: Vec<f64>,
    /// True when the profile sums to 0 (all-zero entry); pairwise BC with any
    /// candidate is 0 and the feature carries no signal.
    is_empty: bool,
}

impl BhattacharyyaStats {
    fn new(v: &[f64]) -> Self {
        let sum: f64 = v.iter().sum();
        if sum <= 0.0 {
            return Self {
                sqrt_profile: vec![0.0; v.len()],
                is_empty: true,
            };
        }
        let sqrt_profile: Vec<f64> = v.iter().map(|&x| (x / sum).max(0.0).sqrt()).collect();
        Self {
            sqrt_profile,
            is_empty: false,
        }
    }
}

/// Bhattacharyya coefficient on L1-normalized sqrt profiles.
/// Range [0, 1]: 1 when the profiles agree after L1-normalization, 0 when
/// disjoint. Equivalent to 1 − H²(p, q)/2 with H the Hellinger distance.
#[inline]
fn bhattacharyya_with_stats(a_stats: &BhattacharyyaStats, b_stats: &BhattacharyyaStats) -> f64 {
    if a_stats.is_empty || b_stats.is_empty {
        return 0.0;
    }
    a_stats
        .sqrt_profile
        .iter()
        .zip(b_stats.sqrt_profile.iter())
        .map(|(a, b)| a * b)
        .sum()
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
            // Sort candidates for deterministic iteration order. Without this,
            // HashSet iteration randomness leaks into tie-breaking of feature
            // selection, making the output non-reproducible across runs.
            let mut cand_sorted: Vec<usize> = cand_set.into_iter().collect();
            cand_sorted.sort_unstable();

            // Build struct-of-arrays: flat row-major profile matrix for cache locality.
            let mut kmer_indices = Vec::with_capacity(cand_sorted.len());
            let mut entropies = Vec::with_capacity(cand_sorted.len());
            let mut profiles_flat = Vec::with_capacity(cand_sorted.len() * n_children);

            for &kmer_idx in &cand_sorted {
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

            // Precompute Bhattacharyya stats for all candidates. This is the
            // only supported redundancy metric: Pearson was removed because
            // it degenerates at n_children = 2 and isn't well-defined on
            // profile-like data.
            let cand_bc_stats: Vec<BhattacharyyaStats> = (0..order.len()).map(|ci| {
                BhattacharyyaStats::new(
                    &sorted_profiles_flat[ci * n_children..(ci + 1) * n_children],
                )
            }).collect();

            let n_cand = sorted_kmer_indices.len();
            let mut is_selected = vec![false; n_cand];
            let mut result_set = HashSet::new();

            // Per-candidate running-max correlation against all selected features so far.
            // Updated incrementally: each outer iteration adds exactly one correlation
            // computation per not-selected candidate (against the newly-selected one),
            // reducing total correlation work from O(R^2 * C) to O(R * C).
            let mut max_corr: Vec<f64> = vec![0.0; n_cand];

            // Phase 3: parallelize argmax + update when candidate pool is large
            // enough to amortize rayon overhead. For small pools, sequential is
            // strictly faster due to zero parallelism overhead and the
            // entropy-descending early-exit optimization.
            const PAR_THRESHOLD: usize = 2048;
            let use_par = n_cand >= PAR_THRESHOLD && config.processors > 1;

            for _ in 0..record_kmers {
                // 1. Argmax over not-selected candidates using cached max_corr.
                let best_ci = if use_par {
                    // Parallel: deterministic reduction. Tie-break on lower index
                    // so the output matches the sequential version (which picks
                    // the first ci at a given gain).
                    let (bc, _bg) = (0..n_cand)
                        .into_par_iter()
                        .filter(|&ci| !is_selected[ci])
                        .map(|ci| {
                            let gain =
                                sorted_entropies[ci] * (1.0 - max_corr[ci]);
                            (ci, gain)
                        })
                        .reduce(
                            || (usize::MAX, f64::NEG_INFINITY),
                            |a, b| {
                                if b.1 > a.1 { b }
                                else if b.1 < a.1 { a }
                                else if b.0 < a.0 { b }
                                else { a }
                            },
                        );
                    if bc == usize::MAX { None } else { Some(bc) }
                } else {
                    // Sequential: retains the entropy-descending early exit,
                    // which cannot be cheaply expressed in a parallel reduction.
                    let mut best_ci = None;
                    let mut best_gain = f64::NEG_INFINITY;
                    for ci in 0..n_cand {
                        if is_selected[ci] { continue; }
                        let base_h = sorted_entropies[ci];
                        if base_h <= best_gain { break; }
                        let gain = base_h * (1.0 - max_corr[ci]);
                        if gain > best_gain {
                            best_gain = gain;
                            best_ci = Some(ci);
                        }
                    }
                    best_ci
                };

                let ci = match best_ci { Some(ci) => ci, None => break };

                // 2. Commit selection.
                is_selected[ci] = true;
                result_set.insert(sorted_kmer_indices[ci]);

                // 3. Incremental max_corr update against the newly-selected
                //    Bhattacharyya feature. Skip candidates already selected
                //    or saturated at 1.0.
                let new_bc = cand_bc_stats[ci].clone();
                if use_par {
                    max_corr
                        .par_iter_mut()
                        .enumerate()
                        .zip(cand_bc_stats.par_iter())
                        .for_each(|((cj, max_c), cj_st)| {
                            if is_selected[cj] || *max_c >= 1.0 { return; }
                            let corr = bhattacharyya_with_stats(cj_st, &new_bc);
                            if corr > *max_c { *max_c = corr; }
                        });
                } else {
                    for cj in 0..n_cand {
                        if is_selected[cj] { continue; }
                        if max_corr[cj] >= 1.0 { continue; }
                        let corr = bhattacharyya_with_stats(
                            &cand_bc_stats[cj], &new_bc,
                        );
                        if corr > max_corr[cj] { max_corr[cj] = corr; }
                    }
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
