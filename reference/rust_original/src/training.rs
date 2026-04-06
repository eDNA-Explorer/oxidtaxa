use std::collections::{HashMap, HashSet};

use crate::alphabet::alphabet_size;
use crate::kmer::{enumerate_sequences, NA_INTEGER};
use crate::matching::{int_match, vector_sum};
use crate::rng::RRng;
use crate::types::{DecisionNode, ProblemSequence, TrainConfig, TrainingSet};

/// Train an IDTAXA classifier.
/// Port of R/LearnTaxa.R (487 lines).
#[allow(clippy::too_many_arguments)]
pub fn learn_taxa(
    sequences: &[String],
    taxonomy_strings: &[String],
    config: &TrainConfig,
    rng: &mut RRng,
    _verbose: bool,
) -> Result<TrainingSet, String> {
    let l = sequences.len();
    if l < 2 {
        return Err("At least two training sequences are required.".to_string());
    }
    if taxonomy_strings.len() != l {
        return Err("taxonomy must be the same length as train.".to_string());
    }

    // DNA-only: fixed alphabet size
    let size: usize = 4;

    // Compute K if not specified (lines 47-75)
    let k = match config.k {
        Some(k) => k,
        None => {
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
    let raw_kmers = enumerate_sequences(sequences, k, false, false, &[], true);
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

    // Build taxonomy tree (lines 155-198)
    let u_classes: Vec<String> = {
        let mut uc_set: HashSet<String> = HashSet::new();
        let mut uc: Vec<String> = Vec::new();
        for c in &classes {
            if uc_set.insert(c.clone()) {
                uc.push(c.clone());
            }
        }
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
    let cross_index: Vec<usize> = classes
        .iter()
        .map(|c| {
            all_taxa
                .iter()
                .position(|t| t == c)
                .map(|p| p + 1) // +1 for Root offset; now 1-indexed in taxonomy
                .unwrap_or(0)
        })
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
    let max_children = config.max_children;
    let record_kmers_fraction = config.record_kmers_fraction;
    let mut decision_kmers: Vec<Option<DecisionNode>> = vec![None; taxonomy.len()];
    create_tree(
        0,
        &children,
        &sequences_per_node,
        &kmers,
        n_kmers,
        max_children,
        record_kmers_fraction,
        &mut decision_kmers,
    );

    // Learn fractions (lines 323-420) — PRNG-critical
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

        for &i in &remaining {
            if kmers[i].is_empty() {
                continue;
            }

            let mut k_node = 0usize; // start at Root
            let mut correct = true;

            loop {
                let subtrees = &children[k_node];
                let dk = &decision_kmers[k_node];

                if dk.is_none() || dk.as_ref().unwrap().keep.is_empty() {
                    break;
                }

                let dk = dk.as_ref().unwrap();
                let n = dk.keep.len();

                if subtrees.len() > 1 {
                    let s = match fraction[k_node] {
                        None => ((n as f64) * config.min_fraction).ceil() as usize,
                        Some(f) => ((n as f64) * f).ceil() as usize,
                    };

                    // *** PRNG CALL — must match R exactly ***
                    let sampling = rng.sample_int_replace(n, s * b);

                    let matches = int_match(&dk.keep, &kmers[i]);
                    let mut hits = vec![vec![0.0f64; b]; subtrees.len()];
                    for (j, _subtree) in subtrees.iter().enumerate() {
                        hits[j] = vector_sum(&matches, &dk.profiles[j], &sampling, b);
                    }

                    // Find max per bootstrap replicate and vote
                    let mut vote_counts = vec![0usize; subtrees.len()];
                    for rep in 0..b {
                        let max_val = hits.iter().map(|h| h[rep]).fold(0.0f64, f64::max);
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

                    if vote_counts[w] < ((b as f64) * 0.8) as usize {
                        break; // less than 80% confidence
                    }

                    // Check classification correctness
                    if !classes[i].starts_with(&end_taxonomy[subtrees[w]]) {
                        correct = false;
                        predicted[i] = taxonomy[subtrees[w]].clone();
                        break;
                    }

                    if children[subtrees[w]].is_empty() {
                        break;
                    }
                    k_node = subtrees[w];
                } else {
                    // Single child
                    if children[subtrees[0]].is_empty() {
                        break;
                    }
                    k_node = subtrees[0];
                }
            }

            if correct {
                incorrect[i] = Some(false);
            } else {
                if fraction[k_node].is_none() {
                    incorrect[i] = None;
                } else {
                    let f = fraction[k_node].unwrap();
                    let new_f = f - delta / n_seqs[k_node] as f64;
                    if new_f <= config.min_fraction {
                        incorrect[i] = None;
                        fraction[k_node] = None;
                    } else {
                        fraction[k_node] = Some(new_f);
                    }
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

    // Compute IDF weights (lines 443-452)
    let class_counts = {
        let mut counts = HashMap::new();
        for c in &classes {
            *counts.entry(c.clone()).or_insert(0usize) += 1;
        }
        counts
    };
    let n_classes = class_counts.len();
    let weights: Vec<f64> = classes
        .iter()
        .map(|c| 1.0 / *class_counts.get(c).unwrap() as f64)
        .collect();

    // Parallel IDF accumulation: partition sequences across threads, merge partial counts
    use rayon::prelude::*;
    let chunk_size = 256;
    let partial_counts: Vec<Vec<f64>> = kmers
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk_kmers)| {
            let mut local = vec![0.0f64; n_kmers];
            let base = chunk_idx * chunk_size;
            for (local_i, class_kmers) in chunk_kmers.iter().enumerate() {
                let w = weights[base + local_i];
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

/// Recursive tree construction for decision k-mers.
/// Uses sparse profiles to avoid processing 65K dense vectors.
/// Port of R/LearnTaxa.R:.createTree (lines 267-319).
fn create_tree(
    node: usize,
    children: &[Vec<usize>],
    sequences: &[Option<Vec<usize>>],
    kmers: &[Vec<i32>],
    n_kmers: usize,
    max_children: usize,
    record_kmers_fraction: f64,
    decision_kmers: &mut Vec<Option<DecisionNode>>,
) -> (SparseProfile, usize) {
    let child_nodes = &children[node];
    let n_children = child_nodes.len();

    if n_children > 0 && n_children <= max_children {
        let mut profiles: Vec<SparseProfile> = Vec::with_capacity(n_children);
        let mut descendants = Vec::with_capacity(n_children);

        for &child in child_nodes {
            let (profile, desc) =
                create_tree(child, children, sequences, kmers, n_kmers, max_children, record_kmers_fraction, decision_kmers);
            profiles.push(profile);
            descendants.push(desc);
        }

        let total_desc: usize = descendants.iter().sum();
        let desc_weights: Vec<f64> = descendants.iter().map(|&d| d as f64).collect();

        // Compute weighted average profile q (sparse merge)
        let q = merge_sparse_profiles(&profiles, &desc_weights, total_desc as f64);

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

        // Round-robin selection of top k-mers (lines 290-306)
        let record_kmers = {
            let max_nonzero = profiles
                .iter()
                .map(|p| p.len())
                .max()
                .unwrap_or(0);
            ((max_nonzero as f64) * record_kmers_fraction).ceil() as usize
        };

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
            // Check if this group has enough sorted entries
            if kmer_idx >= sorted_h[group_idx - 1].len() {
                // Check if all groups are exhausted
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

        // Convert to sorted keep indices (1-indexed like R)
        let mut keep_vec: Vec<usize> = keep_set.into_iter().collect();
        keep_vec.sort_unstable();
        let keep_indices: Vec<i32> = keep_vec.iter().map(|&k| (k + 1) as i32).collect();

        // Build profile matrix for kept k-mers: rows=subtrees, cols=kept kmers
        // For each child profile, look up the kept k-mers
        let selected_profiles: Vec<Vec<f64>> = profiles
            .iter()
            .map(|p| {
                let p_map: HashMap<usize, f64> =
                    p.iter().map(|&(k, v)| (k, v)).collect();
                keep_vec.iter().map(|&k| *p_map.get(&k).unwrap_or(&0.0)).collect()
            })
            .collect();

        decision_kmers[node] = Some(DecisionNode {
            keep: keep_indices,
            profiles: selected_profiles,
        });

        (q, total_desc)
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
        (profile, 1)
    }
}
