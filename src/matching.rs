/// Ordered integer membership test: `result[i] = x[i]` in `y`.
/// Both `x` and `y` must be sorted in ascending order.
/// Port of utils.c:intMatch (lines 19-48).
pub fn int_match(x: &[i32], y: &[i32]) -> Vec<bool> {
    let mut result = vec![false; x.len()];
    let mut j = 0usize;
    for (i, &xi) in x.iter().enumerate() {
        while j < y.len() {
            if xi == y[j] {
                result[i] = true;
                break;
            } else if xi < y[j] {
                break;
            }
            j += 1;
        }
    }
    result
}

/// Weighted vector summation across bootstrap replicates.
/// Port of vector_sums.c:vectorSum (lines 22-55).
///
/// `matches`: boolean vector of k-mer matches (query k-mer `i` matched training set)
/// `weights`: IDF weights per k-mer
/// `sampling`: flat matrix of sampled indices (block_count * block_size), 0-indexed
/// `block_count`: number of bootstrap replicates
///
/// R's vectorSum takes 1-indexed sampling and subtracts 1.
/// Our sampling is already 0-indexed.
pub fn vector_sum(
    matches: &[bool],
    weights: &[f64],
    sampling: &[usize],
    block_count: usize,
) -> Vec<f64> {
    let block_size = sampling.len() / block_count;
    let mut result = vec![0.0f64; block_count];

    for i in 0..block_count {
        let mut cur_weight = 0.0;
        let mut max_weight = 0.0;
        for k in 0..block_size {
            let idx = sampling[i * block_size + k];
            max_weight += weights[idx];
            if matches[idx] {
                cur_weight += weights[idx];
            }
        }
        result[i] = if max_weight > 0.0 {
            cur_weight / max_weight
        } else {
            0.0
        };
    }
    result
}

/// Parallel k-mer matching between a query and multiple training sequences.
/// Returns (hits_flat, column_sums).
///
/// Port of vector_sums.c:parallelMatch (lines 59-146).
/// Uses a single contiguous flat array for the hits matrix (matching C's allocMatrix),
/// laid out as [seq0_rep0, seq0_rep1, ..., seq0_repB, seq1_rep0, ...].
///
/// `query_kmers`: sorted unique query k-mers
/// `train_kmers`: all training sequence k-mers (list of sorted int vecs)
/// `indices`: which training sequences to compare (0-indexed)
/// `weights`: IDF weights for query k-mers
/// `block_count`: number of bootstrap replicates
/// `positions`: flat array — for each query k-mer index, the bootstrap positions it maps to
/// `ranges`: cumulative range boundaries (length = query_kmers.len() + 1)
///
/// Returns: `(hits_flat: Vec<f64> [n_indices * block_count], col_sums: Vec<f64> [n_indices])`.
/// Access `hits_flat[seq_idx * block_count + rep]` for sequence `seq_idx`, replicate `rep`.
pub fn parallel_match(
    query_kmers: &[i32],
    train_kmers: &[Vec<i32>],
    indices: &[usize],
    weights: &[f64],
    block_count: usize,
    positions: &[usize],
    ranges: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let n = indices.len();
    let size_x = query_kmers.len();

    // Single contiguous allocation for all hits (matching C's allocMatrix)
    let mut hits_flat = vec![0.0f64; n * block_count];
    let mut col_sums = vec![0.0f64; n];

    // Process each training sequence
    // Use rayon only when there are enough sequences to justify the overhead
    if n > 32 {
        use rayon::prelude::*;
        let results: Vec<(Vec<f64>, f64)> = indices
            .par_iter()
            .map(|&idx| {
                let train_k = &train_kmers[idx];
                let mut hits = vec![0.0f64; block_count];

                let mut j = 0usize;
                for i in 0..size_x {
                    while j < train_k.len() {
                        if query_kmers[i] <= train_k[j] {
                            if query_kmers[i] == train_k[j] {
                                let range_start = ranges[i];
                                let range_end = ranges[i + 1];
                                let w = weights[i];
                                for &pos in &positions[range_start..range_end] {
                                    hits[pos] += w;
                                }
                            }
                            break;
                        }
                        j += 1;
                    }
                }

                let col_sum: f64 = hits.iter().sum();
                (hits, col_sum)
            })
            .collect();

        for (k, (hits, cs)) in results.into_iter().enumerate() {
            let base = k * block_count;
            hits_flat[base..base + block_count].copy_from_slice(&hits);
            col_sums[k] = cs;
        }
    } else {
        // Sequential path: write directly into flat array, no per-sequence allocation
        for (k, &idx) in indices.iter().enumerate() {
            let train_k = &train_kmers[idx];
            let base = k * block_count;

            let mut j = 0usize;
            for i in 0..size_x {
                while j < train_k.len() {
                    if query_kmers[i] <= train_k[j] {
                        if query_kmers[i] == train_k[j] {
                            let range_start = ranges[i];
                            let range_end = ranges[i + 1];
                            let w = weights[i];
                            for &pos in &positions[range_start..range_end] {
                                hits_flat[base + pos] += w;
                            }
                        }
                        break;
                    }
                    j += 1;
                }
            }

            let cs: f64 = hits_flat[base..base + block_count].iter().sum();
            col_sums[k] = cs;
        }
    }

    (hits_flat, col_sums)
}

/// Sum-only merge-join k-mer matching.
///
/// Returns one row sum per requested training sequence without materializing
/// the per-bootstrap hit matrix. Each matched query k-mer contributes its
/// weight once per sampled bootstrap position.
pub fn match_sums(
    query_kmers: &[i32],
    train_kmers: &[Vec<i32>],
    indices: &[usize],
    weights: &[f64],
    _positions: &[usize],
    ranges: &[usize],
) -> Vec<f64> {
    let size_x = query_kmers.len();
    let mut col_sums = vec![0.0f64; indices.len()];

    for (k, &idx) in indices.iter().enumerate() {
        let train_k = &train_kmers[idx];
        let mut j = 0usize;
        let mut sum = 0.0;
        for i in 0..size_x {
            while j < train_k.len() {
                if query_kmers[i] <= train_k[j] {
                    if query_kmers[i] == train_k[j] {
                        sum += weights[i] * (ranges[i + 1] - ranges[i]) as f64;
                    }
                    break;
                }
                j += 1;
            }
        }
        col_sums[k] = sum;
    }

    col_sums
}

/// Merge-join k-mer matching for a compact set of selected rows.
///
/// `selected_positions` are positions into `indices`; output row order follows
/// `selected_positions` exactly.
pub fn match_selected_rows(
    query_kmers: &[i32],
    train_kmers: &[Vec<i32>],
    indices: &[usize],
    selected_positions: &[usize],
    weights: &[f64],
    block_count: usize,
    positions: &[usize],
    ranges: &[usize],
) -> Vec<f64> {
    if selected_positions.is_empty() || block_count == 0 {
        return Vec::new();
    }
    let selected_indices: Vec<usize> = selected_positions.iter().map(|&pos| indices[pos]).collect();
    let (hits_flat, _) = parallel_match(
        query_kmers,
        train_kmers,
        &selected_indices,
        weights,
        block_count,
        positions,
        ranges,
    );
    hits_flat
}

/// Inverted-index based k-mer matching.
/// Instead of merge-joining query against each training sequence,
/// for each query k-mer, look up which training sequences have it.
pub fn parallel_match_inverted(
    query_kmers: &[i32],
    inverted_index: &[Vec<u32>],
    keep: &[usize],
    weights: &[f64],
    block_count: usize,
    positions: &[usize],
    ranges: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let n = keep.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Dense lookup array: O(1) per posting-list entry instead of HashMap.
    // For typical IDTAXA usage (<100K training sequences), this is <400KB.
    let max_idx = keep.iter().copied().max().unwrap_or(0);
    let mut dense_map = vec![u32::MAX; max_idx + 1];
    for (pos, &idx) in keep.iter().enumerate() {
        dense_map[idx] = pos as u32;
    }

    let mut hits_flat = vec![0.0f64; n * block_count];

    for (i, &kmer) in query_kmers.iter().enumerate() {
        if kmer <= 0 || (kmer as usize) > inverted_index.len() {
            continue;
        }
        let posting_list = &inverted_index[(kmer - 1) as usize];
        let w = weights[i];
        let range_start = ranges[i];
        let range_end = ranges[i + 1];
        for &seq_idx in posting_list {
            let si = seq_idx as usize;
            if si < dense_map.len() {
                let keep_pos = dense_map[si];
                if keep_pos != u32::MAX {
                    let base = keep_pos as usize * block_count;
                    for &pos in &positions[range_start..range_end] {
                        hits_flat[base + pos] += w;
                    }
                }
            }
        }
    }

    let mut col_sums = vec![0.0f64; n];
    for (k, sum) in col_sums.iter_mut().enumerate() {
        let base = k * block_count;
        *sum = hits_flat[base..base + block_count].iter().sum();
    }
    (hits_flat, col_sums)
}

/// Sum-only inverted-index k-mer matching.
///
/// Returns one row sum per kept sequence without materializing per-bootstrap
/// rows. This is the first pass used by compact leaf scoring.
pub fn match_sums_inverted(
    query_kmers: &[i32],
    inverted_index: &[Vec<u32>],
    keep: &[usize],
    weights: &[f64],
    _positions: &[usize],
    ranges: &[usize],
) -> Vec<f64> {
    if keep.is_empty() {
        return Vec::new();
    }

    let max_idx = keep.iter().copied().max().unwrap_or(0);
    let mut dense_map = vec![u32::MAX; max_idx + 1];
    for (pos, &idx) in keep.iter().enumerate() {
        dense_map[idx] = pos as u32;
    }

    let mut col_sums = vec![0.0f64; keep.len()];
    for (i, &kmer) in query_kmers.iter().enumerate() {
        if kmer <= 0 || (kmer as usize) > inverted_index.len() {
            continue;
        }
        let posting_list = &inverted_index[(kmer - 1) as usize];
        let contribution = weights[i] * (ranges[i + 1] - ranges[i]) as f64;
        for &seq_idx in posting_list {
            let si = seq_idx as usize;
            if si < dense_map.len() {
                let keep_pos = dense_map[si];
                if keep_pos != u32::MAX {
                    col_sums[keep_pos as usize] += contribution;
                }
            }
        }
    }

    col_sums
}

/// Inverted-index matching for a compact set of selected rows.
///
/// `selected_positions` are positions into `keep`; output row order follows
/// `selected_positions` exactly.
pub fn match_selected_rows_inverted(
    query_kmers: &[i32],
    inverted_index: &[Vec<u32>],
    keep: &[usize],
    selected_positions: &[usize],
    weights: &[f64],
    block_count: usize,
    positions: &[usize],
    ranges: &[usize],
) -> Vec<f64> {
    if selected_positions.is_empty() || block_count == 0 {
        return Vec::new();
    }

    let selected_indices: Vec<usize> = selected_positions.iter().map(|&pos| keep[pos]).collect();
    let max_idx = selected_indices.iter().copied().max().unwrap_or(0);
    let mut dense_map = vec![u32::MAX; max_idx + 1];
    for (row, &idx) in selected_indices.iter().enumerate() {
        dense_map[idx] = row as u32;
    }

    let mut hits_flat = vec![0.0f64; selected_positions.len() * block_count];
    for (i, &kmer) in query_kmers.iter().enumerate() {
        if kmer <= 0 || (kmer as usize) > inverted_index.len() {
            continue;
        }
        let posting_list = &inverted_index[(kmer - 1) as usize];
        let w = weights[i];
        let range_start = ranges[i];
        let range_end = ranges[i + 1];
        for &seq_idx in posting_list {
            let si = seq_idx as usize;
            if si < dense_map.len() {
                let selected_row = dense_map[si];
                if selected_row != u32::MAX {
                    let base = selected_row as usize * block_count;
                    for &pos in &positions[range_start..range_end] {
                        hits_flat[base + pos] += w;
                    }
                }
            }
        }
    }

    hits_flat
}

/// Index of first maximum in `values` for each group.
/// Port of utils.c:groupMax (lines 51-78).
///
/// Returns 0-indexed indices (R version returns 1-indexed).
pub fn group_max(values: &[f64], groups: &[i32], unique_groups: &[i32]) -> Vec<usize> {
    let mut result = vec![0usize; unique_groups.len()];
    let mut curr = 0usize;

    for (i, &ug) in unique_groups.iter().enumerate() {
        let mut max_val = -1e53f64;
        while curr < values.len() && groups[curr] == ug {
            if values[curr] > max_val {
                result[i] = curr;
                max_val = values[curr];
            }
            curr += 1;
        }
    }
    result
}
