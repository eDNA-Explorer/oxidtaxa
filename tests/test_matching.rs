mod common;

use common::load_json;
use oxidtaxa::matching::{
    int_match, match_selected_rows, match_selected_rows_inverted, match_sums, match_sums_inverted,
    parallel_match, parallel_match_inverted,
};
use std::collections::HashMap;

/// Load intMatch test cases from golden JSON.
#[derive(serde::Deserialize)]
struct IntMatchCase {
    x: Vec<i32>,
    y: Vec<i32>,
}

#[test]
fn test_int_match_basic() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden_basic: Vec<bool> = load_json("s07_im_basic");

    let case = cases.get("basic").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden_basic, "basic intMatch mismatch");
}

#[test]
fn test_int_match_no_overlap() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_no_overlap");

    let case = cases.get("no_overlap").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "no_overlap intMatch mismatch");
}

#[test]
fn test_int_match_all_match() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_all_match");

    let case = cases.get("all_match").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "all_match intMatch mismatch");
}

#[test]
fn test_int_match_empty_x() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_empty_x");

    let case = cases.get("empty_x").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "empty_x intMatch mismatch");
}

#[test]
fn test_int_match_empty_y() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_empty_y");

    let case = cases.get("empty_y").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "empty_y intMatch mismatch");
}

#[test]
fn test_int_match_single() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_single");

    let case = cases.get("single").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "single intMatch mismatch");
}

#[test]
fn test_int_match_large() {
    let cases: HashMap<String, IntMatchCase> = load_json("s07_im_cases");
    let golden: Vec<bool> = load_json("s07_im_large");

    let case = cases.get("large").unwrap();
    let result = int_match(&case.x, &case.y);
    assert_eq!(result, golden, "large intMatch mismatch");
}

fn build_inverted_index(train_kmers: &[Vec<i32>], n_kmers: usize) -> Vec<Vec<u32>> {
    let mut inverted = vec![Vec::new(); n_kmers];
    for (seq_idx, kmers) in train_kmers.iter().enumerate() {
        for &kmer in kmers {
            if kmer > 0 && (kmer as usize) <= n_kmers {
                inverted[(kmer - 1) as usize].push(seq_idx as u32);
            }
        }
    }
    inverted
}

#[test]
fn test_parallel_match_inverted_matches_merge_path() {
    let train_kmers = vec![
        vec![1, 3, 5, 7],
        vec![2, 3, 8],
        vec![1, 4, 9],
        vec![3, 4, 5, 10],
        vec![6, 7],
        vec![1, 8, 10],
    ];
    let inverted = build_inverted_index(&train_kmers, 10);
    let keep = vec![5, 0, 3, 1];
    let query_kmers = vec![0, 1, 3, 5, 8, 11];
    let weights = vec![100.0, 1.5, 2.0, 0.25, 3.0, 50.0];
    let block_count = 4;
    let positions = vec![0, 2, 2, 3, 1, 1, 3, 0, 2];
    let ranges = vec![0, 1, 3, 4, 6, 8, 9];

    let (merge_hits, merge_sums) = parallel_match(
        &query_kmers,
        &train_kmers,
        &keep,
        &weights,
        block_count,
        &positions,
        &ranges,
    );
    let (inverted_hits, inverted_sums) = parallel_match_inverted(
        &query_kmers,
        &inverted,
        &keep,
        &weights,
        block_count,
        &positions,
        &ranges,
    );

    assert_eq!(inverted_hits, merge_hits);
    assert_eq!(inverted_sums, merge_sums);
}

#[test]
fn test_match_sums_helpers_match_dense_row_sums() {
    let train_kmers = vec![vec![1, 2, 4], vec![2, 5], vec![1, 3, 5], vec![4, 6]];
    let inverted = build_inverted_index(&train_kmers, 6);
    let keep = vec![3, 0, 2];
    let query_kmers = vec![1, 2, 4, 6, 20];
    let weights = vec![0.5, 1.25, 2.0, 4.0, 100.0];
    let block_count = 3;
    let positions = vec![0, 1, 1, 2, 0, 0, 2];
    let ranges = vec![0, 2, 3, 5, 7, 7];

    let (dense_hits, dense_sums) = parallel_match(
        &query_kmers,
        &train_kmers,
        &keep,
        &weights,
        block_count,
        &positions,
        &ranges,
    );
    let row_sums: Vec<f64> = dense_hits
        .chunks(block_count)
        .map(|row| row.iter().sum())
        .collect();
    let merge_sums = match_sums(
        &query_kmers,
        &train_kmers,
        &keep,
        &weights,
        &positions,
        &ranges,
    );
    let inverted_sums = match_sums_inverted(
        &query_kmers,
        &inverted,
        &keep,
        &weights,
        &positions,
        &ranges,
    );

    assert_eq!(dense_sums, row_sums);
    assert_eq!(merge_sums, row_sums);
    assert_eq!(inverted_sums, row_sums);
}

#[test]
fn test_selected_row_helpers_match_dense_selected_rows() {
    let train_kmers = vec![
        vec![1, 3],
        vec![2, 4],
        vec![1, 2, 5],
        vec![3, 5],
        vec![4, 5],
    ];
    let inverted = build_inverted_index(&train_kmers, 5);
    let keep = vec![4, 1, 3, 0, 2];
    let selected_positions = vec![2, 4, 0];
    let query_kmers = vec![1, 3, 4, 5];
    let weights = vec![1.0, 2.0, 3.0, 4.0];
    let block_count = 5;
    let positions = vec![0, 4, 1, 1, 3, 2, 0, 4];
    let ranges = vec![0, 2, 5, 6, 8];

    let (dense_hits, _) = parallel_match(
        &query_kmers,
        &train_kmers,
        &keep,
        &weights,
        block_count,
        &positions,
        &ranges,
    );
    let expected: Vec<f64> = selected_positions
        .iter()
        .flat_map(|&pos| dense_hits[pos * block_count..(pos + 1) * block_count].iter())
        .copied()
        .collect();
    let merge_selected = match_selected_rows(
        &query_kmers,
        &train_kmers,
        &keep,
        &selected_positions,
        &weights,
        block_count,
        &positions,
        &ranges,
    );
    let inverted_selected = match_selected_rows_inverted(
        &query_kmers,
        &inverted,
        &keep,
        &selected_positions,
        &weights,
        block_count,
        &positions,
        &ranges,
    );

    assert_eq!(merge_selected, expected);
    assert_eq!(inverted_selected, expected);
}

#[test]
fn test_inverted_match_empty_keep() {
    let train_kmers = vec![vec![1, 2], vec![2, 3]];
    let inverted = build_inverted_index(&train_kmers, 3);
    let query_kmers = vec![1, 2];
    let weights = vec![1.0, 2.0];
    let positions = vec![0, 1];
    let ranges = vec![0, 1, 2];

    let (hits, sums) = parallel_match_inverted(
        &query_kmers,
        &inverted,
        &[],
        &weights,
        2,
        &positions,
        &ranges,
    );
    assert!(hits.is_empty());
    assert!(sums.is_empty());
    assert!(
        match_sums_inverted(&query_kmers, &inverted, &[], &weights, &positions, &ranges).is_empty()
    );
    assert!(match_selected_rows_inverted(
        &query_kmers,
        &inverted,
        &[],
        &[],
        &weights,
        2,
        &positions,
        &ranges
    )
    .is_empty());
}
