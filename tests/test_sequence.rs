mod common;

use common::load_json;
use oxidtaxa::sequence::{remove_gaps, reverse_complement, vcount_pattern};

#[test]
fn test_reverse_complement_matches_golden() {
    let golden_seqs: Vec<String> = load_json("s05_rc_seqs");
    let golden_result: Vec<String> = load_json("s05_rc_result");

    for (i, seq) in golden_seqs.iter().enumerate() {
        let got = reverse_complement(seq);
        assert_eq!(
            got, golden_result[i],
            "reverseComplement mismatch for seq {}",
            i
        );
    }
}

#[test]
fn test_remove_gaps_matches_golden() {
    let golden_seqs: Vec<String> = load_json("s04_gap_seqs");
    let golden_result: Vec<String> = load_json("s04_gap_removed");

    let result = remove_gaps(&golden_seqs);

    for (i, (got, expected)) in result.iter().zip(golden_result.iter()).enumerate() {
        assert_eq!(got, expected, "removeGaps mismatch for seq {}", i);
    }
}

#[test]
fn test_vcount_pattern_n() {
    let seqs: Vec<String> = load_json("s06_vcp_seqs");
    let golden: Vec<i64> = load_json("s06_vcp_N");

    let result = vcount_pattern(b'N', &seqs);
    for (i, (&got, &expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            got, expected as usize,
            "vcountPattern N mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_vcount_pattern_a() {
    let seqs: Vec<String> = load_json("s06_vcp_seqs");
    let golden: Vec<i64> = load_json("s06_vcp_A");

    let result = vcount_pattern(b'A', &seqs);
    for (i, (&got, &expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            got, expected as usize,
            "vcountPattern A mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_vcount_pattern_dash() {
    let seqs: Vec<String> = load_json("s06_vcp_seqs");
    let golden: Vec<i64> = load_json("s06_vcp_-");

    let result = vcount_pattern(b'-', &seqs);
    for (i, (&got, &expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            got, expected as usize,
            "vcountPattern - mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_vcount_pattern_dot() {
    let seqs: Vec<String> = load_json("s06_vcp_seqs");
    let golden: Vec<i64> = load_json("s06_vcp_.");

    let result = vcount_pattern(b'.', &seqs);
    for (i, (&got, &expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            got, expected as usize,
            "vcountPattern . mismatch at index {}",
            i
        );
    }
}
