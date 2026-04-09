mod common;

use common::load_json;
use oxidtaxa::kmer::{enumerate_sequences, NA_INTEGER};

/// Helper: load golden k-mers (list of f64 arrays with -2147483648 sentinel for NA)
fn load_golden_kmers(name: &str) -> Vec<Vec<i32>> {
    let raw: Vec<Vec<f64>> = load_json(name);
    raw.into_iter()
        .map(|v| v.into_iter().map(|x| x as i32).collect())
        .collect()
}

#[test]
fn test_kmer_standard_sequences() {
    let seqs: Vec<String> = load_json("s02a_std_seqs");
    let golden = load_golden_kmers("s02a_std_kmers");

    let result = enumerate_sequences(&seqs, 8, false, false, &[], true, None);

    assert_eq!(result.len(), golden.len(), "sequence count mismatch");
    for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            got.len(),
            expected.len(),
            "kmer count mismatch for seq {}: got {} expected {}",
            i,
            got.len(),
            expected.len()
        );
        assert_eq!(got, expected, "kmer values mismatch for seq {}", i);
    }
}

#[test]
fn test_kmer_short_sequences() {
    let seqs: Vec<String> = load_json("s02b_short_seqs");
    let golden = load_golden_kmers("s02b_short_kmers");

    let result = enumerate_sequences(&seqs, 8, false, false, &[], true, None);

    assert_eq!(result.len(), golden.len());

    // Specific expectations from run_golden.R
    assert_eq!(result[0].len(), 0, "len=0 should yield empty"); // ""
    assert_eq!(result[1].len(), 0, "len=1 should yield empty"); // "A"
    assert_eq!(result[2].len(), 0, "len=3 should yield empty"); // "ACG"
    assert_eq!(result[3].len(), 0, "len=7 should yield empty"); // "ACGTACG"
    assert_eq!(result[4].len(), 1, "len=8 should yield 1 k-mer"); // "ACGTACGT"
    assert_eq!(result[5].len(), 2, "len=9 should yield 2 k-mers"); // "ACGTACGTA"

    for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(got, expected, "kmer values mismatch for short seq {}", i);
    }
}

#[test]
fn test_kmer_ambiguous_bases() {
    let seqs: Vec<String> = load_json("s02c_ambig_seqs");
    let golden = load_golden_kmers("s02c_ambig_kmers");

    let result = enumerate_sequences(&seqs, 8, false, false, &[], true, None);

    // All-N sequence should have all NAs
    assert!(
        result[0].iter().all(|&v| v == NA_INTEGER),
        "all-N should yield all NA k-mers"
    );

    for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(got, expected, "kmer values mismatch for ambig seq {}", i);
    }
}

#[test]
fn test_kmer_repeats() {
    let seqs: Vec<String> = load_json("s02d_repeat_seqs");
    let golden = load_golden_kmers("s02d_repeat_kmers");

    let result = enumerate_sequences(&seqs, 8, false, false, &[], true, None);

    // Poly-A should have all same k-mer value
    let poly_a_valid: Vec<i32> = result[0]
        .iter()
        .filter(|&&v| v != NA_INTEGER)
        .copied()
        .collect();
    if !poly_a_valid.is_empty() {
        let unique: std::collections::HashSet<i32> = poly_a_valid.iter().copied().collect();
        assert_eq!(
            unique.len(),
            1,
            "poly-A k-mers should all have same value"
        );
    }

    for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(got, expected, "kmer values mismatch for repeat seq {}", i);
    }
}

#[test]
fn test_kmer_different_k_values() {
    // Single sequence auto-unboxed to bare string in JSON
    let seq: String = load_json("s02e_ktest_seq");
    let seqs = vec![seq];

    for k_val in [1, 3, 5, 8, 10, 13, 15] {
        let golden = load_golden_kmers(&format!("s02e_kmers_K{}", k_val));
        let result = enumerate_sequences(&seqs, k_val, false, false, &[], true, None);

        assert_eq!(
            result.len(),
            golden.len(),
            "K={}: sequence count mismatch",
            k_val
        );
        for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
            assert_eq!(
                got, expected,
                "K={}: kmer values mismatch for seq {}",
                k_val, i
            );
        }
    }
}

#[test]
fn test_kmer_masking_none() {
    let seqs: Vec<String> = load_json("s02f_mask_seqs");
    let golden = load_golden_kmers("s02f_kmers_nomask");

    let result = enumerate_sequences(&seqs, 8, false, false, &[], true, None);
    for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(got, expected, "no-mask kmer values mismatch for seq {}", i);
    }
}

#[test]
fn test_kmer_masking_repeats() {
    let seqs: Vec<String> = load_json("s02f_mask_seqs");
    let golden = load_golden_kmers("s02f_kmers_maskrep");

    let result = enumerate_sequences(&seqs, 8, true, false, &[], true, None);
    for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            got, expected,
            "repeat-mask kmer values mismatch for seq {}",
            i
        );
    }
}

#[test]
fn test_kmer_masking_lcr() {
    let seqs: Vec<String> = load_json("s02f_mask_seqs");
    let golden = load_golden_kmers("s02f_kmers_masklcr");

    let result = enumerate_sequences(&seqs, 8, false, true, &[], true, None);
    for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            got, expected,
            "LCR-mask kmer values mismatch for seq {}",
            i
        );
    }
}

#[test]
fn test_kmer_masking_both() {
    let seqs: Vec<String> = load_json("s02f_mask_seqs");
    let golden = load_golden_kmers("s02f_kmers_maskboth");

    let result = enumerate_sequences(&seqs, 8, true, true, &[], true, None);
    for (i, (got, expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            got, expected,
            "both-mask kmer values mismatch for seq {}",
            i
        );
    }
}
