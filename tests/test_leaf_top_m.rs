//! Phase 3 (F) — leaf-phase top-M aggregation.
//!
//! Two layers of coverage:
//!
//! 1. Pure-unit tests on the extracted `aggregate_top_m_at_rep` function.
//!    Deterministic by construction; no bootstrap.
//!
//! 2. Integration tests on a small training set with multi-ref groups.
//!    Lock in the M=1 bit-identical contract and the M>1 behavior changes.

mod common;

use oxidtaxa::classify::{aggregate_top_m_at_rep, id_taxa};
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{ClassifyConfig, LeafAggregationMode, OutputType, StrandMode, TrainConfig};

// ---------- 1) Pure-unit aggregator tests ----------

/// Layout the test scratchpad in `hits_flat` row-major: ref `r` at rep `rep`
/// is `r * b + rep`. Two refs (rows), three reps (cols):
///   r0:  10.0   2.0   4.0
///   r1:   6.0   8.0   1.0
fn small_hits_flat() -> Vec<f64> {
    vec![10.0, 2.0, 4.0, 6.0, 8.0, 1.0]
}

#[test]
fn test_aggregate_mean_two_refs() {
    let hits = small_hits_flat();
    let refs = vec![0usize, 1];
    // Mean at rep 0: (10 + 6) / 2 = 8.0
    let v0 = aggregate_top_m_at_rep(&refs, 0, 3, &hits, LeafAggregationMode::Mean);
    assert!((v0 - 8.0).abs() < 1e-12);
    // Mean at rep 1: (2 + 8) / 2 = 5.0
    let v1 = aggregate_top_m_at_rep(&refs, 1, 3, &hits, LeafAggregationMode::Mean);
    assert!((v1 - 5.0).abs() < 1e-12);
}

#[test]
fn test_aggregate_per_rep_best_two_refs() {
    let hits = small_hits_flat();
    let refs = vec![0usize, 1];
    // PerRepBest at rep 0: max(10, 6) = 10.0
    let v0 = aggregate_top_m_at_rep(&refs, 0, 3, &hits, LeafAggregationMode::PerRepBest);
    assert!((v0 - 10.0).abs() < 1e-12);
    // PerRepBest at rep 1: max(2, 8) = 8.0   <-- different ref wins this rep
    let v1 = aggregate_top_m_at_rep(&refs, 1, 3, &hits, LeafAggregationMode::PerRepBest);
    assert!((v1 - 8.0).abs() < 1e-12);
}

#[test]
fn test_aggregate_singleton_is_identity_in_every_mode() {
    let hits = small_hits_flat();
    let refs = vec![0usize];
    for mode in [
        LeafAggregationMode::Mean,
        LeafAggregationMode::TrimmedMean,
        LeafAggregationMode::PerRepBest,
    ] {
        let v0 = aggregate_top_m_at_rep(&refs, 0, 3, &hits, mode);
        assert!((v0 - 10.0).abs() < 1e-12, "mode {:?} at rep 0", mode);
        let v1 = aggregate_top_m_at_rep(&refs, 1, 3, &hits, mode);
        assert!((v1 - 2.0).abs() < 1e-12, "mode {:?} at rep 1", mode);
    }
}

#[test]
fn test_aggregate_trimmed_mean_falls_back_below_three_refs() {
    let hits = small_hits_flat();
    let refs = vec![0usize, 1];
    // M_eff = 2 < 3 ⇒ TrimmedMean degrades to Mean.
    let v0_trim = aggregate_top_m_at_rep(&refs, 0, 3, &hits, LeafAggregationMode::TrimmedMean);
    let v0_mean = aggregate_top_m_at_rep(&refs, 0, 3, &hits, LeafAggregationMode::Mean);
    assert!((v0_trim - v0_mean).abs() < 1e-12);
}

#[test]
fn test_aggregate_trimmed_mean_three_or_more_refs() {
    // 3 refs, 1 rep:
    //   r0: 10.0
    //   r1:  6.0
    //   r2:  2.0
    // Sorted ascending: [2.0, 6.0, 10.0]. Trim ends → [6.0]. Mean = 6.0.
    let hits = vec![10.0, 6.0, 2.0];
    let refs = vec![0usize, 1, 2];
    let v = aggregate_top_m_at_rep(&refs, 0, 1, &hits, LeafAggregationMode::TrimmedMean);
    assert!((v - 6.0).abs() < 1e-12);
}

#[test]
fn test_aggregate_trimmed_mean_five_refs() {
    // 5 refs, 1 rep, values [1, 2, 3, 4, 5]. Sorted asc: [1, 2, 3, 4, 5].
    // Trim ends → [2, 3, 4]. Mean = 3.0.
    let hits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let refs = vec![0usize, 1, 2, 3, 4];
    let v = aggregate_top_m_at_rep(&refs, 0, 1, &hits, LeafAggregationMode::TrimmedMean);
    assert!((v - 3.0).abs() < 1e-12);
}

#[test]
fn test_aggregate_with_negative_values() {
    // Negative values can arise when IDF weights make hits_flat values
    // negative. Aggregation must remain mathematically defined.
    let hits = vec![-5.0, -1.0, 3.0];
    let refs = vec![0usize, 1, 2];
    // Mean: (-5 + -1 + 3) / 3 = -1.0
    let v = aggregate_top_m_at_rep(&refs, 0, 1, &hits, LeafAggregationMode::Mean);
    assert!((v - (-1.0)).abs() < 1e-12);
    // PerRepBest: max(-5, -1, 3) = 3.0
    let v = aggregate_top_m_at_rep(&refs, 0, 1, &hits, LeafAggregationMode::PerRepBest);
    assert!((v - 3.0).abs() < 1e-12);
    // TrimmedMean (n=3): drop min and max → keep median = -1.0
    let v = aggregate_top_m_at_rep(&refs, 0, 1, &hits, LeafAggregationMode::TrimmedMean);
    assert!((v - (-1.0)).abs() < 1e-12);
}

// ---------- 2) Integration tests on a real model ----------

/// Build a training set with three groups of varying ref-density:
///   Canis_lupus      → 1 ref (singleton)
///   Vulpes_vulpes    → 2 refs (low-density)
///   Felis_catus      → 5 refs (high-density, with intentional sequence variation)
fn build_multi_ref_training_set() -> oxidtaxa::types::TrainingSet {
    let canis_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                     GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
                     TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA";
    let vulpes_seq_a = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                       GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                       GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    // Single-base variant of vulpes_seq_a — same group, distinct ref.
    let vulpes_seq_b = "TGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                       GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                       GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    // Five Felis variants — all under same species, but with a few base
    // differences so they're distinguishable training entries.
    let felis_seq_a = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
                      CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                      AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT";
    let felis_seq_b = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
                      CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                      AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAACTTT";
    let felis_seq_c = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
                      CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                      AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAACTTTT";
    let felis_seq_d = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
                      CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                      AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTCAAAATTTT";
    let felis_seq_e = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
                      CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                      AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTCTAAAATTTT";

    let sequences = vec![
        canis_seq.to_string(),
        vulpes_seq_a.to_string(),
        vulpes_seq_b.to_string(),
        felis_seq_a.to_string(),
        felis_seq_b.to_string(),
        felis_seq_c.to_string(),
        felis_seq_d.to_string(),
        felis_seq_e.to_string(),
    ];
    let taxonomy = vec![
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

/// At `leaf_top_m=1`, every aggregation mode must produce identical output to
/// the pre-Phase-3 path. Cross-check by running with `Mean`, `TrimmedMean`,
/// `PerRepBest` at M=1 and asserting they all match.
#[test]
fn test_default_m1_matches_current_leaf_behavior() {
    let ts = build_multi_ref_training_set();

    let queries = vec![
        "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
         CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
         AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT"
            .to_string(),
    ];
    let names = vec!["q1".to_string()];

    let baseline = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig::default(),
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    for mode in [
        LeafAggregationMode::Mean,
        LeafAggregationMode::TrimmedMean,
        LeafAggregationMode::PerRepBest,
    ] {
        let r = id_taxa(
            &queries,
            &names,
            &ts,
            &ClassifyConfig {
                leaf_top_m: 1,
                leaf_aggregation_mode: mode,
                ..Default::default()
            },
            StrandMode::Top,
            OutputType::Extended,
            42,
            true,
        );
        assert_eq!(
            baseline[0].taxon, r[0].taxon,
            "M=1 with mode {:?} must produce identical taxon",
            mode
        );
        assert_eq!(
            baseline[0].confidence, r[0].confidence,
            "M=1 with mode {:?} must produce identical confidence",
            mode
        );
        assert_eq!(
            baseline[0].similarity, r[0].similarity,
            "M=1 with mode {:?} must produce identical similarity",
            mode
        );
    }
}

/// At `leaf_top_m > 1` on a multi-ref training set, the classifier produces
/// a well-formed result with finite similarity and PerRepBest >= Mean at the
/// per-replicate level. Validates the M>1 plumbing end-to-end without
/// requiring a specific numeric inequality (which is hard to engineer on
/// synthetic data — the synthetic kmer overlap can collapse score
/// differences, but the unit tests on `aggregate_top_m_at_rep` already lock
/// in the math).
#[test]
fn test_m_gt_1_produces_well_formed_result() {
    let ts = build_multi_ref_training_set();

    let queries = vec![
        "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"
            .to_string(),
    ];
    let names = vec!["q1".to_string()];

    let m2_mean = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig {
            leaf_top_m: 2,
            leaf_aggregation_mode: LeafAggregationMode::Mean,
            ..Default::default()
        },
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let m2_best = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig {
            leaf_top_m: 2,
            leaf_aggregation_mode: LeafAggregationMode::PerRepBest,
            ..Default::default()
        },
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let m5_mean = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig {
            leaf_top_m: 5,
            leaf_aggregation_mode: LeafAggregationMode::Mean,
            ..Default::default()
        },
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    // All variants produce finite similarity and at least one rank.
    for (label, r) in [
        ("m2_mean", &m2_mean[0]),
        ("m2_best", &m2_best[0]),
        ("m5_mean", &m5_mean[0]),
    ] {
        assert!(
            r.similarity.is_finite(),
            "{}: similarity not finite ({})",
            label,
            r.similarity
        );
        assert!(
            !r.taxon.is_empty(),
            "{}: taxon Vec must be non-empty (Root present)",
            label
        );
    }

    // PerRepBest is per-replicate >= Mean by construction (max >= mean), so
    // the summed-across-reps similarity satisfies the same inequality at the
    // group level whenever both runs hit the same selected group.
    if m2_best[0].taxon == m2_mean[0].taxon {
        assert!(
            m2_best[0].similarity >= m2_mean[0].similarity - 1e-9,
            "PerRepBest similarity ({}) should be >= Mean similarity ({}) \
             on the same selected group",
            m2_best[0].similarity,
            m2_mean[0].similarity
        );
    }
}

/// At `leaf_top_m=10` on a 2-ref Vulpes group, M_eff caps at 2. No crash;
/// classification still terminates. Exercises the M_eff = min(M, n_refs)
/// graceful-degradation contract on sparse groups.
#[test]
fn test_m_exceeds_group_size_uses_meff_cap() {
    let ts = build_multi_ref_training_set();

    // Vulpes-flavored query.
    let queries = vec![
        "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"
            .to_string(),
    ];
    let names = vec!["q1".to_string()];

    // Should not panic.
    let r = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig {
            leaf_top_m: 10,
            leaf_aggregation_mode: LeafAggregationMode::Mean,
            ..Default::default()
        },
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    assert_eq!(r.len(), 1);
    // Classified — the call survived M_eff capping at 2.
    assert!(r[0].reject_reason.is_none());
    // Fixture invariant: query must land in Vulpes_vulpes for the M_eff
    // assertion to mean anything. If the fixture or sampling shifts the
    // landed group elsewhere, the test should fail loudly rather than
    // silently passing on an unintended path.
    assert!(
        r[0].taxon.last().map_or(false, |t| t == "Vulpes_vulpes"),
        "fixture invariant broken: query must land in Vulpes_vulpes (got {:?})",
        r[0].taxon,
    );
}

/// At `leaf_top_m=5, mode=TrimmedMean` on a 2-ref Vulpes group, M_eff=2 < 3
/// triggers fall-back to Mean. Result must equal what we'd get with Mean
/// directly at the same M.
#[test]
fn test_trimmed_mean_falls_back_to_mean_below_three_refs() {
    let ts = build_multi_ref_training_set();

    let queries = vec![
        "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"
            .to_string(),
    ];
    let names = vec!["q1".to_string()];

    let trimmed = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig {
            leaf_top_m: 5,
            leaf_aggregation_mode: LeafAggregationMode::TrimmedMean,
            ..Default::default()
        },
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let mean = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig {
            leaf_top_m: 5,
            leaf_aggregation_mode: LeafAggregationMode::Mean,
            ..Default::default()
        },
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    // Fixture invariant: both runs must land in the same group for the
    // fall-back equality to be meaningful.
    assert!(
        trimmed[0]
            .taxon
            .last()
            .map_or(false, |t| t == "Vulpes_vulpes"),
        "fixture invariant broken: TrimmedMean run must land in \
         Vulpes_vulpes (got {:?})",
        trimmed[0].taxon,
    );
    assert!(
        mean[0].taxon.last().map_or(false, |t| t == "Vulpes_vulpes"),
        "fixture invariant broken: Mean run must land in Vulpes_vulpes \
         (got {:?})",
        mean[0].taxon,
    );
    assert_eq!(trimmed[0].taxon, mean[0].taxon);
    assert_eq!(trimmed[0].confidence, mean[0].confidence);
    assert_eq!(trimmed[0].similarity, mean[0].similarity);
}

/// Singleton group: a query that lands in the Canis_lupus group (1 ref).
/// M=5 Mean is identity-on-1 and must equal M=1 output.
#[test]
fn test_singleton_group_collapses_to_single_ref() {
    let ts = build_multi_ref_training_set();

    let queries = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA"
            .to_string(),
    ];
    let names = vec!["q1".to_string()];

    let m1 = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig::default(),
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let m5 = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig {
            leaf_top_m: 5,
            leaf_aggregation_mode: LeafAggregationMode::Mean,
            ..Default::default()
        },
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    // The selected group is Canis_lupus (1 ref). Its tot_hits at M=5 Mean
    // equals tot_hits at M=1 because mean-of-1 is identity. The query may
    // pick a different group if tot_hits is dominated by an off-target group,
    // but on this query Canis-flavored query should land in Canis_lupus.
    // Fixture invariant: assert this loudly so a fixture drift surfaces as
    // a clear failure instead of a silently-vacuous equality.
    assert!(
        m1[0].taxon.last().map_or(false, |t| t == "Canis_lupus"),
        "fixture invariant broken: M=1 query must land in Canis_lupus \
         (got {:?})",
        m1[0].taxon,
    );
    assert_eq!(
        m1[0].taxon, m5[0].taxon,
        "singleton group must produce same taxon at M=1 and M=5"
    );
    assert_eq!(
        m1[0].confidence, m5[0].confidence,
        "singleton group must produce same confidence at M=1 and M=5"
    );
}
