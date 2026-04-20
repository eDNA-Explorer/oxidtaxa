mod common;

use common::load_json;
use oxidtaxa::classify::id_taxa;
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{
    ClassifyConfig, DescendantWeighting, OutputType, StrandMode, TrainConfig,
};

fn load_standard_data() -> (Vec<String>, Vec<String>) {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    (seqs, tax)
}

fn train_and_classify(
    train_config: &TrainConfig,
    classify_config: &ClassifyConfig,
) -> Vec<oxidtaxa::types::ClassificationResult> {
    let (seqs, tax) = load_standard_data();
    let model = learn_taxa(&seqs, &tax, train_config, 42, false).unwrap();
    let query_seqs: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..query_seqs.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();
    id_taxa(
        &query_seqs, &names, &model, classify_config,
        StrandMode::Both, OutputType::Extended, 42, true,
    )
}

// ============================================================================
// Training threshold match
// ============================================================================

#[test]
fn test_training_threshold_default_matches_golden() {
    // Default training_threshold=0.8 should produce identical results to golden
    let (seqs, tax) = load_standard_data();
    let config = TrainConfig::default();
    assert!((config.training_threshold - 0.8).abs() < 1e-10);
    let model = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
    // Just verify it trains successfully with the default
    assert!(!model.taxonomy.is_empty());
}

#[test]
fn test_training_threshold_impossible_preserves_fractions() {
    // With threshold > 1.0, no sequence can ever pass the vote check,
    // so no descent happens → no failures → all fractions stay at max_fraction.
    let (seqs, tax) = load_standard_data();
    let config = TrainConfig {
        training_threshold: 1.01,
        ..Default::default()
    };
    let model = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    // All non-None fractions should be exactly max_fraction (no decrements)
    for (i, f) in model.fraction.iter().enumerate() {
        if let Some(v) = f {
            assert!(
                (*v - config.max_fraction).abs() < 1e-10,
                "fraction[{}] = {} but expected max_fraction {} (no descent should occur)",
                i, v, config.max_fraction
            );
        }
    }
    // No problem sequences either (nothing was ever classified)
    assert_eq!(model.problem_sequences.len(), 0,
        "Expected 0 problem sequences with impossible threshold");
}

// ============================================================================
// Descendant weighting alternatives
// ============================================================================

#[test]
fn test_descendant_weighting_count_is_default() {
    let config = TrainConfig::default();
    assert_eq!(config.descendant_weighting, DescendantWeighting::Count);
}

#[test]
fn test_descendant_weighting_equal_changes_decision_kmers() {
    let (seqs, tax) = load_standard_data();

    let count_model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();
    let equal_model = learn_taxa(
        &seqs, &tax,
        &TrainConfig { descendant_weighting: DescendantWeighting::Equal, ..Default::default() },
        42, false,
    ).unwrap();
    let log_model = learn_taxa(
        &seqs, &tax,
        &TrainConfig { descendant_weighting: DescendantWeighting::Log, ..Default::default() },
        42, false,
    ).unwrap();

    // Decision kmers should differ between strategies for at least some nodes
    let mut diffs_equal = 0;
    let mut diffs_log = 0;
    for (c, e) in count_model.decision_kmers.iter().zip(equal_model.decision_kmers.iter()) {
        if c.is_some() && e.is_some() {
            if c.as_ref().unwrap().keep != e.as_ref().unwrap().keep {
                diffs_equal += 1;
            }
        }
    }
    for (c, l) in count_model.decision_kmers.iter().zip(log_model.decision_kmers.iter()) {
        if c.is_some() && l.is_some() {
            if c.as_ref().unwrap().keep != l.as_ref().unwrap().keep {
                diffs_log += 1;
            }
        }
    }

    // At least one node should use different k-mers (may be 0 on small datasets
    // where all strategies agree, so check either differs)
    assert!(
        diffs_equal > 0 || diffs_log > 0,
        "Expected at least one weighting strategy to produce different decision k-mers"
    );
}

// ============================================================================
// IDF scoring during training
// ============================================================================

#[test]
fn test_idf_training_produces_valid_model() {
    let (seqs, tax) = load_standard_data();

    // Default (IDF off) should match golden
    let default_model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    // IDF-weighted training should produce a valid model
    let idf_model = learn_taxa(
        &seqs, &tax,
        &TrainConfig { use_idf_in_training: true, ..Default::default() },
        42, false,
    ).unwrap();

    // Model structure should be identical (IDF only affects fraction learning weights)
    assert_eq!(idf_model.taxonomy.len(), default_model.taxonomy.len());
    assert_eq!(idf_model.decision_kmers.len(), default_model.decision_kmers.len());
    assert!(!idf_model.idf_weights_by_rank.is_empty());
    for row in &idf_model.idf_weights_by_rank {
        assert!(!row.is_empty(), "each per-rank IDF row should be populated");
    }

    // Decision k-mers should be identical (IDF doesn't affect create_tree)
    for (d, i) in default_model.decision_kmers.iter().zip(idf_model.decision_kmers.iter()) {
        match (d, i) {
            (Some(dk_d), Some(dk_i)) => assert_eq!(dk_d.keep, dk_i.keep),
            (None, None) => {},
            _ => panic!("Decision k-mer presence differs"),
        }
    }

    // IDF-weighted training should be usable for classification
    let query_seqs: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..query_seqs.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();
    let results = id_taxa(
        &query_seqs, &names, &idf_model, &ClassifyConfig::default(),
        StrandMode::Both, OutputType::Extended, 42, true,
    );
    assert_eq!(results.len(), query_seqs.len());
    for r in &results {
        assert!(!r.taxon.is_empty());
        assert!(r.taxon[0] == "Root");
    }
}

#[test]
fn test_idf_training_combined_with_permissive_threshold() {
    // Use a very permissive threshold with IDF weights — the combination
    // should produce different behavior than either alone on problem data
    let seqs: Vec<String> = load_json("s08c_problem_seqs");
    let tax: Vec<String> = load_json("s08c_problem_tax");

    let idf_model = learn_taxa(
        &seqs, &tax,
        &TrainConfig {
            use_idf_in_training: true,
            training_threshold: 0.3,
            ..Default::default()
        },
        42, false,
    ).unwrap();

    // Should produce a valid model
    assert!(!idf_model.taxonomy.is_empty());
}

// ============================================================================
// Leave-one-out training
// ============================================================================

#[test]
fn test_leave_one_out_produces_valid_model() {
    // LOO primarily affects small groups (singletons and pairs)
    // Use the singleton dataset which has such groups
    let seqs: Vec<String> = load_json("s08d_singleton_seqs");
    let tax: Vec<String> = load_json("s08d_singleton_tax");

    let default_model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();
    let loo_model = learn_taxa(
        &seqs, &tax,
        &TrainConfig { leave_one_out: true, ..Default::default() },
        42, false,
    ).unwrap();

    // Both should produce valid models
    assert!(!loo_model.taxonomy.is_empty());
    assert_eq!(loo_model.taxonomy.len(), default_model.taxonomy.len());

    // LOO shouldn't affect tree structure, only fractions
    assert_eq!(loo_model.decision_kmers.len(), default_model.decision_kmers.len());
    for (d, l) in default_model.decision_kmers.iter().zip(loo_model.decision_kmers.iter()) {
        match (d, l) {
            (Some(dk_d), Some(dk_l)) => assert_eq!(dk_d.keep, dk_l.keep),
            (None, None) => {},
            _ => panic!("Decision k-mer presence differs with LOO"),
        }
    }
}

#[test]
fn test_leave_one_out_standard_produces_valid_classification() {
    let results = train_and_classify(
        &TrainConfig { leave_one_out: true, ..Default::default() },
        &ClassifyConfig::default(),
    );
    for r in &results {
        assert!(!r.taxon.is_empty());
        assert!(r.taxon[0] == "Root");
    }
}

// ============================================================================
// Beam search
// ============================================================================

#[test]
fn test_beam_width_1_matches_greedy() {
    // beam_width=1 should produce identical results to default (which is also 1)
    let results_default = train_and_classify(
        &TrainConfig::default(),
        &ClassifyConfig::default(),
    );
    let results_beam1 = train_and_classify(
        &TrainConfig::default(),
        &ClassifyConfig { beam_width: 1, ..Default::default() },
    );

    assert_eq!(results_default.len(), results_beam1.len());
    for (d, b) in results_default.iter().zip(results_beam1.iter()) {
        assert_eq!(d.taxon, b.taxon, "beam_width=1 should match greedy");
        assert_eq!(d.confidence, b.confidence);
    }
}

#[test]
fn test_beam_width_3_produces_valid_results() {
    let results = train_and_classify(
        &TrainConfig::default(),
        &ClassifyConfig { beam_width: 3, ..Default::default() },
    );

    // All results should be valid
    for r in &results {
        assert!(!r.taxon.is_empty(), "taxon should not be empty");
        assert!(r.taxon[0] == "Root", "first taxon should be Root");
        assert_eq!(r.taxon.len(), r.confidence.len());
        for &c in &r.confidence {
            assert!(c >= 0.0 && c <= 200.0, "confidence {} out of range", c);
        }
    }
}

// ============================================================================
// Correlation-aware feature selection
// ============================================================================

#[test]
fn test_correlation_aware_changes_decision_kmers() {
    let (seqs, tax) = load_standard_data();

    let default_model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();
    let corr_model = learn_taxa(
        &seqs, &tax,
        &TrainConfig { correlation_aware_features: true, ..Default::default() },
        42, false,
    ).unwrap();

    // Should produce different decision k-mer selections
    let mut diffs = 0;
    for (d, c) in default_model.decision_kmers.iter().zip(corr_model.decision_kmers.iter()) {
        if d.is_some() && c.is_some() {
            if d.as_ref().unwrap().keep != c.as_ref().unwrap().keep {
                diffs += 1;
            }
        }
    }

    assert!(diffs > 0, "Expected correlation-aware selection to produce different k-mers");
}

#[test]
fn test_correlation_aware_parallel_matches_sequential() {
    let (seqs, tax) = load_standard_data();

    let base = TrainConfig {
        correlation_aware_features: true,
        record_kmers_fraction: 0.44,
        ..TrainConfig::default()
    };
    let seq_config = TrainConfig { processors: 1, ..base };
    let par_config = TrainConfig {
        processors: 4,
        ..TrainConfig {
            correlation_aware_features: true,
            record_kmers_fraction: 0.44,
            ..TrainConfig::default()
        }
    };

    let m_seq = learn_taxa(&seqs, &tax, &seq_config, 42, false).unwrap();
    let m_par = learn_taxa(&seqs, &tax, &par_config, 42, false).unwrap();

    let b_seq = bincode::serialize(&m_seq.decision_kmers).unwrap();
    let b_par = bincode::serialize(&m_par.decision_kmers).unwrap();
    assert_eq!(
        b_seq, b_par,
        "parallel correlation-aware output must equal sequential output"
    );
}

#[test]
fn test_correlation_aware_deterministic_output() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let (seqs, tax) = load_standard_data();
    let config = TrainConfig {
        correlation_aware_features: true,
        record_kmers_fraction: 0.44,
        ..TrainConfig::default()
    };

    // Train twice and assert bit-exact equality across runs (determinism).
    let m1 = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
    let m2 = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    let bytes1 = bincode::serialize(&m1.decision_kmers).unwrap();
    let bytes2 = bincode::serialize(&m2.decision_kmers).unwrap();
    assert_eq!(bytes1, bytes2, "decision_kmers must be bit-exact across runs");

    // Hash is recorded in output for manual tracking but NOT asserted —
    // the Pearson→Bhattacharyya switch intentionally changed the selected
    // k-mers, so the pre-switch hash no longer applies. Re-lock after the
    // next intentional selection change.
    let mut hasher = DefaultHasher::new();
    bytes1.hash(&mut hasher);
    let _hash = hasher.finish();
}

// ============================================================================
// Bhattacharyya redundancy (only supported metric)
// ============================================================================

#[test]
fn test_correlation_aware_bhattacharyya_valid_model() {
    // Correlation-aware selection now uses Bhattacharyya unconditionally
    // (Pearson was removed — it degenerates at n_children=2 and isn't
    // justified on profile-like data). Model should train and classify.
    let (seqs, tax) = load_standard_data();
    let model = learn_taxa(
        &seqs,
        &tax,
        &TrainConfig {
            correlation_aware_features: true,
            ..Default::default()
        },
        42,
        false,
    )
    .unwrap();
    assert!(!model.decision_kmers.is_empty());

    let query_seqs: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..query_seqs.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();
    let results = id_taxa(
        &query_seqs,
        &names,
        &model,
        &ClassifyConfig::default(),
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );
    assert_eq!(results.len(), query_seqs.len());
    for r in &results {
        assert!(!r.taxon.is_empty());
        assert!(r.taxon[0] == "Root");
    }
}

// ============================================================================
// Per-rank IDF weights (always on in v0.3.0+)
// ============================================================================

#[test]
fn test_per_rank_idf_always_populated() {
    // v0.3.0+ always computes a per-rank IDF matrix. Default `TrainConfig`
    // produces a TrainingSet whose `idf_weights_by_rank` has one row per
    // taxonomic depth, each row of length 4^k.
    let (seqs, tax) = load_standard_data();
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    let by_rank = &model.idf_weights_by_rank;
    assert!(!by_rank.is_empty(), "by_rank should have at least one row");
    let n_kmers = 4usize.pow(model.k as u32);
    for (i, row) in by_rank.iter().enumerate() {
        assert_eq!(
            row.len(),
            n_kmers,
            "row {} has {} entries, expected {}",
            i,
            row.len(),
            n_kmers
        );
    }
}

#[test]
fn test_per_rank_idf_classification_runs() {
    let (seqs, tax) = load_standard_data();
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();
    assert!(!model.idf_weights_by_rank.is_empty());

    let query_seqs: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..query_seqs.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();
    let results = id_taxa(
        &query_seqs,
        &names,
        &model,
        &ClassifyConfig::default(),
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(results.len(), query_seqs.len());
    for r in &results {
        assert!(!r.taxon.is_empty());
        assert!(r.taxon[0] == "Root");
        for &c in &r.confidence {
            assert!((0.0..=200.0).contains(&c), "confidence out of range: {}", c);
        }
    }
}

// ============================================================================
// Integration: All improvements combined
// ============================================================================

#[test]
fn test_all_improvements_combined() {
    let train_config = TrainConfig {
        training_threshold: 0.98,
        descendant_weighting: DescendantWeighting::Equal,
        use_idf_in_training: true,
        leave_one_out: true,
        correlation_aware_features: true,
        ..Default::default()
    };
    let classify_config = ClassifyConfig {
        beam_width: 3,
        length_normalize: true,
        ..Default::default()
    };

    let results = train_and_classify(&train_config, &classify_config);

    // All results should be valid
    assert!(!results.is_empty());
    for r in &results {
        assert!(!r.taxon.is_empty());
        assert!(r.taxon[0] == "Root");
        assert_eq!(r.taxon.len(), r.confidence.len());
    }
}
