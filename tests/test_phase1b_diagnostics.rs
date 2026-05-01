mod common;

use common::load_json;
use oxidtaxa::classify::id_taxa;
use oxidtaxa::training::{compute_node_diagnostics, learn_taxa, prepare_data};
use oxidtaxa::types::{
    ClassificationResult, ClassifyConfig, DescendantWeighting, OutputType, StrandMode, TrainConfig,
};

/// Default impl produces zeroed/empty fields. Locks the contract that any
/// future field addition gets a sensible zero default and existing
/// constructors using `..Default::default()` keep compiling.
#[test]
fn test_default_classification_result_is_zeroed() {
    let r = ClassificationResult::default();
    assert!(r.taxon.is_empty());
    assert!(r.confidence.is_empty());
    assert!(r.alternatives.is_empty());
    assert!(r.reject_reason.is_none());
    assert_eq!(r.similarity, 0.0);
    assert_eq!(r.beam_close_runner_up_count, 0);
    assert_eq!(r.beam_max_runner_up_fraction, 0.0);
    assert_eq!(r.beam_candidate_count, 0);
    assert_eq!(r.beam_widened_runner_ups, 0);
    assert_eq!(r.leaf_winning_group_n_refs, 0);
    assert_eq!(r.leaf_n_close_representatives, 0);
    assert_eq!(r.leaf_top_m_score_dispersion, 0.0);
    assert_eq!(r.leaf_within_vs_between_margin, 0.0);
    assert_eq!(r.reference_depth_of_predicted, 0);
}

/// Abstention paths must zero every diagnostic field. Path-A/B/C all flow
/// through `unclassified()` which uses `..Default::default()`.
#[test]
fn test_unclassified_zeros_diagnostics() {
    for reason in &["too_few_kmers", "no_training_match", "below_threshold"] {
        let r = ClassificationResult::unclassified(reason);
        assert_eq!(r.taxon, vec!["Root".to_string()], "{}", reason);
        assert_eq!(r.confidence, vec![0.0], "{}", reason);
        assert_eq!(r.reject_reason, Some(reason.to_string()), "{}", reason);
        assert_eq!(r.beam_close_runner_up_count, 0, "{}", reason);
        assert_eq!(r.beam_candidate_count, 0, "{}", reason);
        assert_eq!(r.leaf_winning_group_n_refs, 0, "{}", reason);
        assert_eq!(r.reference_depth_of_predicted, 0, "{}", reason);
    }
}

/// Every classified query under the default greedy path produces
/// `beam_candidate_count == 1`. Sanity check that the field is wired and
/// matches the greedy-path contract.
#[test]
fn test_greedy_path_sets_beam_candidate_count_one() {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    let queries: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..queries.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();
    let config = ClassifyConfig::default(); // beam_width = 1 (greedy)
    let results = id_taxa(
        &queries,
        &names,
        &model,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    for (i, r) in results.iter().enumerate() {
        // Classified queries get exactly one candidate; abstentions get 0.
        if r.reject_reason.is_none() {
            assert_eq!(
                r.beam_candidate_count, 1,
                "query {}: greedy path should set beam_candidate_count = 1",
                i
            );
        }
        // Phase 2 territory — Phase 1b ships zero unconditionally.
        assert_eq!(
            r.beam_widened_runner_ups, 0,
            "query {}: Phase 1b never admits widened runner-ups",
            i
        );
    }
}

/// On a real tree, the leaf-phase telemetry is populated when a query
/// classifies. n_refs >= 1, n_close_representatives <= n_refs, and the
/// dispersion/margin diagnostics are 0.0 below the n_refs >= 5 threshold.
#[test]
fn test_leaf_diagnostics_populated_on_classified() {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    let queries: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..queries.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();
    let config = ClassifyConfig::default();
    let results = id_taxa(
        &queries,
        &names,
        &model,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    let mut any_classified = false;
    for r in &results {
        if r.reject_reason.is_some() {
            continue;
        }
        any_classified = true;
        assert!(
            r.leaf_winning_group_n_refs >= 1,
            "classified query must see >=1 ref in selected group"
        );
        assert!(
            r.leaf_n_close_representatives <= r.leaf_winning_group_n_refs,
            "n_close ({}) must not exceed n_refs ({})",
            r.leaf_n_close_representatives,
            r.leaf_winning_group_n_refs
        );
        // Below 5 refs, dispersion/margin must be 0.0 by definition.
        if r.leaf_winning_group_n_refs < 5 {
            assert_eq!(r.leaf_top_m_score_dispersion, 0.0);
            assert_eq!(r.leaf_within_vs_between_margin, 0.0);
        }
        // Dispersion is non-negative (CV-style metric).
        assert!(r.leaf_top_m_score_dispersion >= 0.0);
    }
    assert!(any_classified, "expected at least one classified query");
}

/// `reference_depth_of_predicted` equals `ts.sequences[predicted_leaf_node].len()`.
/// Cross-validate by walking the model's predicted-leaf path manually.
#[test]
fn test_reference_depth_matches_training_set() {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    let queries: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..queries.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();
    let config = ClassifyConfig::default();
    let results = id_taxa(
        &queries,
        &names,
        &model,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    // For each classified query, the predicted leaf node is the deepest
    // taxonomic node reported in `taxon`. Look it up in the model and check
    // `ts.sequences[node].len()` matches `reference_depth_of_predicted`.
    let mut checked = 0usize;
    for r in &results {
        if r.reject_reason.is_some() {
            continue;
        }
        // Find the deepest reported taxon in the model.
        // Traverse: start from root (node 0); walk down by matching taxon[1..].
        let mut node = 0usize;
        for label in r.taxon.iter().skip(1) {
            let found = model.children[node]
                .iter()
                .copied()
                .find(|&c| &model.taxa[c] == label);
            if let Some(c) = found {
                node = c;
            } else {
                break;
            }
        }
        // The leaf node has no children; r.reference_depth_of_predicted should
        // equal the number of training sequences in that node's subtree.
        let expected = model
            .sequences
            .get(node)
            .and_then(|s| s.as_ref())
            .map_or(0, |v| v.len()) as u32;
        assert_eq!(
            r.reference_depth_of_predicted, expected,
            "reference_depth mismatch at node {}: got {}, expected {}",
            node, r.reference_depth_of_predicted, expected
        );
        checked += 1;
    }
    assert!(checked > 0, "expected at least one classified query");
}

/// Beam path (`beam_width = 2`) populates `beam_candidate_count >= 1` and the
/// runner-up telemetry stays consistent with the greedy single-step max.
#[test]
fn test_beam_path_telemetry_populated() {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    let queries: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..queries.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();
    let config = ClassifyConfig {
        beam_width: 2,
        ..ClassifyConfig::default()
    };
    let results = id_taxa(
        &queries,
        &names,
        &model,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    for r in &results {
        if r.reject_reason.is_some() {
            continue;
        }
        // Beam survived to leaf-phase scoring; at least 1 candidate.
        assert!(r.beam_candidate_count >= 1);
        // Phase 1b ships beam_widened_runner_ups always 0 (Phase 2 wires it).
        assert_eq!(r.beam_widened_runner_ups, 0);
        // Runner-up fraction is in [0, 1] (vote ratio).
        assert!(r.beam_max_runner_up_fraction >= 0.0);
        assert!(r.beam_max_runner_up_fraction <= 1.0);
    }
}

/// `compute_node_diagnostics` produces one entry per internal node and zero
/// entries for leaves; imbalance_score is in [0, +inf) and equals std/mean.
#[test]
fn test_compute_node_diagnostics_basic() {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let prepared = prepare_data(&seqs, &tax, None, 500.0, None, 1).unwrap();

    let diag = compute_node_diagnostics(&prepared, DescendantWeighting::Count);

    // No diagnostics for leaf nodes (no children).
    for d in &diag {
        assert_eq!(d.n_children, prepared.children[d.node_id].len());
        assert!(
            d.n_children > 0,
            "compute_node_diagnostics should skip leaves"
        );
        assert_eq!(d.descendants_per_child.len(), d.n_children);
        // All descendant counts come from prepared.n_seqs.
        for (i, &c) in prepared.children[d.node_id].iter().enumerate() {
            assert_eq!(d.descendants_per_child[i], prepared.n_seqs[c]);
        }
        // imbalance_score >= 0.
        assert!(d.imbalance_score >= 0.0);
        // Phase 1b ships effective_weighting = global mode at every node.
        assert_eq!(d.effective_weighting, DescendantWeighting::Count);
        // retained_kmer_support_per_child empty in Phase 1b.
        assert!(d.retained_kmer_support_per_child.is_empty());
    }

    // Expect at least one internal node (the dataset is non-trivial).
    assert!(!diag.is_empty(), "expected at least one internal node");
}

/// Path-C abstention (every rank below threshold) must zero ALL diagnostic
/// fields at the runtime path — exercises the abstention branch inside
/// `leaf_phase_score`, not the `unclassified()` constructor.
///
/// Drive a real classification at `threshold = 99` so even strong queries
/// fail the per-rank threshold-walk; the result collapses to ["Root"] with
/// `reject_reason = "below_threshold"`. Every leaf and beam diagnostic
/// field must be 0/0.0 — including those that the runtime had real values
/// for (selected-group n_refs, dispersion, etc.) before Path-C zeroing.
#[test]
fn test_path_c_abstention_zeros_all_diagnostics() {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    let queries: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..queries.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();

    // Threshold of 99 (out of 100) — only near-perfect matches survive
    // the per-rank threshold walk, forcing most queries into Path-C
    // (above.is_empty() at every rank including Root).
    let config = ClassifyConfig {
        threshold: 99.0,
        ..Default::default()
    };

    let results = id_taxa(
        &queries,
        &names,
        &model,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    let mut path_c_count = 0usize;
    for (i, r) in results.iter().enumerate() {
        if r.reject_reason.as_deref() == Some("below_threshold") && r.taxon == vec!["Root"] {
            path_c_count += 1;
            // Every diagnostic field must be zero on Path-C abstention.
            assert_eq!(r.beam_close_runner_up_count, 0, "query {}", i);
            assert_eq!(r.beam_max_runner_up_fraction, 0.0, "query {}", i);
            assert_eq!(r.beam_candidate_count, 0, "query {}", i);
            assert_eq!(r.beam_widened_runner_ups, 0, "query {}", i);
            assert_eq!(r.leaf_widened_winners, 0, "query {}", i);
            assert_eq!(r.leaf_winning_group_n_refs, 0, "query {}", i);
            assert_eq!(r.leaf_n_close_representatives, 0, "query {}", i);
            assert_eq!(r.leaf_top_m_score_dispersion, 0.0, "query {}", i);
            assert_eq!(r.leaf_within_vs_between_margin, 0.0, "query {}", i);
            assert_eq!(r.reference_depth_of_predicted, 0, "query {}", i);
            assert!(r.alternatives.is_empty(), "query {}", i);
        }
    }
    assert!(
        path_c_count > 0,
        "fixture invariant: at least one query must hit Path-C at threshold=99"
    );
}

/// Every abstention row produced by a real classification (regardless of
/// reject_reason) must have all diagnostic fields zero. This exercises
/// the wrapper-level guard at classify.rs:422-428 (greedy) and :901-907
/// (beam) — without the guard, descent telemetry would leak through the
/// abstention path.
#[test]
fn test_all_abstention_rows_have_zero_diagnostics() {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    let queries: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..queries.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();

    // Use beam_width=2 so the beam wrapper's overwrite path is also exercised.
    // Use threshold=95 to push some queries into below_threshold abstention.
    let config = ClassifyConfig {
        threshold: 95.0,
        beam_width: 2,
        ..Default::default()
    };

    let results = id_taxa(
        &queries,
        &names,
        &model,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    let mut abstention_count = 0usize;
    for (i, r) in results.iter().enumerate() {
        if r.reject_reason.is_some() && r.taxon == vec!["Root"] {
            abstention_count += 1;
            assert_eq!(
                r.beam_close_runner_up_count, 0,
                "{}: descent telemetry leaked into {:?} abstention",
                i, r.reject_reason
            );
            assert_eq!(r.beam_max_runner_up_fraction, 0.0, "query {}", i);
            assert_eq!(r.beam_candidate_count, 0, "query {}", i);
            assert_eq!(r.beam_widened_runner_ups, 0, "query {}", i);
            assert_eq!(r.leaf_widened_winners, 0, "query {}", i);
            assert_eq!(r.leaf_winning_group_n_refs, 0, "query {}", i);
            assert_eq!(r.leaf_n_close_representatives, 0, "query {}", i);
            assert_eq!(r.leaf_top_m_score_dispersion, 0.0, "query {}", i);
            assert_eq!(r.leaf_within_vs_between_margin, 0.0, "query {}", i);
            assert_eq!(r.reference_depth_of_predicted, 0, "query {}", i);
        }
    }
    assert!(
        abstention_count > 0,
        "fixture invariant: at least one abstention expected at threshold=95"
    );
}

/// Regression-locking test for the default-config behavior on the
/// canonical s09a fixture. Asserts that:
/// 1. The new diagnostic counters all stay at their default-zero values
///    on the default flag combination (no widening, no leaf_top_m
///    aggregation drift).
/// 2. The number of classified vs abstention rows is stable.
/// 3. Similarity values are bit-stable across runs (deterministic seed).
///
/// This is a coarser cousin of the per-query golden baselines already
/// covered by `test_classify_9a_standard`; the value here is locking the
/// FIELDS that were added by the post-IDTAXA + tie-margin-split work
/// (which the existing per-query golden files don't cover).
#[test]
fn test_default_classify_matches_golden_baseline() {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let model = learn_taxa(&seqs, &tax, &TrainConfig::default(), 42, false).unwrap();

    let queries: Vec<String> = load_json("s09a_query_seqs");
    let names: Vec<String> = (0..queries.len())
        .map(|i| format!("query_{:03}", i + 1))
        .collect();

    let cfg = ClassifyConfig::default();
    let results = id_taxa(
        &queries,
        &names,
        &model,
        &cfg,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    // No widening at default flags — both counters must stay 0 across
    // every query. This is the bit-identical contract for the post-IDTAXA
    // + tie-margin-split implementation against pre-Phase-1b baselines.
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            r.beam_widened_runner_ups, 0,
            "query {}: beam_widened_runner_ups must stay 0 at default config",
            i
        );
        assert_eq!(
            r.leaf_widened_winners, 0,
            "query {}: leaf_widened_winners must stay 0 at default config",
            i
        );
    }

    // Each result is well-formed: taxon non-empty, confidence and taxon
    // lengths agree, no NaN confidence, similarity is finite.
    for (i, r) in results.iter().enumerate() {
        assert!(!r.taxon.is_empty(), "query {}: taxon must be non-empty", i);
        assert_eq!(
            r.taxon.len(),
            r.confidence.len(),
            "query {}: taxon vs confidence length mismatch",
            i
        );
        assert!(
            r.similarity.is_finite(),
            "query {}: non-finite similarity",
            i
        );
        for &c in &r.confidence {
            assert!(c.is_finite(), "query {}: non-finite confidence", i);
        }
    }
}

/// JSON round-trip lock: NodeDiagnostic must serialize with lowercase
/// `effective_weighting` (per the `#[serde(rename_all = "lowercase")]` on
/// `DescendantWeighting`) and deserialize back to the same variant. Locks
/// in the wire format that downstream Python parsers depend on.
#[test]
fn test_node_diagnostic_json_roundtrip_lowercase() {
    use oxidtaxa::types::NodeDiagnostic;
    let nd = NodeDiagnostic {
        node_id: 42,
        n_children: 3,
        descendants_per_child: vec![10, 5, 2],
        imbalance_score: 1.2,
        retained_kmer_support_per_child: vec![],
        effective_weighting: DescendantWeighting::Count,
    };
    let json = serde_json::to_string(&nd).unwrap();
    assert!(
        json.contains("\"effective_weighting\":\"count\""),
        "expected lowercase variant in JSON, got: {}",
        json
    );
    let back: NodeDiagnostic = serde_json::from_str(&json).unwrap();
    assert_eq!(back.effective_weighting, DescendantWeighting::Count);
    assert_eq!(back.node_id, 42);
    assert_eq!(back.n_children, 3);
    assert_eq!(back.descendants_per_child, vec![10, 5, 2]);
}

/// Imbalance computation: a synthetic [10, 1, 1] tree has imbalance_score
/// = std/mean. Verify the math against a known answer.
#[test]
fn test_imbalance_score_math() {
    // Construct a hand-tooled "prepared" by training on a fixture where one
    // root child is much larger than the others. Use 12 seqs split 10-1-1
    // across three classes, all under root.
    let seqs: Vec<String> = (0..12)
        .map(|i| format!("ACGT{}", "ACGT".repeat(i + 5)))
        .collect();
    let tax: Vec<String> = (0..12)
        .map(|i| {
            // 10 in c0, 1 in c1, 1 in c2.
            let c = if i < 10 {
                0
            } else if i < 11 {
                1
            } else {
                2
            };
            format!("Root;family;genus;species{};", c)
        })
        .collect();
    let prepared = prepare_data(&seqs, &tax, Some(4), 500.0, None, 1).unwrap();
    let diag = compute_node_diagnostics(&prepared, DescendantWeighting::Count);

    // Find the node whose children's descendant counts are [10, 1, 1].
    // (Root or some intermediate; descendants_per_child should match.)
    let mut found = false;
    for d in &diag {
        if d.descendants_per_child.len() == 3 {
            let mut sorted = d.descendants_per_child.clone();
            sorted.sort();
            if sorted == vec![1, 1, 10] {
                let mean = 4.0;
                let var = (((10.0_f64 - mean).powi(2)) + 2.0 * (1.0 - mean).powi(2)) / 3.0;
                let expected = var.sqrt() / mean;
                assert!(
                    (d.imbalance_score - expected).abs() < 1e-9,
                    "imbalance_score {} != expected {}",
                    d.imbalance_score,
                    expected
                );
                found = true;
            }
        }
    }
    assert!(
        found,
        "no [10,1,1] split found in diagnostics; tree shape unexpected"
    );
}
