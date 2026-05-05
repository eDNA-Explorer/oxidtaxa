mod common;

use common::{golden_json_dir, load_json};
use oxidtaxa::classify::id_taxa;
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{ClassificationResult, ClassifyConfig, OutputType, StrandMode, TrainConfig};
use std::collections::HashMap;

/// Golden classification result from JSON.
#[derive(serde::Deserialize, Debug)]
struct GoldenClassResult {
    taxon: Vec<String>,
    confidence: Vec<f64>,
}

const CONF_TOLERANCE: f64 = 5.0; // tolerance for confidence comparison

fn train_standard_model() -> oxidtaxa::types::TrainingSet {
    let seqs: Vec<String> = load_json("s08a_filtered_seqs");
    let tax: Vec<String> = load_json("s08a_taxonomy_vec");
    let config = TrainConfig::default();
    learn_taxa(&seqs, &tax, &config, 42, false).unwrap()
}

fn train_problem_model() -> oxidtaxa::types::TrainingSet {
    let seqs: Vec<String> = load_json("s08c_problem_seqs");
    let tax: Vec<String> = load_json("s08c_problem_tax");
    let config = TrainConfig::default();
    learn_taxa(&seqs, &tax, &config, 42, false).unwrap()
}

fn train_singleton_model() -> oxidtaxa::types::TrainingSet {
    let seqs: Vec<String> = load_json("s08d_singleton_seqs");
    let tax: Vec<String> = load_json("s08d_singleton_tax");
    let config = TrainConfig::default();
    learn_taxa(&seqs, &tax, &config, 42, false).unwrap()
}

/// Compare classification results against golden data stored as a named dict.
fn compare_classification_dict(
    label: &str,
    golden: &HashMap<String, GoldenClassResult>,
    results: &[ClassificationResult],
    names: &[String],
) {
    assert_eq!(
        results.len(),
        golden.len(),
        "{}: count mismatch {} vs {}",
        label,
        results.len(),
        golden.len()
    );

    let mut max_conf_diff: f64 = 0.0;
    let mut all_taxa_ok = true;

    for (i, result) in results.iter().enumerate() {
        let name = &names[i];
        let key = name.split_whitespace().next().unwrap_or(name);
        let gold = golden
            .get(key)
            .or_else(|| golden.get(name.as_str()))
            .unwrap_or_else(|| panic!("{}: no golden for '{}'", label, key));

        if result.taxon != gold.taxon {
            eprintln!("{} seq {}: taxon mismatch", label, i);
            eprintln!("  got:    {:?}", result.taxon);
            eprintln!("  golden: {:?}", gold.taxon);
            all_taxa_ok = false;
        }

        if result.confidence.len() == gold.confidence.len() {
            for (&got, &exp) in result.confidence.iter().zip(gold.confidence.iter()) {
                let diff = (got - exp).abs();
                if diff > max_conf_diff {
                    max_conf_diff = diff;
                }
            }
        }
    }

    assert!(all_taxa_ok, "{}: taxa not identical", label);
    assert!(
        max_conf_diff < CONF_TOLERANCE,
        "{}: confidence diff {} >= {}",
        label,
        max_conf_diff,
        CONF_TOLERANCE
    );
}

/// Compare classification results against golden data stored as an array.
fn compare_classification_array(
    label: &str,
    golden: &[GoldenClassResult],
    results: &[ClassificationResult],
) {
    assert_eq!(
        results.len(),
        golden.len(),
        "{}: count mismatch {} vs {}",
        label,
        results.len(),
        golden.len()
    );

    let mut max_conf_diff: f64 = 0.0;
    let mut all_taxa_ok = true;

    for (i, (result, gold)) in results.iter().zip(golden.iter()).enumerate() {
        if result.taxon != gold.taxon {
            eprintln!("{} seq {}: taxon mismatch", label, i);
            eprintln!("  got:    {:?}", result.taxon);
            eprintln!("  golden: {:?}", gold.taxon);
            all_taxa_ok = false;
        }

        if result.confidence.len() == gold.confidence.len() {
            for (&got, &exp) in result.confidence.iter().zip(gold.confidence.iter()) {
                let diff = (got - exp).abs();
                if diff > max_conf_diff {
                    max_conf_diff = diff;
                }
            }
        }
    }

    assert!(all_taxa_ok, "{}: taxa not identical", label);
    assert!(
        max_conf_diff < CONF_TOLERANCE,
        "{}: confidence diff {} >= {}",
        label,
        max_conf_diff,
        CONF_TOLERANCE
    );
}

fn make_names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{}_{:03}", prefix, i + 1)).collect()
}

#[test]
fn test_inverted_index_and_merge_fallback_classification_match() {
    let ts = train_standard_model();
    let mut merge_ts = ts.clone();
    merge_ts.inverted_index = None;
    let query_seqs: Vec<String> = load_json("s09a_query_seqs");
    let names = make_names("query", query_seqs.len());
    let config = ClassifyConfig {
        threshold: 40.0,
        min_descend: 0.98,
        leaf_top_m: 3,
        deepest_rank_diagnostics: true,
        deepest_rank_diagnostic_top_k: 3,
        rank_trace_diagnostics: true,
        ..Default::default()
    };

    let inverted_results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let merge_results = id_taxa(
        &query_seqs,
        &names,
        &merge_ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(inverted_results.len(), merge_results.len());
    for (inverted, merge) in inverted_results.iter().zip(merge_results.iter()) {
        assert_eq!(inverted.taxon, merge.taxon);
        assert_eq!(inverted.confidence, merge.confidence);
        assert_eq!(inverted.alternatives, merge.alternatives);
        assert_eq!(inverted.reject_reason, merge.reject_reason);
        assert_eq!(inverted.similarity, merge.similarity);
        assert_eq!(inverted.leaf_widened_winners, merge.leaf_widened_winners);
        assert_eq!(
            inverted.leaf_winning_group_n_refs,
            merge.leaf_winning_group_n_refs
        );
        assert_eq!(
            inverted.leaf_n_close_representatives,
            merge.leaf_n_close_representatives
        );
        assert_eq!(
            inverted.leaf_top_m_score_dispersion,
            merge.leaf_top_m_score_dispersion
        );
        assert_eq!(
            inverted.leaf_within_vs_between_margin,
            merge.leaf_within_vs_between_margin
        );
        assert_eq!(inverted.leaf_top_confidence, merge.leaf_top_confidence);
        assert_eq!(
            inverted.leaf_runner_up_confidence,
            merge.leaf_runner_up_confidence
        );
        assert_eq!(inverted.leaf_margin_delta, merge.leaf_margin_delta);
        assert_eq!(inverted.leaf_margin_ratio, merge.leaf_margin_ratio);
        assert_eq!(
            inverted.leaf_margin_candidate_count,
            merge.leaf_margin_candidate_count
        );
        assert_eq!(inverted.rank_trace.len(), merge.rank_trace.len());
    }
}

// ============================================================================
// 9a: Standard classification
// ============================================================================
#[test]
fn test_classify_9a_standard() {
    let ts = train_standard_model();
    let query_seqs: Vec<String> = load_json("s09a_query_seqs");
    let names = make_names("query", query_seqs.len());
    let golden: HashMap<String, GoldenClassResult> = load_json("s09a_ids_standard");

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_dict("9a_standard", &golden, &results, &names);
}

// ============================================================================
// 9b: Perfect match (query IS a training sequence)
// ============================================================================
#[test]
fn test_classify_9b_perfect() {
    let ts = train_standard_model();
    let query_seqs: Vec<String> = load_json("s09b_perfect_query");
    let golden: HashMap<String, GoldenClassResult> = load_json("s09b_ids_perfect");
    let names = vec![
        "seq_001".to_string(),
        "seq_020".to_string(),
        "seq_040".to_string(),
        "seq_060".to_string(),
    ];

    let config = ClassifyConfig {
        threshold: 60.0,
        min_descend: 0.98,
        processors: 1,
        ..Default::default()
    };
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_dict("9b_perfect", &golden, &results, &names);
}

// ============================================================================
// 9c: Novel organism
// ============================================================================
#[test]
fn test_classify_9c_novel() {
    let ts = train_standard_model();
    let query_seqs: Vec<String> = load_json("s09c_novel_seqs");
    let names = vec![
        "random1".to_string(),
        "random2".to_string(),
        "random3".to_string(),
    ];
    let golden: HashMap<String, GoldenClassResult> = load_json("s09c_ids_novel");

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_dict("9c_novel", &golden, &results, &names);
}

// ============================================================================
// 9d: Threshold sweep
// ============================================================================
#[test]
fn test_classify_9d_threshold_sweep() {
    let ts = train_standard_model();
    let all_query: Vec<String> = load_json("s09a_query_seqs");
    let query_seqs = all_query[..5].to_vec();
    let names = make_names("query", 5);

    for thresh in [0.0, 30.0, 50.0, 60.0, 80.0, 95.0, 100.0] {
        let golden_name = format!("s09d_ids_thresh_{}", thresh as i32);
        let golden: HashMap<String, GoldenClassResult> = load_json(&golden_name);

        let config = ClassifyConfig {
            threshold: thresh,
            min_descend: 0.98,
            processors: 1,
            ..Default::default()
        };
        let results = id_taxa(
            &query_seqs,
            &names,
            &ts,
            &config,
            StrandMode::Both,
            OutputType::Extended,
            42,
            true,
        );

        compare_classification_dict(
            &format!("9d_thresh_{}", thresh as i32),
            &golden,
            &results,
            &names,
        );
    }
}

// ============================================================================
// 9e: Strand variations
// ============================================================================
#[test]
fn test_classify_9e_strand_top() {
    let ts = train_standard_model();
    let all_query: Vec<String> = load_json("s09a_query_seqs");
    let query_seqs = all_query[..5].to_vec();
    let names = make_names("query", 5);
    let golden: HashMap<String, GoldenClassResult> = load_json("s09e_ids_strand_top");

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_dict("9e_top", &golden, &results, &names);
}

#[test]
fn test_classify_9e_strand_bottom() {
    let ts = train_standard_model();
    let all_query: Vec<String> = load_json("s09a_query_seqs");
    let query_seqs = all_query[..5].to_vec();
    let names = make_names("query", 5);
    let golden: HashMap<String, GoldenClassResult> = load_json("s09e_ids_strand_bottom");

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Bottom,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_dict("9e_bottom", &golden, &results, &names);
}

#[test]
fn test_classify_9e_strand_both() {
    let ts = train_standard_model();
    let all_query: Vec<String> = load_json("s09a_query_seqs");
    let query_seqs = all_query[..5].to_vec();
    let names = make_names("query", 5);
    let golden: HashMap<String, GoldenClassResult> = load_json("s09e_ids_strand_both");

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_dict("9e_both", &golden, &results, &names);
}

// ============================================================================
// 9f: Duplicate queries (de-replication)
// ============================================================================
#[test]
fn test_classify_9f_duplicates() {
    let ts = train_standard_model();
    let query_seqs: Vec<String> = load_json("s09f_dup_query");
    let names: Vec<String> = (0..query_seqs.len())
        .map(|i| format!("dup_{}", i + 1))
        .collect();
    let golden: HashMap<String, GoldenClassResult> = load_json("s09f_ids_dup");

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_dict("9f_dup", &golden, &results, &names);
}

// ============================================================================
// 9g: Very short query
// ============================================================================
#[test]
fn test_classify_9g_short() {
    let ts = train_standard_model();
    // Golden is a bare string (auto_unboxed single element)
    let path = golden_json_dir().join("s09g_short_query.json");
    let content = std::fs::read_to_string(&path).unwrap();
    let query_seqs: Vec<String> = match serde_json::from_str::<Vec<String>>(&content) {
        Ok(v) => v,
        Err(_) => vec![serde_json::from_str::<String>(&content).unwrap()],
    };
    let names = vec!["tiny".to_string()];

    // Golden may be a dict or array
    let golden_path = golden_json_dir().join("s09g_ids_short.json");
    let golden_content = std::fs::read_to_string(&golden_path).unwrap();
    let golden: Vec<GoldenClassResult> = match serde_json::from_str(&golden_content) {
        Ok(v) => v,
        Err(_) => {
            let g: HashMap<String, GoldenClassResult> =
                serde_json::from_str(&golden_content).unwrap();
            g.into_values().collect()
        }
    };

    let config = ClassifyConfig {
        threshold: 60.0,
        min_descend: 0.98,
        processors: 1,
        ..Default::default()
    };
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_array("9g_short", &golden, &results);
}

// ============================================================================
// 9h: Bootstrap sweep
// ============================================================================
#[test]
fn test_classify_9h_bootstrap_sweep() {
    let ts = train_standard_model();
    let all_query: Vec<String> = load_json("s09a_query_seqs");
    let query_seqs = all_query[..3].to_vec();
    let names = make_names("query", 3);

    // Bootstraps hardcoded at 100; test against boots=100 golden only
    for boots in [100] {
        let golden_name = format!("s09h_ids_boots_{}", boots);
        let golden: HashMap<String, GoldenClassResult> = load_json(&golden_name);

        let config = ClassifyConfig::default();
        let results = id_taxa(
            &query_seqs,
            &names,
            &ts,
            &config,
            StrandMode::Both,
            OutputType::Extended,
            42,
            true,
        );

        let tol = CONF_TOLERANCE;
        assert_eq!(
            results.len(),
            golden.len(),
            "9h_boots_{}: count mismatch",
            boots
        );
        let mut all_taxa_ok = true;
        let mut max_diff = 0.0f64;
        for (i, result) in results.iter().enumerate() {
            let key = &names[i];
            if let Some(gold) = golden.get(key.as_str()) {
                if result.taxon != gold.taxon {
                    eprintln!("9h_boots_{} seq {}: taxon mismatch", boots, i);
                    all_taxa_ok = false;
                }
                for (&g, &e) in result.confidence.iter().zip(gold.confidence.iter()) {
                    let d = (g - e).abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                }
            }
        }
        assert!(all_taxa_ok, "9h_boots_{}: taxa mismatch", boots);
        assert!(
            max_diff < tol,
            "9h_boots_{}: conf diff {} >= {}",
            boots,
            max_diff,
            tol
        );
    }
}

// ============================================================================
// 9i: minDescend sweep
// ============================================================================
#[test]
fn test_classify_9i_mindescend_sweep() {
    let ts = train_standard_model();
    let all_query: Vec<String> = load_json("s09a_query_seqs");
    let query_seqs = all_query[..5].to_vec();
    let names = make_names("query", 5);

    for md in [0.5, 0.7, 0.9, 0.98, 1.0] {
        let md_str = format!("{}", md).replace('.', "_");
        let golden_name = format!("s09i_ids_minDescend_{}", md_str);
        let golden: HashMap<String, GoldenClassResult> = load_json(&golden_name);

        let config = ClassifyConfig {
            threshold: 60.0,
            min_descend: md,
            processors: 1,
            ..Default::default()
        };
        let results = id_taxa(
            &query_seqs,
            &names,
            &ts,
            &config,
            StrandMode::Both,
            OutputType::Extended,
            42,
            true,
        );

        compare_classification_dict(&format!("9i_md_{}", md_str), &golden, &results, &names);
    }
}

// ============================================================================
// 9k: Problem group classification
// ============================================================================
#[test]
fn test_classify_9k_problem() {
    let ts = train_problem_model();
    let query_seqs: Vec<String> = load_json("s09k_problem_query");
    let names = make_names("problem", query_seqs.len());
    let golden: Vec<GoldenClassResult> = load_json("s09k_ids_problem");

    let config = ClassifyConfig {
        threshold: 60.0,
        min_descend: 0.98,
        processors: 1,
        ..Default::default()
    };
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_array("9k_problem", &golden, &results);
}

// ============================================================================
// 9l: Singleton classification
// ============================================================================
#[test]
fn test_classify_9l_singleton() {
    let ts = train_singleton_model();
    let query_seqs: Vec<String> = load_json("s09l_singleton_query");
    let names = make_names("singleton", query_seqs.len());
    let golden: Vec<GoldenClassResult> = load_json("s09l_ids_singleton");

    let config = ClassifyConfig {
        threshold: 60.0,
        min_descend: 0.98,
        processors: 1,
        ..Default::default()
    };
    let results = id_taxa(
        &query_seqs,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    compare_classification_array("9l_singleton", &golden, &results);
}
