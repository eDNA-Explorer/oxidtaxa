mod common;

use common::{assert_approx_eq, golden_json_dir, load_json};
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::TrainConfig;

/// Golden training set structure from JSON.
#[derive(serde::Deserialize, Debug)]
struct GoldenTrainingSet {
    taxonomy: Vec<String>,
    taxa: Vec<String>,
    levels: Vec<i32>,
    #[serde(rename = "K")]
    k: usize,
    children: Vec<Vec<usize>>,
    parents: Vec<usize>,
    #[serde(rename = "crossIndex")]
    cross_index: Vec<usize>,
    kmers: Vec<Vec<f64>>, // stored as float in JSON
    #[serde(rename = "IDFweights")]
    idf_weights: Vec<f64>,
    fraction: Vec<Option<f64>>,
    #[serde(rename = "decisionKmers")]
    decision_kmers: Vec<Option<GoldenDecisionNode>>,
    #[serde(rename = "problemSequences")]
    problem_sequences: Vec<GoldenProblemSeq>,
    #[serde(rename = "problemGroups")]
    problem_groups: Vec<String>,
}

#[derive(serde::Deserialize, Debug)]
struct GoldenDecisionNode {
    keep: Vec<f64>, // stored as float
    profiles: Vec<Vec<f64>>,
}

#[derive(serde::Deserialize, Debug)]
struct GoldenProblemSeq {
    index: usize,
    expected: String,
    predicted: String,
}

fn load_training_inputs(seqs_name: &str, tax_name: &str) -> (Vec<String>, Vec<String>) {
    let seqs: Vec<String> = load_json(seqs_name);
    let tax: Vec<String> = load_json(tax_name);
    (seqs, tax)
}

fn compare_training_set(label: &str, golden: &GoldenTrainingSet, result: &oxidtaxa::types::TrainingSet) {
    assert_eq!(result.k, golden.k, "{}: K mismatch", label);
    assert_eq!(result.taxonomy, golden.taxonomy, "{}: taxonomy mismatch", label);
    assert_eq!(result.taxa, golden.taxa, "{}: taxa mismatch", label);
    assert_eq!(result.levels, golden.levels, "{}: levels mismatch", label);

    // children: Rust is 0-indexed, R golden is 1-indexed → add 1 to Rust values
    let rust_children_1idx: Vec<Vec<usize>> = result.children.iter()
        .map(|ch| ch.iter().map(|&c| c + 1).collect())
        .collect();
    assert_eq!(rust_children_1idx, golden.children, "{}: children mismatch", label);

    // parents: same offset
    let rust_parents_1idx: Vec<usize> = result.parents.iter().map(|&p| if p == 0 && result.parents[0] == 0 { 0 } else { p + 1 }).collect();
    // R's parents[1] = 0 (Root has no parent), parents[2] = 1, etc.
    // Actually R stores parents as: Root's parent = 0, others = 1-indexed
    // Our Rust: parents[0] = 0 (Root), parents[1] = 0 (child of Root), etc.
    // R: parents[1] = 0, parents[2] = 1, parents[3] = 2 ...
    // So for node i: R's parents[i+1] = Rust's parents[i] + 1 (except Root: R=0, Rust=0)
    // But index 0 in R's parents array is parents[1] = 0. Let me just check:
    // R parents are 1-indexed internally, and parents[Root] = 0 means "no parent"
    // Our Rust parents[0] = 0 means Root is its own parent (or no parent)
    // For other nodes: Rust parents[i] = j means j is parent (0-indexed)
    // R parents[i] = j means j is parent (1-indexed), where 0 = no parent
    // So: Rust parents[i] → R parents[i] = Rust parents[i] + 1, except Root (0→0)
    let rust_parents_as_r: Vec<usize> = result.parents.iter().enumerate()
        .map(|(i, &p)| if i == 0 { 0 } else { p + 1 })
        .collect();
    assert_eq!(rust_parents_as_r, golden.parents, "{}: parents mismatch", label);

    // crossIndex: Rust 0-indexed → R 1-indexed
    let rust_ci_1idx: Vec<usize> = result.cross_index.iter().map(|&c| c + 1).collect();
    assert_eq!(rust_ci_1idx, golden.cross_index, "{}: crossIndex mismatch", label);

    // kmers: golden stores as float, convert
    assert_eq!(result.kmers.len(), golden.kmers.len(), "{}: kmers length mismatch", label);
    for (i, (got, exp)) in result.kmers.iter().zip(golden.kmers.iter()).enumerate() {
        let exp_i32: Vec<i32> = exp.iter().map(|&v| v as i32).collect();
        assert_eq!(got, &exp_i32, "{}: kmers[{}] mismatch", label, i);
    }

    // IDFweights (float comparison)
    assert_eq!(result.idf_weights.len(), golden.idf_weights.len(), "{}: IDFweights length mismatch", label);
    let max_diff: f64 = result.idf_weights.iter()
        .zip(golden.idf_weights.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(max_diff < 1e-10, "{}: IDFweights max diff {} >= 1e-10", label, max_diff);

    // fraction (with NA matching)
    assert_eq!(result.fraction.len(), golden.fraction.len(), "{}: fraction length mismatch", label);
    for (i, (got, exp)) in result.fraction.iter().zip(golden.fraction.iter()).enumerate() {
        match (got, exp) {
            (None, None) => {},
            (Some(g), Some(e)) => {
                assert!((g - e).abs() < 1e-10, "{}: fraction[{}] mismatch: {} vs {}", label, i, g, e);
            },
            _ => panic!("{}: fraction[{}] NA pattern mismatch: {:?} vs {:?}", label, i, got, exp),
        }
    }

    // decisionKmers structure
    assert_eq!(result.decision_kmers.len(), golden.decision_kmers.len(), "{}: decisionKmers length mismatch", label);

    // problemSequences count
    assert_eq!(result.problem_sequences.len(), golden.problem_sequences.len(),
        "{}: problemSequences count mismatch", label);

    // problemGroups
    assert_eq!(result.problem_groups, golden.problem_groups, "{}: problemGroups mismatch", label);
}

#[test]
fn test_training_8a_standard() {
    let (seqs, tax) = load_training_inputs("s08a_filtered_seqs", "s08a_taxonomy_vec");
    let golden: GoldenTrainingSet = load_json("s08a_training_set");

    let config = TrainConfig::default();
    let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    compare_training_set("8a_standard", &golden, &result);
}

#[test]
fn test_training_8b_asymmetric() {
    let (seqs, tax) = load_training_inputs("s08b_asym_seqs", "s08b_asym_tax");
    let golden: GoldenTrainingSet = load_json("s08b_asym_training_set");

    let config = TrainConfig::default();
    let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    compare_training_set("8b_asym", &golden, &result);
}

#[test]
fn test_training_8c_problem_groups() {
    let (seqs, tax) = load_training_inputs("s08c_problem_seqs", "s08c_problem_tax");
    let golden: GoldenTrainingSet = load_json("s08c_problem_training_set");

    let config = TrainConfig::default();
    let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    compare_training_set("8c_problem", &golden, &result);
}

#[test]
fn test_training_8d_singletons() {
    let (seqs, tax) = load_training_inputs("s08d_singleton_seqs", "s08d_singleton_tax");
    let golden: GoldenTrainingSet = load_json("s08d_singleton_training_set");

    let config = TrainConfig::default();
    let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();

    compare_training_set("8d_singleton", &golden, &result);
}

#[test]
fn test_training_8e_explicit_k() {
    let (seqs, tax) = load_training_inputs("s08a_filtered_seqs", "s08a_taxonomy_vec");

    // K=5
    let golden_k5: GoldenTrainingSet = load_json("s08e_training_set_k5");
    let config_k5 = TrainConfig { k: Some(5), ..Default::default() };
    let result_k5 = learn_taxa(&seqs, &tax, &config_k5, 42, false).unwrap();
    compare_training_set("8e_k5", &golden_k5, &result_k5);

    // K=10
    let golden_k10: GoldenTrainingSet = load_json("s08e_training_set_k10");
    let config_k10 = TrainConfig { k: Some(10), ..Default::default() };
    let result_k10 = learn_taxa(&seqs, &tax, &config_k10, 42, false).unwrap();
    compare_training_set("8e_k10", &golden_k10, &result_k10);
}
