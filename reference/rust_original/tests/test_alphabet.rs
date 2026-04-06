mod common;

use common::{assert_approx_eq, load_json};
use idtaxa::alphabet::alphabet_size;

/// Load alphabet_size sequences — may be a single string or array of strings.
fn load_seqs(name: &str) -> Vec<String> {
    // Try as Vec<String> first, then as single String
    let path = common::golden_json_dir().join(format!("{}.json", name));
    let content = std::fs::read_to_string(&path).unwrap();
    match serde_json::from_str::<Vec<String>>(&content) {
        Ok(v) => v,
        Err(_) => {
            let s: String = serde_json::from_str(&content).unwrap();
            vec![s]
        }
    }
}

/// Load alphabet_size golden value — may be a single f64 or array of f64.
fn load_val(name: &str) -> f64 {
    let path = common::golden_json_dir().join(format!("{}.json", name));
    let content = std::fs::read_to_string(&path).unwrap();
    match serde_json::from_str::<f64>(&content) {
        Ok(v) => v,
        Err(_) => {
            let v: Vec<f64> = serde_json::from_str(&content).unwrap();
            v[0]
        }
    }
}

#[test]
fn test_alphabet_size_uniform() {
    let seqs = load_seqs("s03_as_seqs_uniform");
    let golden = load_val("s03_as_val_uniform");
    let result = alphabet_size(&seqs);
    assert_approx_eq(result, golden, 1e-10, "uniform alphabet_size");
}

#[test]
fn test_alphabet_size_biased_a() {
    let seqs = load_seqs("s03_as_seqs_biased_A");
    let golden = load_val("s03_as_val_biased_A");
    let result = alphabet_size(&seqs);
    assert_approx_eq(result, golden, 1e-10, "biased_A alphabet_size");
}

#[test]
fn test_alphabet_size_biased_gc() {
    let seqs = load_seqs("s03_as_seqs_biased_GC");
    let golden = load_val("s03_as_val_biased_GC");
    let result = alphabet_size(&seqs);
    assert_approx_eq(result, golden, 1e-10, "biased_GC alphabet_size");
}

#[test]
fn test_alphabet_size_all_a() {
    let seqs = load_seqs("s03_as_seqs_all_A");
    let golden = load_val("s03_as_val_all_A");
    let result = alphabet_size(&seqs);
    assert_approx_eq(result, golden, 1e-10, "all_A alphabet_size");
}

#[test]
fn test_alphabet_size_many_seqs() {
    let seqs = load_seqs("s03_as_seqs_many_seqs");
    let golden = load_val("s03_as_val_many_seqs");
    let result = alphabet_size(&seqs);
    assert_approx_eq(result, golden, 1e-10, "many_seqs alphabet_size");
}
