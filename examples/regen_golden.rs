//! Regenerate golden JSON fixtures for training tests.
//! Run: cargo run --example regen_golden

use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::TrainConfig;
use serde_json::json;
use std::path::PathBuf;

fn golden_json_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden_json")
}

fn load_json<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = golden_json_dir().join(format!("{}.json", name));
    let content = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&content).unwrap()
}

fn save_golden(name: &str, ts: &oxidtaxa::types::TrainingSet) {
    // Convert to the golden JSON format matching R's structure
    // children: 0-indexed Rust → 1-indexed R
    let children_1idx: Vec<Vec<usize>> = ts
        .children
        .iter()
        .map(|ch| ch.iter().map(|&c| c + 1).collect())
        .collect();

    // parents: 0-indexed Rust → 1-indexed R (Root → 0)
    let parents_1idx: Vec<usize> = ts
        .parents
        .iter()
        .enumerate()
        .map(|(i, &p)| if i == 0 { 0 } else { p + 1 })
        .collect();

    // crossIndex: 0-indexed Rust → 1-indexed R
    let cross_index_1idx: Vec<usize> = ts.cross_index.iter().map(|&c| c + 1).collect();

    // kmers: Vec<Vec<i32>> → Vec<Vec<f64>> (R stores as float)
    let kmers_f64: Vec<Vec<f64>> = ts
        .kmers
        .iter()
        .map(|v| v.iter().map(|&k| k as f64).collect())
        .collect();

    // decisionKmers: convert to golden format
    let decision_kmers: Vec<Option<serde_json::Value>> = ts
        .decision_kmers
        .iter()
        .map(|dk| {
            dk.as_ref().map(|d| {
                json!({
                    "keep": d.keep.iter().map(|&k| k as f64).collect::<Vec<f64>>(),
                    "profiles": d.profiles,
                })
            })
        })
        .collect();

    // problemSequences
    let problem_sequences: Vec<serde_json::Value> = ts
        .problem_sequences
        .iter()
        .map(|ps| {
            json!({
                "index": ps.index,
                "expected": ps.expected,
                "predicted": ps.predicted,
            })
        })
        .collect();

    let golden = json!({
        "taxonomy": ts.taxonomy,
        "taxa": ts.taxa,
        "levels": ts.levels,
        "K": ts.k,
        "children": children_1idx,
        "parents": parents_1idx,
        "crossIndex": cross_index_1idx,
        "kmers": kmers_f64,
        "IDFweights": ts.idf_weights_by_rank.last().unwrap_or(&Vec::new()),
        "fraction": ts.fraction,
        "decisionKmers": decision_kmers,
        "problemSequences": problem_sequences,
        "problemGroups": ts.problem_groups,
    });

    let path = golden_json_dir().join(format!("{}.json", name));
    let content = serde_json::to_string_pretty(&golden).unwrap();
    std::fs::write(&path, content).unwrap();
    println!("  Wrote {}", path.display());
}

fn main() {
    println!("Regenerating golden training JSON fixtures...\n");

    // 8a: standard
    {
        let seqs: Vec<String> = load_json("s08a_filtered_seqs");
        let tax: Vec<String> = load_json("s08a_taxonomy_vec");
        let config = TrainConfig::default();
        let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
        save_golden("s08a_training_set", &result);
    }

    // 8b: asymmetric
    {
        let seqs: Vec<String> = load_json("s08b_asym_seqs");
        let tax: Vec<String> = load_json("s08b_asym_tax");
        let config = TrainConfig::default();
        let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
        save_golden("s08b_asym_training_set", &result);
    }

    // 8c: problem groups
    {
        let seqs: Vec<String> = load_json("s08c_problem_seqs");
        let tax: Vec<String> = load_json("s08c_problem_tax");
        let config = TrainConfig::default();
        let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
        save_golden("s08c_problem_training_set", &result);
    }

    // 8d: singletons
    {
        let seqs: Vec<String> = load_json("s08d_singleton_seqs");
        let tax: Vec<String> = load_json("s08d_singleton_tax");
        let config = TrainConfig::default();
        let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
        save_golden("s08d_singleton_training_set", &result);
    }

    // 8e: explicit K=5
    {
        let seqs: Vec<String> = load_json("s08a_filtered_seqs");
        let tax: Vec<String> = load_json("s08a_taxonomy_vec");
        let config = TrainConfig {
            k: Some(5),
            ..Default::default()
        };
        let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
        save_golden("s08e_training_set_k5", &result);
    }

    // 8e: explicit K=10
    {
        let seqs: Vec<String> = load_json("s08a_filtered_seqs");
        let tax: Vec<String> = load_json("s08a_taxonomy_vec");
        let config = TrainConfig {
            k: Some(10),
            ..Default::default()
        };
        let result = learn_taxa(&seqs, &tax, &config, 42, false).unwrap();
        save_golden("s08e_training_set_k10", &result);
    }

    println!("\nDone! Now run: cargo test --test test_training");
}
