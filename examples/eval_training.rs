//! Evaluation harness for training algorithm changes.
//!
//! Trains two models and compares classification results to validate
//! that training changes produce equivalent classification accuracy.
//!
//! Usage:
//!   cargo run --example eval_training --release -- \
//!       ref.fasta ref_taxonomy.tsv query.fasta --seed 42
//!
//!   # Compare original order vs shuffled (tests order-independence):
//!   cargo run --example eval_training --release -- \
//!       ref.fasta ref_taxonomy.tsv query.fasta --seed 42 --compare-shuffled
//!
//!   # Save baseline JSON:
//!   cargo run --example eval_training --release -- \
//!       ref.fasta ref_taxonomy.tsv query.fasta --seed 42 --save-baseline baseline.json

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use oxidtaxa::classify::id_taxa;
use oxidtaxa::fasta::{read_fasta, read_taxonomy};
use oxidtaxa::sequence::remove_gaps;
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{
    ClassificationResult, ClassifyConfig, OutputType, StrandMode, TrainConfig,
};

// ── Metrics ─────────────────────────────────────────────────────────────────

struct ComparisonMetrics {
    n_queries: usize,
    exact_path_agreement: f64,
    genus_level_agreement: f64,
    mean_confidence_diff: f64,
    max_confidence_diff: f64,
    problem_seq_count_a: usize,
    problem_seq_count_b: usize,
    problem_group_count_a: usize,
    problem_group_count_b: usize,
}

impl ComparisonMetrics {
    fn pass(&self) -> bool {
        self.exact_path_agreement >= 98.0
            && self.genus_level_agreement >= 99.0
            && self.mean_confidence_diff < 3.0
            && self.max_confidence_diff < 15.0
            && {
                let a = self.problem_seq_count_a as f64;
                let b = self.problem_seq_count_b as f64;
                let max = a.max(b);
                let min = a.min(b);
                max == 0.0 || (max - min) / max <= 0.20
            }
            && (self.problem_group_count_a as i64 - self.problem_group_count_b as i64).unsigned_abs() <= 2
    }

    fn print(&self) {
        println!("\n{:=<70}", "= Comparison Metrics ");
        println!(
            "  {:40} {:>10} {:>10}",
            "Metric", "Value", "Threshold"
        );
        println!("{:-<70}", "");

        let pf = |ok: bool| if ok { "PASS" } else { "FAIL" };

        let epa = self.exact_path_agreement;
        println!(
            "  {:40} {:>9.1}% {:>10} [{}]",
            "Exact path agreement",
            epa,
            ">= 98%",
            pf(epa >= 98.0)
        );
        let gla = self.genus_level_agreement;
        println!(
            "  {:40} {:>9.1}% {:>10} [{}]",
            "Genus-level agreement",
            gla,
            ">= 99%",
            pf(gla >= 99.0)
        );
        let mcd = self.mean_confidence_diff;
        println!(
            "  {:40} {:>10.2} {:>10} [{}]",
            "Mean confidence diff",
            mcd,
            "< 3.0",
            pf(mcd < 3.0)
        );
        let xcd = self.max_confidence_diff;
        println!(
            "  {:40} {:>10.2} {:>10} [{}]",
            "Max confidence diff",
            xcd,
            "< 15.0",
            pf(xcd < 15.0)
        );
        let psa = self.problem_seq_count_a;
        let psb = self.problem_seq_count_b;
        println!(
            "  {:40} {:>4} vs {:<4} {:>10} [{}]",
            "Problem seq count (A vs B)",
            psa,
            psb,
            "within 20%",
            pf({
                let max = psa.max(psb) as f64;
                let min = psa.min(psb) as f64;
                max == 0.0 || (max - min) / max <= 0.20
            })
        );
        let pga = self.problem_group_count_a;
        let pgb = self.problem_group_count_b;
        println!(
            "  {:40} {:>4} vs {:<4} {:>10} [{}]",
            "Problem group count (A vs B)",
            pga,
            pgb,
            "within 2",
            pf((pga as i64 - pgb as i64).unsigned_abs() <= 2)
        );

        println!("{:-<70}", "");
        if self.pass() {
            println!("  VERDICT: PASS");
        } else {
            println!("  VERDICT: FAIL");
        }
        println!("{:=<70}", "");
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn filter_for_training(seqs: &[String], taxonomy: &[String]) -> (Vec<String>, Vec<String>) {
    let mut filtered_seqs = Vec::new();
    let mut filtered_tax = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        let tax = &taxonomy[i];
        let full_tax = format!("Root; {}", tax.replace(";", "; "));
        let rank_count = full_tax.split("; ").count();
        if rank_count < 4 || seq.len() < 30 {
            continue;
        }
        let n_count = seq.bytes().filter(|&b| b == b'N' || b == b'n').count();
        if (n_count as f64 / seq.len() as f64) > 0.3 {
            continue;
        }
        filtered_seqs.push(seq.clone());
        filtered_tax.push(full_tax);
    }
    (filtered_seqs, filtered_tax)
}

fn extract_path(r: &ClassificationResult) -> String {
    let mut taxa = r.taxon.clone();
    if taxa.len() > 1 {
        taxa.remove(0); // skip Root
    }
    taxa.into_iter()
        .filter(|t| !t.starts_with("unclassified_"))
        .collect::<Vec<_>>()
        .join(";")
}

fn extract_confidence(r: &ClassificationResult) -> f64 {
    if r.confidence.len() > 1 {
        r.confidence[1..]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    } else {
        0.0
    }
}

fn genus_level_path(path: &str) -> String {
    // Take up to 6 ranks (domain;phylum;class;order;family;genus)
    let parts: Vec<&str> = path.split(';').collect();
    parts[..parts.len().min(6)].join(";")
}

fn compare_results(
    results_a: &[ClassificationResult],
    results_b: &[ClassificationResult],
    problem_seq_a: usize,
    problem_seq_b: usize,
    problem_group_a: usize,
    problem_group_b: usize,
) -> ComparisonMetrics {
    let n = results_a.len();
    assert_eq!(n, results_b.len());

    let mut exact_match = 0usize;
    let mut genus_match = 0usize;
    let mut total_conf_diff = 0.0f64;
    let mut max_conf_diff = 0.0f64;

    for (a, b) in results_a.iter().zip(results_b.iter()) {
        let path_a = extract_path(a);
        let path_b = extract_path(b);
        if path_a == path_b {
            exact_match += 1;
        }
        if genus_level_path(&path_a) == genus_level_path(&path_b) {
            genus_match += 1;
        }
        let conf_a = extract_confidence(a);
        let conf_b = extract_confidence(b);
        let diff = (conf_a - conf_b).abs();
        total_conf_diff += diff;
        if diff > max_conf_diff {
            max_conf_diff = diff;
        }
    }

    ComparisonMetrics {
        n_queries: n,
        exact_path_agreement: if n > 0 { 100.0 * exact_match as f64 / n as f64 } else { 100.0 },
        genus_level_agreement: if n > 0 { 100.0 * genus_match as f64 / n as f64 } else { 100.0 },
        mean_confidence_diff: if n > 0 { total_conf_diff / n as f64 } else { 0.0 },
        max_confidence_diff: max_conf_diff,
        problem_seq_count_a: problem_seq_a,
        problem_seq_count_b: problem_seq_b,
        problem_group_count_a: problem_group_a,
        problem_group_count_b: problem_group_b,
    }
}

/// Deterministic shuffle based on hash of accession + seed.
fn deterministic_shuffle(
    seqs: &[String],
    tax: &[String],
    seed: u64,
) -> (Vec<String>, Vec<String>) {
    let n = seqs.len();
    let mut indices: Vec<usize> = (0..n).collect();
    // Fisher-Yates using deterministic hash
    for i in (1..n).rev() {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        i.hash(&mut hasher);
        let h = hasher.finish() as usize;
        let j = h % (i + 1);
        indices.swap(i, j);
    }
    let shuffled_seqs: Vec<String> = indices.iter().map(|&i| seqs[i].clone()).collect();
    let shuffled_tax: Vec<String> = indices.iter().map(|&i| tax[i].clone()).collect();
    (shuffled_seqs, shuffled_tax)
}

fn train_and_classify(
    train_seqs: &[String],
    train_tax: &[String],
    query_seqs: &[String],
    query_names: &[String],
    seed: u32,
    label: &str,
    max_iterations: Option<usize>,
) -> (Vec<ClassificationResult>, usize, usize) {
    let t0 = std::time::Instant::now();
    let config = TrainConfig {
        max_iterations: max_iterations.unwrap_or(10),
        ..Default::default()
    };
    let model = learn_taxa(train_seqs, train_tax, &config, seed, false)
        .unwrap_or_else(|e| panic!("Training failed ({}): {}", label, e));
    let train_time = t0.elapsed();

    let problem_seqs = model.problem_sequences.len();
    let problem_groups = model.problem_groups.len();
    println!(
        "  [{}] Trained in {:.2}s — {} taxa, {} problem seqs, {} problem groups",
        label,
        train_time.as_secs_f64(),
        model.taxonomy.len(),
        problem_seqs,
        problem_groups,
    );

    let t0 = std::time::Instant::now();
    let classify_config = ClassifyConfig::default();
    let results = id_taxa(
        query_seqs,
        query_names,
        &model,
        &classify_config,
        StrandMode::Both,
        OutputType::Extended,
        seed,
        true,
    );
    let classify_time = t0.elapsed();
    println!(
        "  [{}] Classified {} queries in {:.2}s",
        label,
        results.len(),
        classify_time.as_secs_f64(),
    );

    (results, problem_seqs, problem_groups)
}

// ── Baseline serialization ──────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct BaselineEntry {
    path: String,
    confidence: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Baseline {
    n_queries: usize,
    problem_sequences: usize,
    problem_groups: usize,
    results: Vec<BaselineEntry>,
}

fn save_baseline(
    path: &str,
    results: &[ClassificationResult],
    problem_seqs: usize,
    problem_groups: usize,
) {
    let baseline = Baseline {
        n_queries: results.len(),
        problem_sequences: problem_seqs,
        problem_groups,
        results: results
            .iter()
            .map(|r| BaselineEntry {
                path: extract_path(r),
                confidence: extract_confidence(r),
            })
            .collect(),
    };
    let json = serde_json::to_string_pretty(&baseline).unwrap();
    std::fs::write(path, json).unwrap();
    println!("  Baseline saved to {}", path);
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        eprintln!(
            "Usage: {} <ref.fasta> <ref_taxonomy.tsv> <query.fasta> [--seed N] [--compare-shuffled] [--save-baseline path.json]",
            args[0]
        );
        std::process::exit(1);
    }

    let ref_fasta = &args[1];
    let ref_taxonomy = &args[2];
    let query_fasta = &args[3];

    let mut seed: u32 = 42;
    let mut compare_shuffled = false;
    let mut save_baseline_path: Option<String> = None;
    let mut max_iterations: Option<usize> = None;

    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("Invalid seed");
            }
            "--compare-shuffled" => {
                compare_shuffled = true;
            }
            "--save-baseline" => {
                i += 1;
                save_baseline_path = Some(args[i].clone());
            }
            "--max-iterations" => {
                i += 1;
                max_iterations = Some(args[i].parse().expect("Invalid max-iterations"));
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load data
    println!("Loading data...");
    let (ref_names, ref_seqs) = read_fasta(ref_fasta).unwrap();
    let ref_tax = read_taxonomy(ref_taxonomy, &ref_names).unwrap();
    let (query_names, query_seqs) = read_fasta(query_fasta).unwrap();
    let clean_queries = remove_gaps(&query_seqs);

    println!(
        "  {} ref sequences, {} query sequences",
        ref_seqs.len(),
        clean_queries.len()
    );

    // Filter for training
    let (train_seqs, train_tax) = filter_for_training(&ref_seqs, &ref_tax);
    println!("  {} sequences after quality filtering", train_seqs.len());

    // Train model A (original order)
    println!("\nTraining model A (original order)...");
    let (results_a, ps_a, pg_a) =
        train_and_classify(&train_seqs, &train_tax, &clean_queries, &query_names, seed, "A", max_iterations);

    // Save baseline if requested
    if let Some(ref path) = save_baseline_path {
        save_baseline(path, &results_a, ps_a, pg_a);
    }

    if compare_shuffled {
        // Train model B (shuffled order)
        println!("\nTraining model B (shuffled order)...");
        let (shuffled_seqs, shuffled_tax) =
            deterministic_shuffle(&train_seqs, &train_tax, seed as u64);
        let (results_b, ps_b, pg_b) = train_and_classify(
            &shuffled_seqs,
            &shuffled_tax,
            &clean_queries,
            &query_names,
            seed,
            "B-shuffled",
            max_iterations,
        );

        let metrics = compare_results(&results_a, &results_b, ps_a, ps_b, pg_a, pg_b);
        metrics.print();

        println!("\n  ({} queries compared)", metrics.n_queries);

        if !metrics.pass() {
            std::process::exit(1);
        }
    } else {
        println!("\nModel A summary:");
        println!("  Queries classified: {}", results_a.len());
        println!("  Problem sequences:  {}", ps_a);
        println!("  Problem groups:     {}", pg_a);
        println!("\n  (use --compare-shuffled to test order-independence)");
    }
}
