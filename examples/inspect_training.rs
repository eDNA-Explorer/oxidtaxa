//! Inspect training model to understand problem sequences.
use oxidtaxa::fasta::{read_fasta, read_taxonomy};
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::TrainConfig;

fn filter_for_training(seqs: &[String], taxonomy: &[String]) -> (Vec<String>, Vec<String>) {
    let mut fs = Vec::new();
    let mut ft = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        let tax = &taxonomy[i];
        let full_tax = format!("Root; {}", tax.replace(";", "; "));
        if full_tax.split("; ").count() < 4 || seq.len() < 30 { continue; }
        let n_count = seq.bytes().filter(|&b| b == b'N' || b == b'n').count();
        if (n_count as f64 / seq.len() as f64) > 0.3 { continue; }
        fs.push(seq.clone());
        ft.push(full_tax);
    }
    (fs, ft)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (ref_names, ref_seqs) = read_fasta(&args[1]).unwrap();
    let ref_tax = read_taxonomy(&args[2], &ref_names).unwrap();
    let (train_seqs, train_tax) = filter_for_training(&ref_seqs, &ref_tax);

    println!("Training {} sequences...", train_seqs.len());
    let config = TrainConfig::default();
    let model = learn_taxa(&train_seqs, &train_tax, &config, 42, false).unwrap();

    // Fraction statistics
    let total_nodes = model.fraction.len();
    let none_count = model.fraction.iter().filter(|f| f.is_none()).count();
    let below_half = model.fraction.iter().filter(|f| matches!(f, Some(v) if *v < 0.03)).count();
    let at_max = model.fraction.iter().filter(|f| matches!(f, Some(v) if (*v - 0.06).abs() < 1e-10)).count();
    println!("\nFraction distribution ({} nodes):", total_nodes);
    println!("  At max (0.06):    {:>5} ({:.1}%)", at_max, 100.0 * at_max as f64 / total_nodes as f64);
    println!("  Below 0.03:       {:>5} ({:.1}%)", below_half, 100.0 * below_half as f64 / total_nodes as f64);
    println!("  At None (floor):  {:>5} ({:.1}%)", none_count, 100.0 * none_count as f64 / total_nodes as f64);

    println!("\nProblem groups ({}):", model.problem_groups.len());
    for g in &model.problem_groups {
        println!("  {}", g);
    }

    println!("\nProblem sequences: {} / {} ({:.1}%)",
        model.problem_sequences.len(), train_seqs.len(),
        100.0 * model.problem_sequences.len() as f64 / train_seqs.len() as f64);

    // Show where problem sequences fail
    let mut fail_nodes: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for ps in &model.problem_sequences {
        // The predicted field shows where misclassification happened
        let depth = ps.predicted.split(';').filter(|s| !s.is_empty()).count();
        let expected_depth = ps.expected.split(';').filter(|s| !s.is_empty()).count();
        *fail_nodes.entry(if ps.predicted.is_empty() { "(stopped at Root)" } else { "misclassified" }).or_insert(0) += 1;
    }

    println!("\nProblem sequence breakdown:");
    for (kind, count) in &fail_nodes {
        println!("  {}: {}", kind, count);
    }

    // Show a few examples
    println!("\nSample problem sequences (first 10):");
    for ps in model.problem_sequences.iter().take(10) {
        let exp_short: String = ps.expected.split(';').filter(|s| !s.is_empty()).last().unwrap_or("?").to_string();
        let pred_short: String = if ps.predicted.is_empty() {
            "(gave up)".to_string()
        } else {
            ps.predicted.split(';').filter(|s| !s.is_empty()).last().unwrap_or("?").to_string()
        };
        println!("  seq {:>4}: expected ...{:<30} predicted ...{}", ps.index, exp_short, pred_short);
    }

    // Show the taxonomy depth where misclassifications happen
    let mut fail_depth_counts: Vec<usize> = vec![0; 20];
    for ps in &model.problem_sequences {
        let pred_depth = if ps.predicted.is_empty() { 0 } else {
            ps.predicted.split(';').filter(|s| !s.is_empty()).count()
        };
        if pred_depth < 20 { fail_depth_counts[pred_depth] += 1; }
    }
    println!("\nMisclassification depth distribution:");
    for (d, &c) in fail_depth_counts.iter().enumerate() {
        if c > 0 {
            let rank = match d {
                0 => "gave up (no prediction)",
                1 => "Root",
                2 => "Domain",
                3 => "Phylum",
                4 => "Class",
                5 => "Order",
                6 => "Family",
                7 => "Genus",
                8 => "Species",
                _ => "deeper",
            };
            println!("  depth {}: {:>4} ({:<25})", d, c, rank);
        }
    }
}
