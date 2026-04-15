use std::time::Instant;
use oxidtaxa::fasta::{read_fasta, read_taxonomy};
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::TrainConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size = args.get(1).map(|s| s.as_str()).unwrap_or("1000");
    let processors: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks").join("data");
    let fasta = base.join(format!("bench_{}_ref.fasta", size));
    let tax_path = base.join(format!("bench_{}_ref_taxonomy.tsv", size));

    if !fasta.exists() {
        eprintln!("Dataset not found: {}", fasta.display());
        std::process::exit(1);
    }

    let (names, seqs) = read_fasta(fasta.to_str().unwrap()).unwrap();
    let taxonomy = read_taxonomy(tax_path.to_str().unwrap(), &names).unwrap();

    // Filter (same as benchmarks)
    let mut filtered_seqs = Vec::new();
    let mut filtered_tax = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        let tax = &taxonomy[i];
        let full_tax = format!("Root; {}", tax.replace(";", "; "));
        let rank_count = full_tax.split("; ").count();
        if rank_count < 4 || seq.len() < 30 { continue; }
        let n_count = seq.bytes().filter(|&b| b == b'N' || b == b'n').count();
        if (n_count as f64 / seq.len() as f64) > 0.3 { continue; }
        filtered_seqs.push(seq.clone());
        filtered_tax.push(full_tax);
    }

    println!("Dataset: {} sequences (filtered from {})", filtered_seqs.len(), seqs.len());
    println!("Processors: {}", processors);

    let config = TrainConfig {
        correlation_aware_features: true,
        record_kmers_fraction: 0.44,
        processors,
        ..TrainConfig::default()
    };

    println!("Training with correlation_aware_features=true, record_kmers_fraction=0.44...");
    let start = Instant::now();
    let result = learn_taxa(&filtered_seqs, &filtered_tax, &config, 42, true);
    let elapsed = start.elapsed();

    match result {
        Ok(ts) => {
            let n_nodes = ts.decision_kmers.iter().filter(|d| d.is_some()).count();
            println!("Done in {:.3}s ({} decision nodes)", elapsed.as_secs_f64(), n_nodes);
        }
        Err(e) => {
            println!("Error after {:.3}s: {}", elapsed.as_secs_f64(), e);
        }
    }
}
