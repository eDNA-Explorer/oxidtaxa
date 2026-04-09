mod common;

use common::load_json;
use oxidtaxa::classify::id_taxa;
use oxidtaxa::fasta::{read_fasta, write_classification_tsv};
use oxidtaxa::sequence::remove_gaps;
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{ClassifyConfig, OutputType, StrandMode, TrainConfig};

#[derive(serde::Deserialize)]
struct GoldenTsvRow {
    read_id: String,
    taxonomic_path: String,
    confidence: f64,
    #[serde(default)]
    alternatives: String,
}

#[test]
fn test_full_pipeline_e2e() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = manifest_dir.join("tests").join("data");

    // Read FASTA and taxonomy
    let (names, seqs) = read_fasta(data_dir.join("test_ref.fasta").to_str().unwrap()).unwrap();
    let taxonomy = oxidtaxa::fasta::read_taxonomy(
        data_dir.join("test_ref_taxonomy.tsv").to_str().unwrap(),
        &names,
    )
    .unwrap();

    // Filter for training (same as train_idtaxa.R)
    let mut filtered_seqs = Vec::new();
    let mut filtered_tax = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        let tax = &taxonomy[i];
        let full_tax = format!("Root; {}", tax.replace(";", "; "));
        let rank_count = full_tax.split("; ").count();
        if rank_count < 4 {
            continue;
        }
        if seq.len() < 30 {
            continue;
        }
        let n_count = seq.bytes().filter(|&b| b == b'N' || b == b'n').count();
        if (n_count as f64 / seq.len() as f64) > 0.3 {
            continue;
        }
        filtered_seqs.push(seq.clone());
        filtered_tax.push(full_tax);
    }

    // Train
    let config = TrainConfig::default();
    let ts = learn_taxa(&filtered_seqs, &filtered_tax, &config, 42, false).unwrap();

    // Read query FASTA
    let (query_names, query_seqs) =
        read_fasta(data_dir.join("test_query.fasta").to_str().unwrap()).unwrap();
    let clean_seqs = remove_gaps(&query_seqs);

    // Classify
    let classify_config = ClassifyConfig::default();
    let results = id_taxa(
        &clean_seqs,
        &query_names,
        &ts,
        &classify_config,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    // Write TSV
    let tmp_dir = std::env::temp_dir();
    let output_path = tmp_dir.join("idtaxa_e2e_test.tsv");
    write_classification_tsv(output_path.to_str().unwrap(), &query_names, &results).unwrap();

    // Compare against golden
    let golden: Vec<GoldenTsvRow> = load_json("s10a_e2e_tsv");
    let output_content = std::fs::read_to_string(&output_path).unwrap();
    let lines: Vec<&str> = output_content.lines().collect();

    assert_eq!(
        lines[0],
        "read_id\ttaxonomic_path\tconfidence\talternatives"
    );
    assert_eq!(lines.len() - 1, golden.len(), "row count mismatch");

    for (i, line) in lines[1..].iter().enumerate() {
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            parts.len(),
            4,
            "expected 4 columns at row {}, got {}",
            i,
            parts.len()
        );
        assert_eq!(parts[0], golden[i].read_id, "read_id mismatch at row {}", i);
        assert_eq!(
            parts[1], golden[i].taxonomic_path,
            "taxonomic_path mismatch at row {}",
            i
        );
        let conf: f64 = parts[2].parse().unwrap();
        assert!(
            (conf - golden[i].confidence).abs() < 5.0,
            "confidence mismatch at row {}: {} vs {}",
            i,
            conf,
            golden[i].confidence
        );
        assert_eq!(
            parts[3], golden[i].alternatives,
            "alternatives mismatch at row {}",
            i
        );
    }

    // Cleanup
    let _ = std::fs::remove_file(output_path);
}
