mod common;

use common::load_json;
use oxidtaxa::fasta::read_fasta;

#[test]
fn test_fasta_read_matches_golden() {
    let golden_seqs: Vec<String> = load_json("s01_fasta_seqs");
    let golden_names: Vec<String> = load_json("s01_fasta_names");

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fasta_path = manifest_dir
        .join("tests")
        .join("data")
        .join("test_ref.fasta");

    let (names, seqs) = read_fasta(fasta_path.to_str().unwrap()).unwrap();

    assert_eq!(names.len(), golden_names.len(), "sequence count mismatch");
    assert_eq!(seqs.len(), golden_seqs.len(), "sequence count mismatch");

    for (i, name) in names.iter().enumerate() {
        assert_eq!(name, &golden_names[i], "name mismatch at index {}", i);
    }

    for (i, seq) in seqs.iter().enumerate() {
        assert_eq!(
            seq, &golden_seqs[i],
            "sequence mismatch at index {}",
            i
        );
    }
}
