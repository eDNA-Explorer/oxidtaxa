mod common;

use oxidtaxa::classify::id_taxa;
use oxidtaxa::fasta::write_classification_tsv;
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{ClassifyConfig, OutputType, StrandMode, TrainConfig};

/// Build a minimal training set:
///   Root > Mammalia > Carnivora > Canidae > Canis   > {Canis_lupus, Canis_latrans}
///                                         > Vulpes  > {Vulpes_vulpes}
///                               > Felidae > Felis   > {Felis_catus}
///
/// `Canis_lupus` and `Canis_latrans` are given IDENTICAL sequences so they
/// will tie at `tot_hits` during classification. `Vulpes_vulpes` and
/// `Felis_catus` have distinct sequences so the classifier has enough
/// context to disambiguate and so the tree has realistic shape.
fn build_tied_training_set() -> oxidtaxa::types::TrainingSet {
    // Synthetic ~200bp sequences. Identical for Canis_lupus / Canis_latrans.
    let tied_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                    GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
                    TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA";
    let vulpes_seq = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                      GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                      GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let felis_seq = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
                     CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                     AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT";

    let sequences = vec![
        tied_seq.to_string(),
        tied_seq.to_string(),
        vulpes_seq.to_string(),
        felis_seq.to_string(),
    ];
    let taxonomy = vec![
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_latrans".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

#[test]
fn two_way_tie_caps_at_genus_and_populates_alternatives() {
    let ts = build_tied_training_set();

    // Query = the exact tied sequence → should hit both Canis species identically.
    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA"
            .to_string(),
    ];
    let query_names = vec!["tied_query".to_string()];

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query,
        &query_names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true, // deterministic
    );

    assert_eq!(results.len(), 1);
    let r = &results[0];

    // Lineage should end at Canis, then the unclassified placeholder.
    assert!(
        r.taxon.contains(&"Canis".to_string()),
        "expected Canis in taxon, got {:?}",
        r.taxon
    );
    assert_eq!(
        r.taxon.last().unwrap(),
        "unclassified_Canis",
        "expected lineage to terminate at unclassified_Canis, got {:?}",
        r.taxon
    );
    assert!(
        !r.taxon.contains(&"Canis_lupus".to_string()),
        "species rank must not leak into taxon: {:?}",
        r.taxon
    );
    assert!(
        !r.taxon.contains(&"Canis_latrans".to_string()),
        "species rank must not leak into taxon: {:?}",
        r.taxon
    );

    // Alternatives should contain both tied species, sorted alphabetically.
    assert_eq!(
        r.alternatives,
        vec!["Canis_latrans".to_string(), "Canis_lupus".to_string()],
        "expected alternatives = [Canis_latrans, Canis_lupus], got {:?}",
        r.alternatives
    );
}

#[test]
fn non_tied_classification_has_empty_alternatives() {
    let ts = build_tied_training_set();

    // Query = the Vulpes sequence → no tie expected.
    let query = vec![
        "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT"
            .to_string(),
    ];
    let query_names = vec!["vulpes_query".to_string()];

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query,
        &query_names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(results.len(), 1);
    assert!(
        results[0].alternatives.is_empty(),
        "non-tied classification should have empty alternatives, got {:?}",
        results[0].alternatives
    );
}

#[test]
fn tied_alternatives_appear_in_tsv_output() {
    let ts = build_tied_training_set();

    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA"
            .to_string(),
    ];
    let query_names = vec!["tied_query".to_string()];

    let config = ClassifyConfig::default();
    let results = id_taxa(
        &query,
        &query_names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let tmp_dir = std::env::temp_dir();
    let output_path = tmp_dir.join("oxidtaxa_ties_test.tsv");
    write_classification_tsv(output_path.to_str().unwrap(), &query_names, &results).unwrap();

    let content = std::fs::read_to_string(&output_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    assert_eq!(lines[0], "read_id\ttaxonomic_path\tconfidence\talternatives");
    assert_eq!(
        lines.len(),
        2,
        "expected header + 1 row, got {} lines",
        lines.len()
    );

    let parts: Vec<&str> = lines[1].split('\t').collect();
    assert_eq!(parts.len(), 4, "expected 4 columns, got {}", parts.len());
    assert_eq!(parts[0], "tied_query");
    assert_eq!(parts[3], "Canis_latrans|Canis_lupus");

    let _ = std::fs::remove_file(output_path);
}
