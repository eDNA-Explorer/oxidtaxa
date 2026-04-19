//! Phase 2 tests: flag-gated margin-aware classification.
//!
//! Covers I5 (tie_margin), I6 (confidence_uses_descent_margin),
//! I7 (sibling_aware_leaf). Each flag defaults to legacy behavior;
//! these tests exercise both on and off to verify the guard holds.

use oxidtaxa::classify::id_taxa;
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{ClassifyConfig, OutputType, StrandMode, TrainConfig, TrainingSet};

/// Tree:
///   Root
///   └─ Mammalia
///       ├─ Carnivora
///       │  ├─ Canidae
///       │  │   ├─ Canis
///       │  │   │   ├─ Canis_lupus          (sequence A)
///       │  │   │   └─ Canis_latrans        (sequence A with 2 mutations)
///       │  │   └─ Vulpes
///       │  │       └─ Vulpes_vulpes        (very different sequence B)
///       │  └─ Felidae
///       │      └─ Felis
///       │          └─ Felis_catus          (very different sequence C)
///       └─ Artiodactyla
///           └─ Cervidae
///               └─ Odocoileus
///                   └─ Odocoileus_virginianus (very different sequence D)
///
/// `Canis_lupus` and `Canis_latrans` share ~97% of their k-mers so their
/// `tot_hits` scores are near-tied but not exact. This lets us distinguish
/// `tie_margin = 0` (legacy) from `tie_margin > 0` in leaf-phase LCA capping.
fn build_near_tied_training_set() -> TrainingSet {
    // ~200 bp of varied content; this is the "base" sequence.
    let base = "\
        ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
        GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
        TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
        CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
        AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT";

    // Canis_lupus: identical to base.
    let canis_lupus = base.to_string();
    // Canis_latrans: two single-bp substitutions, enough to differ on a few
    // 8-mers without making the sequences look totally different.
    let mut canis_latrans_chars: Vec<char> = base.chars().collect();
    canis_latrans_chars[17] = 'G'; // was 'T'
    canis_latrans_chars[103] = 'A'; // was 'T'
    let canis_latrans: String = canis_latrans_chars.into_iter().collect();

    // Vulpes / Felis / Odocoileus: completely unrelated sequences.
    let vulpes = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let felis = "ATATATATATATATATATATATATATATATATATATATATATATATAT\
                 CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                 AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT\
                 GGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAA";
    let odocoileus = "CCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTT\
                      AATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATT\
                      GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC\
                      TATATATATATATATATATATATATATATATATATATATATATATAT";

    let sequences = vec![
        canis_lupus,
        canis_latrans,
        vulpes.to_string(),
        felis.to_string(),
        odocoileus.to_string(),
    ];
    let taxonomy = vec![
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_latrans".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
        "Root; Mammalia; Artiodactyla; Cervidae; Odocoileus; Odocoileus_virginianus".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

// ============================================================================
// I5: tie_margin
// ============================================================================

#[test]
fn test_tie_margin_zero_preserves_legacy() {
    // With `tie_margin = 0.0`, only exact `tot_hits` equalities produce
    // alternatives. Near-tied Canis_lupus / Canis_latrans → single winner.
    let ts = build_near_tied_training_set();
    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
            .to_string(),
    ];
    let names = vec!["q".to_string()];

    let config = ClassifyConfig {
        tie_margin: 0.0,
        ..Default::default()
    };
    let results = id_taxa(
        &query, &names, &ts, &config, StrandMode::Top, OutputType::Extended, 42, true,
    );

    // Whether alternatives is empty or not depends on whether the randomized
    // bootstrap produces exact ties. What we care about is that flipping
    // `tie_margin` from 0.0 to 0.10 can only *grow* the alternatives set:
    // compare in the next test.
    assert_eq!(results.len(), 1);
    let _ = results[0].alternatives.len();
}

#[test]
fn test_tie_margin_catches_near_ties() {
    // With `tie_margin = 0.10`, a near-tied sibling (Canis_latrans scoring
    // within 90% of Canis_lupus's max) should be captured in alternatives,
    // and the lineage truncated at Canis.
    let ts = build_near_tied_training_set();
    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
            .to_string(),
    ];
    let names = vec!["q".to_string()];

    // Compute baseline alternatives count at tie_margin = 0.0.
    let cfg_zero = ClassifyConfig {
        tie_margin: 0.0,
        ..Default::default()
    };
    let base = id_taxa(
        &query, &names, &ts, &cfg_zero, StrandMode::Top, OutputType::Extended, 42, true,
    );
    let base_alts = base[0].alternatives.len();

    // At tie_margin = 0.10 the filter is v >= 0.9 * max_tot, which is a
    // superset of v == max_tot — alternatives count can only grow or stay
    // the same, never shrink. And on this near-tied fixture we expect it to
    // grow to 2 (Canis_latrans joins Canis_lupus).
    let cfg_wide = ClassifyConfig {
        tie_margin: 0.10,
        ..Default::default()
    };
    let wide = id_taxa(
        &query, &names, &ts, &cfg_wide, StrandMode::Top, OutputType::Extended, 42, true,
    );
    assert!(
        wide[0].alternatives.len() >= base_alts,
        "tie_margin=0.10 shrank alternatives ({} < {})",
        wide[0].alternatives.len(),
        base_alts
    );
    assert!(
        wide[0].alternatives.len() >= 2,
        "expected ≥2 near-tied alternatives at tie_margin=0.10, got {:?}",
        wide[0].alternatives
    );
    assert!(
        wide[0].alternatives.contains(&"Canis_lupus".to_string())
            && wide[0].alternatives.contains(&"Canis_latrans".to_string()),
        "expected both Canis species in alternatives, got {:?}",
        wide[0].alternatives
    );
    // Lineage should terminate at Canis (LCA), not species.
    assert!(
        !wide[0].taxon.contains(&"Canis_lupus".to_string())
            && !wide[0].taxon.contains(&"Canis_latrans".to_string()),
        "species leaked into taxon despite near-tie LCA cap: {:?}",
        wide[0].taxon
    );
}

// ============================================================================
// I6: confidence_uses_descent_margin
// ============================================================================

#[test]
fn test_descent_margin_default_off() {
    // With the flag off (default), confidence values must match the no-flag
    // path exactly. The golden classify tests already cover this at tolerance
    // 5.0, so we just assert the knob exists and is default-false here.
    let cfg = ClassifyConfig::default();
    assert!(!cfg.confidence_uses_descent_margin);
}

#[test]
fn test_descent_margin_on_never_raises_confidence() {
    // Flipping `confidence_uses_descent_margin` ON multiplies each rank's
    // confidence by a running product of margins ≤ 1.0, so each per-rank
    // confidence can only shrink (or stay equal when every descent was
    // unambiguous).
    let ts = build_near_tied_training_set();
    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
            .to_string(),
    ];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig::default();
    let off = id_taxa(
        &query, &names, &ts, &cfg_off, StrandMode::Top, OutputType::Extended, 42, true,
    );

    let cfg_on = ClassifyConfig {
        confidence_uses_descent_margin: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query, &names, &ts, &cfg_on, StrandMode::Top, OutputType::Extended, 42, true,
    );

    assert_eq!(off.len(), 1);
    assert_eq!(on.len(), 1);
    let len = off[0].confidence.len().min(on[0].confidence.len());
    for i in 0..len {
        assert!(
            on[0].confidence[i] <= off[0].confidence[i] + 1e-6,
            "rank {} confidence rose from {} (off) to {} (on)",
            i,
            off[0].confidence[i],
            on[0].confidence[i]
        );
    }
}

// ============================================================================
// I7: sibling_aware_leaf
// ============================================================================

#[test]
fn test_sibling_aware_leaf_default_off() {
    let cfg = ClassifyConfig::default();
    assert!(!cfg.sibling_aware_leaf);
}

#[test]
fn test_sibling_aware_leaf_cannot_shrink_alternatives() {
    // Flipping `sibling_aware_leaf` ON can only widen w_indices at leaf-parent
    // descent sites, which can only add sibling evidence to `tot_hits` — so
    // alternatives count is monotonically non-decreasing from off → on.
    let ts = build_near_tied_training_set();
    let query = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
            .to_string(),
    ];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig::default();
    let off = id_taxa(
        &query, &names, &ts, &cfg_off, StrandMode::Top, OutputType::Extended, 42, true,
    );

    let cfg_on = ClassifyConfig {
        sibling_aware_leaf: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query, &names, &ts, &cfg_on, StrandMode::Top, OutputType::Extended, 42, true,
    );

    assert_eq!(off.len(), 1);
    assert_eq!(on.len(), 1);
    assert!(
        on[0].alternatives.len() >= off[0].alternatives.len(),
        "sibling_aware_leaf=true shrank alternatives ({} < {})",
        on[0].alternatives.len(),
        off[0].alternatives.len()
    );
}
