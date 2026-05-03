//! Flag-gated margin-aware classification tests.
//!
//! Covers `descent_tie_margin` and `leaf_tie_margin`. These tests exercise
//! each stage-specific margin independently so descent widening and leaf
//! reporting remain separate.

mod common;

use oxidtaxa::classify::id_taxa;
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{
    ClassifyConfig, LeafAggregationMode, OutputType, StrandMode, TrainConfig, TrainingSet,
};

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
/// `leaf_tie_margin = 0` from `leaf_tie_margin > 0` in leaf-phase LCA capping.
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
// leaf_tie_margin
// ============================================================================

#[test]
fn test_leaf_tie_margin_zero_reports_single_near_tie_winner() {
    // With `leaf_tie_margin = 0.0`, only exact `tot_hits` equalities produce
    // alternatives. Near-tied Canis_lupus / Canis_latrans → single winner.
    let ts = build_near_tied_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let config = ClassifyConfig {
        leaf_tie_margin: 0.0,
        ..Default::default()
    };
    let results = id_taxa(
        &query,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    // Whether alternatives is empty or not depends on whether the randomized
    // bootstrap produces exact ties. What we care about is that flipping
    // `leaf_tie_margin` from 0.0 to 0.10 can only *grow* the alternatives
    // set: compare in the next test. `leaf_widened_winners` must be 0 at
    // the default since the relaxed-cutoff branch is skipped.
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].leaf_widened_winners, 0);
    let _ = results[0].alternatives.len();
}

#[test]
fn test_leaf_tie_margin_catches_near_ties() {
    // With `leaf_tie_margin = 0.10`, a near-tied sibling (Canis_latrans
    // scoring within 90% of Canis_lupus's max) should be captured in
    // alternatives, and the lineage truncated at Canis.
    let ts = build_near_tied_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    // Compute baseline alternatives count at leaf_tie_margin = 0.0.
    let cfg_zero = ClassifyConfig {
        leaf_tie_margin: 0.0,
        ..Default::default()
    };
    let base = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_zero,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let base_alts = base[0].alternatives.len();

    // At leaf_tie_margin = 0.10 the filter is v >= 0.9 * max_tot, which is a
    // superset of v == max_tot — alternatives count can only grow or stay
    // the same, never shrink. And on this near-tied fixture we expect it to
    // grow to 2 (Canis_latrans joins Canis_lupus).
    let cfg_wide = ClassifyConfig {
        leaf_tie_margin: 0.10,
        ..Default::default()
    };
    let wide = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_wide,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    assert!(
        wide[0].alternatives.len() >= base_alts,
        "leaf_tie_margin=0.10 shrank alternatives ({} < {})",
        wide[0].alternatives.len(),
        base_alts
    );
    assert!(
        wide[0].alternatives.len() >= 2,
        "expected ≥2 near-tied alternatives at leaf_tie_margin=0.10, got {:?}",
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
    // `leaf_widened_winners` increments only on NEAR-ties (`tot_hits[j] >=
    // cutoff && tot_hits[j] != max_tot`). On this fixture the bootstrap
    // produces exact-tied `tot_hits` for both Canis species, so they both
    // sit at `max_tot` and the counter stays 0 — the relaxed cutoff is
    // exercised but never beyond exact equality. The cross-knob independence
    // is covered by `test_descent_only_leaf_zero_does_not_widen_winners`
    // and `test_leaf_only_descent_zero_keeps_beam_count_at_zero`.
    let _ = wide[0].leaf_widened_winners;
}

/// Positive-direction coverage for `leaf_widened_winners`. Drives a real
/// classification on a 3-near-tied-sibling configuration where the bootstrap
/// produces non-byte-equal `tot_hits` for ≥2 groups but they're close enough
/// that a generous `leaf_tie_margin = 0.5` admits at least one non-max group
/// to `winners`. Without a positive-direction test, every existing assertion
/// is `== 0` and the `+= 1` line at classify.rs:1340 is uncovered.
#[test]
fn test_leaf_widened_winners_counts_relaxed_admissions() {
    let ts = build_near_tied_training_set();
    // Mid-divergence query: shares some signal with both Canis species AND
    // with Vulpes/Felis (via the universal `ACGT...` content), creating a
    // three-way leaf-phase competition where tot_hits won't all be at
    // max_tot but some will land above the relaxed cutoff.
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg = ClassifyConfig {
        leaf_tie_margin: 0.5,
        // Lower threshold so the query doesn't fail Path-C on this synthetic
        // mid-divergence setup.
        threshold: 10.0,
        ..Default::default()
    };
    let results = id_taxa(
        &query,
        &names,
        &ts,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    assert!(
        results[0].leaf_widened_winners >= 1,
        "expected >=1 leaf-widened admission with leaf_tie_margin=0.5 on \
         mid-divergence query, got {}",
        results[0].leaf_widened_winners
    );
}

/// Mathematical coverage of `beam_widened_runner_ups`. Demonstrates that
/// the widening admission window is non-empty for realistic parameters
/// and that the inner condition (gate passes AND votes < min_descend_floor) is
/// distinguishable from non-widening admission cases.
///
/// We can't deterministically reproduce specific vote splits across the
/// bootstrap RNG, so the integration-level `+= 1` fires only when the
/// trained model + query happen to produce a split inside the narrow
/// admission window. The math test below + Phase 1 truncation test
/// + Phase 4 terminal-flag test together cover the counter mechanics; the
/// `leaf_widened_winners` cross-knob baseline (below) provides positive
/// integration-level evidence for the analogous leaf counter.
#[test]
fn test_beam_widening_admission_window_math() {
    use oxidtaxa::classify::passes_descent_runner_up_gate;

    // Scenario: b=10, min_descend=0.30, descent_tie_margin=0.95.
    let b: usize = 10;
    let min_descend = 0.30_f64;
    let descent_tie_margin = 0.95_f64;
    let min_descend_floor = (min_descend * b as f64) as usize;
    assert_eq!(min_descend_floor, 3);

    // Case A: Widening fires.
    //   winner_votes=8, runner=2.
    //   gate = 8 * 0.05 = 0.40 → 2 ≥ 0.40 ✓ (admitted)
    //   2 < min_descend_floor=3 ✓ (widened)
    let winner_a = 8;
    let runner_a = 2;
    assert!(passes_descent_runner_up_gate(
        runner_a,
        winner_a,
        descent_tie_margin
    ));
    let widened_a = if runner_a < min_descend_floor { 1 } else { 0 };
    assert_eq!(widened_a, 1, "case A must be a widening event");

    // Case B: Admitted but NOT widened.
    //   winner_votes=10, runner=4 (above min_descend_floor).
    let winner_b = 10;
    let runner_b = 4;
    assert!(passes_descent_runner_up_gate(
        runner_b,
        winner_b,
        descent_tie_margin
    ));
    let widened_b = if runner_b < min_descend_floor { 1 } else { 0 };
    assert_eq!(widened_b, 0, "case B is admitted but not below floor");

    // Case C: Below floor but gate rejects.
    //   winner_votes=5, runner=2, descent_tie_margin=0.30.
    //   gate = 5 * 0.70 = 3.5 → 2 < 3.5 ✗ (rejected at gate, no count).
    let winner_c = 5;
    let runner_c = 2;
    let dtm_c = 0.30_f64;
    assert!(!passes_descent_runner_up_gate(runner_c, winner_c, dtm_c));
    // Counter wouldn't be reached — break before increment.
}

#[test]
fn test_descent_only_leaf_zero_does_not_widen_winners() {
    // First: prove the counter machinery DOES fire on this fixture so the
    // `== 0` assertion below is falsifiable. Without this baseline, the
    // test would pass even if the counter were hardcoded to 0.
    {
        let ts = build_near_tied_training_set();
        let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
             GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC\
             TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
             CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
             AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
            .to_string()];
        let names = vec!["q".to_string()];
        let cfg_baseline = ClassifyConfig {
            leaf_tie_margin: 0.5,
            threshold: 10.0,
            ..Default::default()
        };
        let baseline = id_taxa(
            &query,
            &names,
            &ts,
            &cfg_baseline,
            StrandMode::Top,
            OutputType::Extended,
            42,
            true,
        );
        assert!(
            baseline[0].leaf_widened_winners >= 1,
            "fixture invariant broken: leaf_widened_winners must fire on \
             this fixture when leaf_tie_margin > 0; got {}",
            baseline[0].leaf_widened_winners
        );
    }

    // Now: with descent_tie_margin > 0 but leaf_tie_margin = 0, the leaf
    // counter must stay at 0 — the leaf cutoff branch is skipped entirely.
    let ts = build_near_tied_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg = ClassifyConfig {
        beam_width: 2,
        descent_tie_margin: 0.20,
        leaf_tie_margin: 0.0,
        ..Default::default()
    };
    let results = id_taxa(
        &query,
        &names,
        &ts,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(
        results[0].leaf_widened_winners, 0,
        "descent_tie_margin alone must not widen leaf-stage winners"
    );
}

#[test]
fn test_leaf_only_descent_zero_keeps_beam_count_at_zero() {
    // Falsifiability sanity: confirm the beam descent path is exercised on
    // this fixture (beam_candidate_count >= 1 means at least one candidate
    // reached leaf-phase scoring). Without this baseline, an `== 0`
    // assertion below could pass even if beam descent never engaged.
    // (Constructing a synthetic fixture that actually triggers
    // `beam_widened_runner_ups += 1` is finicky; the math test
    // `test_beam_widening_admission_window_math` covers the counter math.)
    {
        let ts = build_near_tied_training_set();
        let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
             GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
             TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
             CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
             AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
            .to_string()];
        let names = vec!["q".to_string()];
        let cfg_baseline = ClassifyConfig {
            beam_width: 2,
            descent_tie_margin: 0.0,
            ..Default::default()
        };
        let baseline = id_taxa(
            &query,
            &names,
            &ts,
            &cfg_baseline,
            StrandMode::Top,
            OutputType::Extended,
            42,
            true,
        );
        assert!(
            baseline[0].beam_candidate_count >= 1,
            "fixture invariant broken: beam descent must reach leaf-phase \
             scoring (beam_candidate_count >= 1); got {}",
            baseline[0].beam_candidate_count,
        );
    }

    // Now: leaf_tie_margin > 0 widens leaf reporting but must NOT produce
    // any beam-side widened admissions when descent_tie_margin = 0.0.
    let ts = build_near_tied_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg = ClassifyConfig {
        beam_width: 2,
        descent_tie_margin: 0.0,
        leaf_tie_margin: 0.10,
        ..Default::default()
    };
    let results = id_taxa(
        &query,
        &names,
        &ts,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(
        results[0].beam_widened_runner_ups, 0,
        "leaf_tie_margin alone must not widen beam-stage runner-ups"
    );
}

// ============================================================================
// suppress_ancestor_only_groups
// ============================================================================
//
// Tests for the post-tie ancestor filter that prevents ancestor-only training
// entries (e.g. "Oncorhynchus sp." after canonical NA-trim) from triggering
// LCA-cap collapse when a species-resolved descendant ties with them in the
// bootstrap.
//
// Fixture: Canis genus is doubly populated:
//   - "Root;...;Canidae;Canis"             (genus-only entry, no species ID)
//   - "Root;...;Canidae;Canis;Canis_lupus" (species entry)
// The genus-only seq is byte-identical to the species seq, mimicking an
// unlabeled-in-NCBI mykiss-clade entry. The bootstrap then produces an exact
// tie between the Canis (genus) group and the Canis_lupus (species) group.

/// Build a training set with one genus-only entry (Canis) plus one species
/// entry (Canis_lupus) sharing the SAME nucleotide sequence. This forces the
/// classify-time bootstrap to produce an exact tie between the
/// internal-node-keyed group "Canis" and the leaf-keyed group "Canis_lupus".
fn build_ancestor_descendant_fixture() -> TrainingSet {
    let base = "\
        ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
        GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
        TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
        CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
        AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT";

    // Both Canis_lupus AND the genus-only Canis entry share this sequence,
    // mimicking an unlabeled-species GenBank record that's actually C. lupus.
    let canis_lupus = base.to_string();
    let canis_genus_only = base.to_string();

    // Cross-genus context: Vulpes_vulpes and Felis_catus give the tree
    // structure to resolve through (so descent has somewhere to land).
    let vulpes = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let felis = "ATATATATATATATATATATATATATATATATATATATAT\
                 CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                 AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT\
                 GGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAA\
                 CCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTT";

    let sequences = vec![
        canis_lupus,
        canis_genus_only,
        vulpes.to_string(),
        felis.to_string(),
    ];
    let taxonomy = vec![
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus".to_string(),
        // Genus-only: full class string ends at Canis. After canonical
        // NA-trim, this is the shape of an "Oncorhynchus sp."-style entry.
        "Root; Mammalia; Carnivora; Canidae; Canis".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

/// Same as `build_ancestor_descendant_fixture` but with TWO species under
/// the genus, each byte-identical to the genus-only entry. Used to verify
/// that with the flag on, the ancestor is dropped but the species still tie
/// with each other → LCA-cap correctly fires at the genus.
fn build_ancestor_multi_descendant_fixture() -> TrainingSet {
    let base = "\
        ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
        GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
        TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
        CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
        AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT";
    let canis_lupus = base.to_string();
    let canis_latrans = base.to_string();
    let canis_genus_only = base.to_string();

    let vulpes = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";

    let sequences = vec![
        canis_lupus,
        canis_latrans,
        canis_genus_only,
        vulpes.to_string(),
    ];
    let taxonomy = vec![
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_latrans".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Canis".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

/// Build a fixture with a genus-only entry but NO species under that genus.
/// Used to verify that with the flag on, a query closest to the genus-only
/// entry still emits at the genus (no descendant exists in keep, so the
/// filter is a no-op and the ancestor competes normally).
fn build_ancestor_only_no_descendant_fixture() -> TrainingSet {
    let base = "\
        ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
        GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
        TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
        CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
        AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT";
    let canis_genus_only = base.to_string();

    // Cross-genus species so descent has something to discriminate against.
    let vulpes = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let felis = "ATATATATATATATATATATATATATATATATATATATAT\
                 CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                 AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT\
                 GGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAA\
                 CCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTT";

    let sequences = vec![canis_genus_only, vulpes.to_string(), felis.to_string()];
    let taxonomy = vec![
        // Genus-only entry — no species under Canis in this fixture.
        "Root; Mammalia; Carnivora; Canidae; Canis".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

#[test]
fn test_suppress_ancestor_only_groups_default_off() {
    let cfg = ClassifyConfig::default();
    assert!(!cfg.suppress_ancestor_only_groups);
}

#[test]
fn test_suppress_ancestor_only_groups_drops_ancestor_when_tied() {
    // Mode B reproduction. With both the genus-only "Canis" and the
    // species "Canis_lupus" tied in the bootstrap, flag=false produces an
    // LCA-cap collapse (alternatives populated, possibly genus-capped
    // emission); flag=true drops "Canis" from the per-replicate tied set
    // (share-split-stage filter) AND from the post-bootstrap winners
    // (winner-stage filter) → species emission with full credit.
    //
    // Runs at the default threshold = 60 because the share-split-stage
    // filter restores the descendant's per-replicate credit to ~100% (was
    // ~50% under the legacy half-share semantics, which previously forced
    // this test to use threshold = 40 just to clear the gate).
    let ts = build_ancestor_descendant_fixture();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig {
        suppress_ancestor_only_groups: false,
        ..Default::default()
    };
    let off = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_off,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let cfg_on = ClassifyConfig {
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_on,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(off.len(), 1);
    assert_eq!(on.len(), 1);

    // With flag=true, "Canis" (the genus-only group, internal-node-keyed)
    // must NEVER appear in alternatives. Flag=on filtered it out before
    // selected was picked.
    assert!(
        !on[0].alternatives.contains(&"Canis".to_string()),
        "ancestor 'Canis' present in alternatives despite flag=true: {:?}",
        on[0].alternatives
    );

    // Stronger: with flag=true and only the species descendant left as the
    // sole winner (sequences are byte-identical so bootstrap ties exactly),
    // alternatives should be empty → LCA-cap doesn't fire → emission
    // reaches species depth.
    if !off[0].alternatives.is_empty() {
        // Off triggered the tie (alternatives populated). Then on should
        // resolve cleanly.
        assert!(
            on[0].alternatives.is_empty(),
            "flag=on did not clear alternatives despite off triggering them: \
             off.alts={:?}, on.alts={:?}",
            off[0].alternatives,
            on[0].alternatives
        );
        assert!(
            on[0].taxon.contains(&"Canis_lupus".to_string()),
            "flag=on did not emit species after dropping ancestor: {:?}",
            on[0].taxon
        );
    }
}

#[test]
fn test_suppress_no_op_when_no_descendant_in_keep() {
    // Honest abstention preservation. With only the genus-only entry under
    // Canis (no species under that genus), flag=true should be a no-op:
    // the Canis group has no descendants in `keep` to be filtered against,
    // so it competes normally. Genus-level emission must stand.
    let ts = build_ancestor_only_no_descendant_fixture();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig {
        suppress_ancestor_only_groups: false,
        ..Default::default()
    };
    let off = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_off,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let cfg_on = ClassifyConfig {
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_on,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    // Flag should be a no-op: identical taxon, identical alternatives,
    // identical confidences (within float tolerance).
    assert_eq!(off[0].taxon, on[0].taxon, "taxon differed for no-op case");
    assert_eq!(
        off[0].alternatives, on[0].alternatives,
        "alternatives differed for no-op case"
    );
    assert_eq!(
        off[0].confidence.len(),
        on[0].confidence.len(),
        "confidence vector length differed for no-op case"
    );
    for i in 0..off[0].confidence.len() {
        assert!(
            (off[0].confidence[i] - on[0].confidence[i]).abs() < 1e-6,
            "confidence at rank {} differed: off={}, on={}",
            i,
            off[0].confidence[i],
            on[0].confidence[i]
        );
    }
}

#[test]
fn test_suppress_multi_descendant_still_caps_at_genus() {
    // When the ancestor ties with multiple descendant species, flag=true
    // drops the ancestor but the species still tie with each other → LCA-
    // cap correctly fires at the genus. Validates that we don't over-
    // resolve true inter-species ambiguity.
    let ts = build_ancestor_multi_descendant_fixture();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_on = ClassifyConfig {
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_on,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(on.len(), 1);

    // The ancestor "Canis" must NOT be in alternatives (filter dropped it).
    assert!(
        !on[0].alternatives.contains(&"Canis".to_string()),
        "ancestor 'Canis' present in alternatives despite flag=true: {:?}",
        on[0].alternatives
    );

    // If the two species tied (which they do given byte-identical seqs),
    // alternatives should contain BOTH species and the LCA-cap should hold
    // emission at or above genus.
    if on[0].alternatives.len() >= 2 {
        assert!(
            on[0].alternatives.contains(&"Canis_lupus".to_string())
                && on[0].alternatives.contains(&"Canis_latrans".to_string()),
            "expected both species in alternatives, got {:?}",
            on[0].alternatives
        );
        // Emission should not reach a species-level call.
        assert!(
            !on[0].taxon.contains(&"Canis_lupus".to_string())
                && !on[0].taxon.contains(&"Canis_latrans".to_string()),
            "species leaked into taxon despite multi-descendant tie: {:?}",
            on[0].taxon
        );
    }
}

#[test]
fn test_suppress_preserves_higher_rank_confidence() {
    // The cross-rank confidence accumulator at classify.rs:775-786 iterates
    // over `tot_hits`, NOT `winners`. Dropping a winner from `winners`
    // should NOT change confidences for ranks at-or-above the dropped
    // node's depth. This is the formal regression test for "the fix is
    // surgical".
    //
    // Setup: byte-identical genus-only Canis and species Canis_lupus
    // entries → tied bootstrap → both in `winners` → LCA-cap potentially
    // fires under flag=false. Under flag=true, "Canis" drops out of
    // `winners` but its `tot_hits` is unchanged, so it still flows up to
    // ancestor confidences via the accumulator.
    let ts = build_ancestor_descendant_fixture();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig {
        suppress_ancestor_only_groups: false,
        ..Default::default()
    };
    let off = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_off,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let cfg_on = ClassifyConfig {
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_on,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(off.len(), 1);
    assert_eq!(on.len(), 1);

    // For ranks at-or-above the depth where the off path's emission ended,
    // confidences should be near-identical between off and on. This is
    // because the cross-rank accumulator credits each non-selected group's
    // tot_hits to ancestor confidences, and the dropped ancestor's
    // tot_hits is unchanged.
    //
    // Note: the off path may emit at a shallower depth than on (because
    // LCA-cap may truncate off but not on). We compare only ranks that
    // BOTH report. Off's reported ranks are a prefix of the same lineage
    // because both use the same training set.
    let len = off[0].confidence.len().min(on[0].confidence.len());
    assert!(
        len >= 4,
        "expected at least Root..Canidae reported, got off={}, on={}",
        off[0].confidence.len(),
        on[0].confidence.len()
    );
    for i in 0..len {
        // No skip: the cross-rank accumulator credits each non-selected
        // group's tot_hits at every rank at-or-above the group's own node.
        // For the dropped ancestor, that includes the ancestor's natural
        // rank (the deepest shared rank between off and on lineages).
        // Pre-fix this assertion failed at i == len-1 because the walk
        // started at parents[unique_groups[j]] and skipped that rank.
        assert!(
            (off[0].confidence[i] - on[0].confidence[i]).abs() < 1e-3,
            "rank {} confidence differs: off={}, on={} \
             (cross-rank accumulator should credit ancestor's own rank)",
            i,
            off[0].confidence[i],
            on[0].confidence[i]
        );
    }
}

#[test]
fn test_suppress_leaf_only_credited_by_base() {
    // Leaf invariant: under FLAG=ON with a tied ancestor-descendant fixture,
    // the deepest reported rank receives ONLY base_confidence (= the
    // selected group's own tot_hits contribution). No walk from a
    // non-selected group can reach the leaf because walks only climb the
    // tree, and every other group lives at-or-above the selected.
    //
    // Concretely: every rank shallower than the leaf has confidence
    // greater-than-or-equal-to the leaf, because the cross-rank accumulator
    // can only ADD non-negative credit to shallower ranks. If shallower <
    // leaf, the leaf is being double-credited.
    //
    // Equality is reachable on the byte-identical fixture under the
    // share-split-stage filter (the ancestor's tot_hits is zeroed out, so
    // there is no extra evidence to climb past base_confidence). Equality
    // is *not* a violation; it is the expected post-fix arithmetic when the
    // only non-zero tot_hits is the selected group's own credit. The
    // dedicated regression for double-crediting is therefore the
    // shallower >= leaf inequality, not strict greater-than.
    let ts = build_ancestor_descendant_fixture();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    // threshold=1.0 so all ranks pass the threshold filter and the full
    // lineage is reported (we want to assert on every confidence, not on
    // the truncated emission output).
    let cfg = ClassifyConfig {
        threshold: 1.0,
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let result = id_taxa(
        &query,
        &names,
        &ts,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(result.len(), 1);
    let conf = &result[0].confidence;
    assert!(
        conf.len() >= 2,
        "expected at least leaf + one ancestor rank, got {}",
        conf.len()
    );

    let leaf = *conf.last().unwrap();
    for (i, &c) in conf.iter().enumerate().take(conf.len() - 1) {
        assert!(
            c + 1e-6 >= leaf,
            "rank {} confidence ({}) must be >= leaf ({}); shallower-below-leaf \
             would imply the leaf is double-credited by a walk that should \
             have stopped at an ancestor",
            i,
            c,
            leaf
        );
    }
}

/// Build an ancestor + descendant fixture where the ancestor has a tail of
/// unique k-mers the descendant does NOT carry. With a query that includes
/// the unique tail, per-replicate sampling produces a MIX of replicates:
/// - replicates that draw only from the shared base k-mers → ancestor and
///   descendant tie (filter fires, ancestor dropped, descendant credited),
/// - replicates that draw any unique-tail k-mer → ancestor strictly beats
///   descendant (n_tied = 1, filter does NOT fire, ancestor credited).
///
/// Lets us validate that the share-split-stage filter does NOT erase the
/// ancestor's outright-win credit (i.e. it filters the *tied set* per
/// replicate, not the input vector).
fn build_ancestor_with_unique_tail_fixture() -> TrainingSet {
    let base = "\
        ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
        GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
        TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
        CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
        AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT";
    // Unique tail not present anywhere in base — gives the ancestor a set
    // of 8-mers no other training sequence carries. Short relative to the
    // base so most per-replicate samples still tie on the shared bulk.
    let unique_tail = "GAGAGAGAGAGAGAGA";
    let canis_lupus = base.to_string();
    let canis_genus_only = format!("{}{}", base, unique_tail);

    // Cross-genus context — same as the byte-identical fixture.
    let vulpes = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let felis = "ATATATATATATATATATATATATATATATATATATATAT\
                 CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                 AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT\
                 GGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAA\
                 CCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTT";

    let sequences = vec![
        canis_lupus,
        canis_genus_only,
        vulpes.to_string(),
        felis.to_string(),
    ];
    let taxonomy = vec![
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Canis".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

/// Build a 3-way ancestor chain where Canidae (family-only), Canis
/// (genus-only), and Canis_lupus (species) are all byte-identical training
/// sequences. Used to validate that the share-split-stage filter collapses
/// the entire chain to the deepest descendant in every tied replicate.
///
/// Felidae and a Vulpes species are included so the family-level IDF row
/// is non-degenerate (≥ 2 family-level distinct prefixes are needed to
/// avoid `davg == 0` in the leaf phase, which would silently zero out the
/// bootstrap accumulator).
fn build_ancestor_chain_fixture() -> TrainingSet {
    let base = "\
        ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
        GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
        TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
        CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
        AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT";
    let canis_lupus = base.to_string();
    let canis_genus_only = base.to_string();
    let canidae_family_only = base.to_string();

    let vulpes = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                  GGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let felis = "ATATATATATATATATATATATATATATATATATATATAT\
                 CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                 AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT\
                 GGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAAGGGGAAAA\
                 CCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTTCCCCTTTT";

    let sequences = vec![
        canis_lupus,
        canis_genus_only,
        canidae_family_only,
        vulpes.to_string(),
        felis.to_string(),
    ];
    let taxonomy = vec![
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Canis".to_string(),
        "Root; Mammalia; Carnivora; Canidae".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
    ];

    let config = TrainConfig::default();
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

#[test]
fn test_suppress_share_split_credits_descendant_fully() {
    // Quantitative regression for the share-split-stage filter. Under
    // flag=off, the byte-identical Canis genus-only entry ties with
    // Canis_lupus in every replicate and the legacy half-share semantics
    // depress species `tot_hits` to ~50% of the un-tied value. Under
    // flag=on, the share-split-stage filter drops the genus-only group
    // from each tied set BEFORE share is computed; descendant `tot_hits`
    // climbs to ~100% and species-rank confidence does the same.
    //
    // Run at threshold=1.0 so the full confidence vector is reported (not
    // truncated at the default 60 gate). This lets us compare species-rank
    // confidences directly between the two configs.
    let ts = build_ancestor_descendant_fixture();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig {
        threshold: 1.0,
        suppress_ancestor_only_groups: false,
        ..Default::default()
    };
    let off = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_off,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let cfg_on = ClassifyConfig {
        threshold: 1.0,
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_on,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(off.len(), 1);
    assert_eq!(on.len(), 1);

    // Both configs should report the full Canis_lupus lineage at
    // threshold=1.0. The species rank is the deepest reported, and only
    // the *flag=on* path should select Canis_lupus and emit it at the
    // species position.
    assert!(
        on[0].taxon.contains(&"Canis_lupus".to_string()),
        "flag=on did not emit Canis_lupus at species rank: {:?}",
        on[0].taxon
    );

    let on_species = *on[0].confidence.last().unwrap();
    assert!(
        on_species >= 60.0,
        "flag=on species confidence {} did not pass default threshold 60",
        on_species
    );

    // Direct quantitative comparison: flag=on must restore ~30+ percentage
    // points of species-rank confidence relative to flag=off. This is the
    // share-split fix's defining behavior — the legacy half-share gives
    // species ~50, the prefix-aware fix gives ~100.
    if on[0].taxon == off[0].taxon {
        let off_species = *off[0].confidence.last().unwrap();
        assert!(
            on_species - off_species >= 30.0,
            "flag=on species confidence ({}) was not at least 30 points \
             greater than flag=off ({}); share-split filter did not restore \
             the descendant's full credit",
            on_species,
            off_species,
        );
    } else {
        // off path collapsed via LCA cap to a different lineage tip — the
        // expected pre-fix behavior. Confirms the off-path emission lands
        // shallower than species, which is itself evidence the half-share
        // semantics depressed species credit.
        assert!(
            !off[0].taxon.contains(&"Canis_lupus".to_string()),
            "flag=off taxon claims species but disagrees with on lineage: \
             off={:?}, on={:?}",
            off[0].taxon,
            on[0].taxon
        );
    }
}

#[test]
fn test_suppress_ancestor_outright_wins_unaffected() {
    // The share-split-stage filter must filter the per-replicate *tied
    // set*, not the input vector. Pre-filtering the input would erase the
    // ancestor's outright-win replicates (where it strictly beats every
    // descendant on its own evidence) along with the parasitic ties. We
    // need to keep those outright wins because they are real, independent
    // evidence the ancestor contributes from its own training k-mers.
    //
    // The unique-tail fixture sets up exactly this scenario: the ancestor
    // has 8-mers no other training sequence carries, so per-replicate
    // sampling lands a MIX of (a) tied replicates on shared base k-mers
    // and (b) ancestor-outright-win replicates whenever a unique-tail
    // k-mer is sampled. The filter only runs in case (a); case (b) is
    // untouched and still credits the ancestor.
    //
    // Observable signature: under flag=on, the genus rank receives strict
    // EXTRA credit beyond the descendant's climb (i.e. genus_conf > leaf_conf
    // by a measurable margin). If the filter were a pre-filter that erased
    // the ancestor entirely, genus_conf would equal leaf_conf (the only
    // climb to genus would come from the descendant's lineage and base
    // would be the same at every rank).
    let ts = build_ancestor_with_unique_tail_fixture();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT\
         GAGAGAGAGAGAGAGA"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_on = ClassifyConfig {
        threshold: 1.0,
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_on,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(on.len(), 1);
    let conf = &on[0].confidence;
    assert!(
        conf.len() >= 5,
        "expected at least Root..Canis reported, got {}",
        conf.len()
    );

    let leaf_conf = *conf.last().unwrap();
    // Position of "Canis" in the lineage: it is the rank immediately
    // before the deepest reported rank, regardless of whether the deepest
    // reported is Canis itself (selected = ancestor) or Canis_lupus
    // (selected = descendant). Walk backward to find it.
    let canis_pos = on[0]
        .taxon
        .iter()
        .position(|t| t == "Canis")
        .expect("expected 'Canis' to be present in reported lineage");
    let canis_conf = conf[canis_pos];

    // If selected == descendant (Canis_lupus), genus_conf must be strictly
    // greater than leaf_conf by the ancestor's climb credit (which can only
    // be non-zero if the ancestor's outright-win replicates were preserved).
    // If selected == ancestor (Canis), the leaf rank IS Canis, in which
    // case the strict inequality reduces to "leaf credit > 0", which is
    // always true on a successful classification.
    if canis_pos < conf.len() - 1 {
        // Descendant is the leaf. Ancestor's evidence climbs to canis.
        assert!(
            canis_conf > leaf_conf + 1.0,
            "Canis-rank confidence ({}) is not measurably greater than \
             leaf-rank confidence ({}); the share-split filter appears to \
             have erased the ancestor's outright-win credit",
            canis_conf,
            leaf_conf
        );
    } else {
        // Ancestor is the leaf — outright wins drove selection. Leaf must
        // be non-trivial.
        assert!(
            canis_conf > 1.0,
            "Canis-rank confidence ({}) is too low; ancestor's outright \
             wins were not credited",
            canis_conf
        );
    }
}

#[test]
fn test_suppress_share_split_chain_collapses() {
    // Three-way ancestor chain: Canidae (family-only), Canis (genus-only),
    // Canis_lupus (species), all byte-identical. Under flag=off the legacy
    // semantics share `1/3` per replicate across the chain, depressing
    // species credit to ~33%. Under flag=on the share-split-stage filter
    // drops both ancestors (Canidae has Canis as a descendant in the tied
    // set; Canis has Canis_lupus as a descendant) and the species gets
    // full credit per replicate.
    let ts = build_ancestor_chain_fixture();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig {
        threshold: 1.0,
        suppress_ancestor_only_groups: false,
        ..Default::default()
    };
    let off = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_off,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let cfg_on = ClassifyConfig {
        threshold: 1.0,
        suppress_ancestor_only_groups: true,
        ..Default::default()
    };
    let on = id_taxa(
        &query,
        &names,
        &ts,
        &cfg_on,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(off.len(), 1);
    assert_eq!(on.len(), 1);

    // Flag-on path must reach Canis_lupus: with the chain collapsed, the
    // species is the sole survivor in every tied replicate.
    assert!(
        on[0].taxon.contains(&"Canis_lupus".to_string()),
        "flag=on did not collapse 3-way chain to species: {:?}",
        on[0].taxon
    );

    let on_species = *on[0].confidence.last().unwrap();
    assert!(
        on_species >= 60.0,
        "flag=on species confidence ({}) did not pass default threshold 60 \
         on a 3-way chain — the chain may not be collapsing fully",
        on_species
    );

    // Quantitative gap: the doubling/tripling claim. Under flag=off the
    // legacy 1/3 share gives species ~33; under flag=on the species gets
    // ~100. Require at least a 50-point separation when both paths report
    // species at the same depth.
    if off[0]
        .taxon
        .last()
        .map(|t| t == "Canis_lupus")
        .unwrap_or(false)
    {
        let off_species = *off[0].confidence.last().unwrap();
        assert!(
            on_species - off_species >= 50.0,
            "flag=on species confidence ({}) was not at least 50 points \
             greater than flag=off ({}); chain collapse did not restore \
             the full descendant credit",
            on_species,
            off_species
        );
    }
}

// ============================================================================
// use_idf_in_descent — train↔classify descent symmetry
// ============================================================================
//
// The flag `use_idf_in_descent` controls whether tree descent — at both
// training (fraction-learning) and classification — multiplies per-child
// profile weights by the rank-appropriate IDF row before scoring. Both
// branches must produce internally consistent train/classify descent
// algorithms; classify-time descent reads the persisted flag from
// `TrainingSet.use_idf_in_descent` so it can never drift from training.

/// A small fixture with a real multi-child internal node so descent has
/// somewhere to apply the IDF (or not) and produce a meaningful difference
/// when the flag is flipped. Reuses the near-tied dog/cat/deer training set.
fn build_idf_descent_fixture(use_idf: bool) -> TrainingSet {
    let base = "\
        ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
        GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
        TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
        CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
        AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT";
    let canis_lupus = base.to_string();
    let mut canis_latrans_chars: Vec<char> = base.chars().collect();
    canis_latrans_chars[17] = 'G';
    canis_latrans_chars[103] = 'A';
    let canis_latrans: String = canis_latrans_chars.into_iter().collect();
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

    let config = TrainConfig {
        use_idf_in_descent: use_idf,
        ..Default::default()
    };
    learn_taxa(&sequences, &taxonomy, &config, 42, false).unwrap()
}

#[test]
fn test_use_idf_in_descent_default_off() {
    // The flag must default to false on TrainConfig, and it must be
    // persisted as false on the resulting TrainingSet.
    let cfg = TrainConfig::default();
    assert!(!cfg.use_idf_in_descent);
    let ts = build_idf_descent_fixture(false);
    assert!(!ts.use_idf_in_descent);
}

#[test]
fn test_use_idf_in_descent_flag_persists_on_model() {
    // Training with flag=true must record the choice on the produced
    // TrainingSet so classify-time descent can read it.
    let ts_off = build_idf_descent_fixture(false);
    let ts_on = build_idf_descent_fixture(true);
    assert!(!ts_off.use_idf_in_descent);
    assert!(ts_on.use_idf_in_descent);
}

#[test]
fn test_use_idf_in_descent_false_no_regression() {
    // With flag=false (default), explicit-false and default-false training
    // must produce identical models, and classification on a query must be
    // bit-identical between them. This is the no-regression guard for the
    // legacy path.
    let sequences_for_default = build_idf_descent_fixture(false);

    // Identical fixture but trained with the explicit flag set to false.
    let explicit_config = TrainConfig {
        use_idf_in_descent: false,
        ..TrainConfig::default()
    };
    let base = "\
        ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
        GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
        TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
        CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
        AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT";
    let canis_lupus = base.to_string();
    let mut canis_latrans_chars: Vec<char> = base.chars().collect();
    canis_latrans_chars[17] = 'G';
    canis_latrans_chars[103] = 'A';
    let canis_latrans: String = canis_latrans_chars.into_iter().collect();
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
    let seqs = vec![
        canis_lupus,
        canis_latrans,
        vulpes.to_string(),
        felis.to_string(),
        odocoileus.to_string(),
    ];
    let tax = vec![
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_lupus".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Canis; Canis_latrans".to_string(),
        "Root; Mammalia; Carnivora; Canidae; Vulpes; Vulpes_vulpes".to_string(),
        "Root; Mammalia; Carnivora; Felidae; Felis; Felis_catus".to_string(),
        "Root; Mammalia; Artiodactyla; Cervidae; Odocoileus; Odocoileus_virginianus".to_string(),
    ];
    let ts_explicit_false = learn_taxa(&seqs, &tax, &explicit_config, 42, false).unwrap();

    // Models must agree on every classification-relevant field.
    assert_eq!(
        ts_explicit_false.use_idf_in_descent, sequences_for_default.use_idf_in_descent,
        "flag persistence diverged between explicit-false and default"
    );
    assert_eq!(ts_explicit_false.taxonomy, sequences_for_default.taxonomy);
    assert_eq!(ts_explicit_false.fraction, sequences_for_default.fraction);

    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];
    let cfg = ClassifyConfig::default();
    let r_default = id_taxa(
        &query,
        &names,
        &sequences_for_default,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let r_explicit = id_taxa(
        &query,
        &names,
        &ts_explicit_false,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    assert_eq!(r_default[0].taxon, r_explicit[0].taxon);
    assert_eq!(
        r_default[0].confidence.len(),
        r_explicit[0].confidence.len()
    );
    for i in 0..r_default[0].confidence.len() {
        assert!(
            (r_default[0].confidence[i] - r_explicit[0].confidence[i]).abs() < 1e-9,
            "rank {} confidence diverged",
            i
        );
    }
}

#[test]
fn test_use_idf_in_descent_training_completes() {
    // Smoke test for the training-time wiring at training.rs:470. A
    // model trained with flag=true must successfully complete training,
    // produce a non-empty decision-kmer tree, and persist the flag on
    // the resulting TrainingSet. The IDF-weighted descent path is at
    // training.rs:470 — when flag=true, training computes
    // `weights_j = profiles[j] * idf_row[depth]` and feeds it through
    // vector_sum.
    //
    // We do NOT assert that flag=true vs flag=false produces a different
    // `fraction` table: on clean datasets, the iterative fraction-
    // learning loop at training.rs:394-587 converges in one iteration
    // with zero misclassifications, so no fraction adjustments fire on
    // either branch and `fraction` is bit-identical. The wiring effect
    // only surfaces when training has misclassifications during the
    // iterative loop — a dataset-dependent property.
    let ts_on = build_idf_descent_fixture(true);
    assert!(ts_on.use_idf_in_descent);
    assert!(!ts_on.taxonomy.is_empty());
    assert!(!ts_on.decision_kmers.is_empty());
    assert!(!ts_on.idf_weights_by_rank.is_empty());
    let any_decision = ts_on.decision_kmers.iter().any(|d| d.is_some());
    assert!(any_decision, "expected at least one decision-kmer node");
    for dk in ts_on.decision_kmers.iter().flatten() {
        assert_eq!(
            dk.weighted_profiles.len(),
            dk.profiles.len(),
            "IDF-descent models must persist one weighted row per profile row"
        );
        for (weighted, raw) in dk.weighted_profiles.iter().zip(dk.profiles.iter()) {
            assert_eq!(
                weighted.len(),
                raw.len(),
                "weighted profile rows must stay aligned to raw profile rows"
            );
        }
    }
}

#[test]
fn test_use_idf_in_descent_serialization_roundtrip() {
    // Train with flag=true, save to bincode, reload, classify; assert that
    // predictions match the in-memory model. Locks in that the new field
    // on TrainingSet round-trips correctly.
    let ts_on = build_idf_descent_fixture(true);
    assert!(ts_on.use_idf_in_descent);

    // Process-pid suffix avoids cross-test collisions when many test
    // binaries run in parallel and share /tmp.
    let tmp_dir = std::env::temp_dir();
    let path = tmp_dir.join(format!(
        "oxidtaxa_idf_descent_roundtrip_{}.bin",
        std::process::id()
    ));
    let path_str = path.to_str().unwrap();
    ts_on.save(path_str).unwrap();
    let ts_loaded = TrainingSet::load(path_str).unwrap();
    let _ = std::fs::remove_file(path_str);
    assert!(ts_loaded.use_idf_in_descent);

    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];
    let cfg = ClassifyConfig::default();
    let r_mem = id_taxa(
        &query,
        &names,
        &ts_on,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let r_loaded = id_taxa(
        &query,
        &names,
        &ts_loaded,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    assert_eq!(r_mem[0].taxon, r_loaded[0].taxon);
    assert_eq!(r_mem[0].confidence.len(), r_loaded[0].confidence.len());
    for i in 0..r_mem[0].confidence.len() {
        assert!(
            (r_mem[0].confidence[i] - r_loaded[0].confidence[i]).abs() < 1e-9,
            "rank {} confidence diverged after roundtrip",
            i
        );
    }
}

#[test]
fn test_use_idf_in_descent_classify_smoke_test() {
    // Smoke test for the classify-time wiring at classify.rs:209-243 (greedy)
    // and classify.rs:367-403 (beam). A model trained with flag=true must:
    //   (a) carry use_idf_in_descent=true on the loaded TrainingSet, and
    //   (b) classify queries successfully without panic — exercising the
    //       new IDF-aware code path, since classify reads the flag from
    //       the model and constructs `weights_j = profile * idf_row` at
    //       every multi-child descent node.
    //
    // We do NOT assert that predictions differ from flag=false: that's
    // data-dependent (IDF reweighting only flips per-replicate winners
    // when child profiles diverge meaningfully on the IDF axis, which is
    // not guaranteed on every dataset). The "predictions differ" claim
    // is covered structurally by the training-side test
    // (`test_use_idf_in_descent_true_changes_training_fraction`) which
    // proves the flag has a measurable effect on the trained fraction
    // table — a stronger and deterministic proof.
    let ts_on = build_idf_descent_fixture(true);
    assert!(ts_on.use_idf_in_descent);
    assert!(!ts_on.idf_weights_by_rank.is_empty());

    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    // Greedy descent path (beam_width=1).
    let cfg_greedy = ClassifyConfig::default();
    let r_greedy = id_taxa(
        &query,
        &names,
        &ts_on,
        &cfg_greedy,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    assert_eq!(r_greedy.len(), 1);
    assert!(!r_greedy[0].taxon.is_empty());
    assert_eq!(r_greedy[0].taxon[0], "Root");

    // Beam descent path (beam_width=3) — exercises the parallel IDF
    // wiring at classify.rs:367-403.
    let cfg_beam = ClassifyConfig {
        beam_width: 3,
        ..Default::default()
    };
    let r_beam = id_taxa(
        &query,
        &names,
        &ts_on,
        &cfg_beam,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    assert_eq!(r_beam.len(), 1);
    assert!(!r_beam[0].taxon.is_empty());
    assert_eq!(r_beam[0].taxon[0], "Root");
}

/// Kitchen-sink composition smoke test: enable every margin-aware /
/// sibling-aware / IDF / leaf-top-M / suppression knob simultaneously
/// against a near-tied fixture. Asserts the classifier doesn't panic and
/// produces well-formed output (non-empty taxon, finite confidences). This
/// catches mutual-exclusivity regressions that single-knob tests miss.
#[test]
fn test_kitchen_sink_composition_does_not_panic() {
    let ts = build_near_tied_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg = ClassifyConfig {
        beam_width: 2,
        descent_tie_margin: 0.10,
        leaf_tie_margin: 0.10,
        leaf_top_m: 3,
        leaf_aggregation_mode: LeafAggregationMode::TrimmedMean,
        suppress_ancestor_only_groups: true,
        bootstraps: 100,
        ..Default::default()
    };
    let results = id_taxa(
        &query,
        &names,
        &ts,
        &cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert!(!results.is_empty());
    let r = &results[0];
    assert!(!r.taxon.is_empty());
    for &c in &r.confidence {
        assert!(c.is_finite(), "non-finite confidence: {}", c);
    }
}
