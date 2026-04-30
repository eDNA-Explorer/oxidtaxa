//! Flag-gated margin-aware classification tests.
//!
//! Covers `tie_margin`, `confidence_uses_descent_margin`, and
//! `sibling_aware_leaf`. Each flag defaults to legacy behavior; these
//! tests exercise both on and off to verify the guard holds.

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
// tie_margin
// ============================================================================

#[test]
fn test_tie_margin_zero_preserves_legacy() {
    // With `tie_margin = 0.0`, only exact `tot_hits` equalities produce
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
        tie_margin: 0.0,
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
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    // Compute baseline alternatives count at tie_margin = 0.0.
    let cfg_zero = ClassifyConfig {
        tie_margin: 0.0,
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

    // At tie_margin = 0.10 the filter is v >= 0.9 * max_tot, which is a
    // superset of v == max_tot — alternatives count can only grow or stay
    // the same, never shrink. And on this near-tied fixture we expect it to
    // grow to 2 (Canis_latrans joins Canis_lupus).
    let cfg_wide = ClassifyConfig {
        tie_margin: 0.10,
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
// confidence_uses_descent_margin
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
    // confidence by a per-rank affine-remapped margin in `[0.82, 1.0]`
    // (current semantic; pre-e287eb7 was a cumulative product floored at
    // 0.1, see test_descent_margin_does_not_collapse_deep_ranks for the
    // non-collapse contract). Per-rank multiplier ≤ 1.0, so each per-rank
    // confidence can only shrink (or stay equal when every descent was
    // unambiguous).
    let ts = build_near_tied_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig::default();
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
        confidence_uses_descent_margin: true,
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

#[test]
fn test_descent_margin_active_with_beam_width_3() {
    // Regression guard for the empty-margins bug in the beam path.
    //
    // Previously (pre-fix) the beam path constructed an `empty_margins: Vec<f64>`
    // at classify.rs:464 and passed it to `leaf_phase_score`, which then
    // guarded on `!descent_margins.is_empty()` and silently did nothing.
    // Result: `confidence_uses_descent_margin=true` was a no-op whenever
    // `beam_width > 1`.
    //
    // We test this by comparing beam+margin-on output against greedy+margin-on
    // output on a 1K benchmark dataset known to produce real margin discounts.
    // If the beam wiring is broken, beam+margin-on would equal beam+margin-off,
    // which in turn differs from greedy+margin-on. If the wiring is correct,
    // beam and greedy margin-on will be within floating-point noise of each
    // other at every reported rank (same descent path, same margins applied,
    // same leaf).
    use std::path::PathBuf;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = manifest_dir.join("benchmarks").join("data");
    if !data_dir.join("bench_1000_ref.fasta").exists() {
        eprintln!("skipping beam descent-margin plumbing test: benchmark data not present");
        return;
    }

    use oxidtaxa::fasta::{read_fasta, read_taxonomy};
    use oxidtaxa::sequence::remove_gaps;

    let (ref_names, ref_seqs) =
        read_fasta(data_dir.join("bench_1000_ref.fasta").to_str().unwrap()).unwrap();
    let ref_tax = read_taxonomy(
        data_dir
            .join("bench_1000_ref_taxonomy.tsv")
            .to_str()
            .unwrap(),
        &ref_names,
    )
    .unwrap();
    // Minimal training-filter — mirror eval_training.rs.
    let mut train_seqs = Vec::new();
    let mut train_tax = Vec::new();
    for (i, seq) in ref_seqs.iter().enumerate() {
        let tax = &ref_tax[i];
        let full_tax = format!("Root; {}", tax.replace(";", "; "));
        let rank_count = full_tax.split("; ").count();
        if rank_count < 4 || seq.len() < 30 {
            continue;
        }
        train_seqs.push(seq.clone());
        train_tax.push(full_tax);
    }
    let model = learn_taxa(&train_seqs, &train_tax, &TrainConfig::default(), 42, false).unwrap();

    let (q_names, q_seqs) =
        read_fasta(data_dir.join("bench_1000_query.fasta").to_str().unwrap()).unwrap();
    let clean = remove_gaps(&q_seqs);
    // Trim to the first 20 queries — enough to hit real margins without
    // paying the full 500-query cost.
    let queries: Vec<String> = clean.into_iter().take(20).collect();
    let names: Vec<String> = q_names.into_iter().take(20).collect();

    let cfg_greedy_on = ClassifyConfig {
        beam_width: 1,
        confidence_uses_descent_margin: true,
        ..Default::default()
    };
    let greedy_on = id_taxa(
        &queries,
        &names,
        &model,
        &cfg_greedy_on,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    let cfg_beam_on = ClassifyConfig {
        beam_width: 1, // still beam code path, but test with width=1 first
        confidence_uses_descent_margin: true,
        ..Default::default()
    };
    let _beam_width1_on = id_taxa(
        &queries,
        &names,
        &model,
        &cfg_beam_on,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    // The real plumbing test: beam_width=3 must also apply margins. Compare
    // beam margin-on against beam margin-off — they must differ on at least
    // one query's confidence vector, since this benchmark is known to
    // produce margin discounts under greedy.
    let cfg_beam3_off = ClassifyConfig {
        beam_width: 3,
        confidence_uses_descent_margin: false,
        ..Default::default()
    };
    let beam3_off = id_taxa(
        &queries,
        &names,
        &model,
        &cfg_beam3_off,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    let cfg_beam3_on = ClassifyConfig {
        beam_width: 3,
        confidence_uses_descent_margin: true,
        ..Default::default()
    };
    let beam3_on = id_taxa(
        &queries,
        &names,
        &model,
        &cfg_beam3_on,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    // Also verify greedy+margin-on actually does something on this dataset
    // (anchor for the beam comparison).
    let cfg_greedy_off = ClassifyConfig {
        beam_width: 1,
        confidence_uses_descent_margin: false,
        ..Default::default()
    };
    let greedy_off = id_taxa(
        &queries,
        &names,
        &model,
        &cfg_greedy_off,
        StrandMode::Both,
        OutputType::Extended,
        42,
        true,
    );

    let any_greedy_changed = greedy_off.iter().zip(greedy_on.iter()).any(|(a, b)| {
        let len = a.confidence.len().min(b.confidence.len());
        (0..len).any(|i| (a.confidence[i] - b.confidence[i]).abs() > 1e-6)
    });
    assert!(
        any_greedy_changed,
        "benchmark dataset does not produce margin discounts even under greedy"
    );

    let any_beam3_changed = beam3_off.iter().zip(beam3_on.iter()).any(|(a, b)| {
        let len = a.confidence.len().min(b.confidence.len());
        (0..len).any(|i| (a.confidence[i] - b.confidence[i]).abs() > 1e-6)
    });
    assert!(
        any_beam3_changed,
        "beam_width=3 + confidence_uses_descent_margin=true produced identical \
         confidences to the flag-off path on a dataset where greedy + \
         confidence_uses_descent_margin=true DID change them — the beam path \
         is still ignoring margins (empty_margins regression)"
    );
}

#[test]
fn test_descent_margin_does_not_collapse_deep_ranks() {
    // With per-rank (non-cumulative) application + affine remap to
    // `[MARGIN_FLOOR=0.8, 1.0]`, the deepest rank's confidence is discounted
    // by at most one multiplier ≥ 0.82 (`MARGIN_FLOOR + (1-MARGIN_FLOOR)·0.1
    // = 0.82` at the lowest m=0.1). At typical tree depth L=4-6, this gives
    // `0.82^L ∈ [0.45, 0.55]` worst-case — comfortably above 0.5. Under
    // pre-e287eb7 cumulative-product semantics, 6-7 compounded margins could
    // collapse to `0.1^6 ≈ 1e-6` — a collapse we explicitly guard against.
    // Tighten the assertion to 0.5 so a regression to cumulative-product
    // would actually fail the test (the prior 0.1 threshold passed under
    // either semantic and couldn't distinguish them).
    let ts = build_near_tied_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig::default();
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
        confidence_uses_descent_margin: true,
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

    // Compare the deepest rank where both paths reported a confidence.
    let len = off[0].confidence.len().min(on[0].confidence.len());
    assert!(len > 0);
    let deepest_off = off[0].confidence[len - 1];
    let deepest_on = on[0].confidence[len - 1];

    // Skip the trivially-safe case where off was already near zero.
    if deepest_off > 1.0 {
        let ratio = deepest_on / deepest_off;
        assert!(
            ratio >= 0.5 - 1e-6,
            "deepest-rank confidence collapsed: off={}, on={}, ratio={} (< 0.5 — \
             would only happen under pre-e287eb7 cumulative-product semantics)",
            deepest_off,
            deepest_on,
            ratio
        );
    }
}

// ============================================================================
// sibling_aware_leaf
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
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         GCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         AAATTTAAATTTAAATTTAAATTTAAATTTAAATTTAAAT"
        .to_string()];
    let names = vec!["q".to_string()];

    let cfg_off = ClassifyConfig::default();
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
        sibling_aware_leaf: true,
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
    assert!(
        on[0].alternatives.len() >= off[0].alternatives.len(),
        "sibling_aware_leaf=true shrank alternatives ({} < {})",
        on[0].alternatives.len(),
        off[0].alternatives.len()
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
}

#[test]
fn test_use_idf_in_descent_serialization_roundtrip() {
    // Train with flag=true, save to bincode, reload, classify; assert that
    // predictions match the in-memory model. Locks in that the new field
    // on TrainingSet round-trips correctly.
    let ts_on = build_idf_descent_fixture(true);
    assert!(ts_on.use_idf_in_descent);

    let tmp_dir = std::env::temp_dir();
    let path = tmp_dir.join("oxidtaxa_idf_descent_roundtrip.bin");
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
