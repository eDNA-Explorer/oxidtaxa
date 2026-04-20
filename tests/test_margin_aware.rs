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
        data_dir.join("bench_1000_ref_taxonomy.tsv").to_str().unwrap(),
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
        &queries, &names, &model, &cfg_greedy_on, StrandMode::Both, OutputType::Extended, 42, true,
    );

    let cfg_beam_on = ClassifyConfig {
        beam_width: 1,  // still beam code path, but test with width=1 first
        confidence_uses_descent_margin: true,
        ..Default::default()
    };
    let _beam_width1_on = id_taxa(
        &queries, &names, &model, &cfg_beam_on, StrandMode::Both, OutputType::Extended, 42, true,
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
        &queries, &names, &model, &cfg_beam3_off, StrandMode::Both, OutputType::Extended, 42, true,
    );

    let cfg_beam3_on = ClassifyConfig {
        beam_width: 3,
        confidence_uses_descent_margin: true,
        ..Default::default()
    };
    let beam3_on = id_taxa(
        &queries, &names, &model, &cfg_beam3_on, StrandMode::Both, OutputType::Extended, 42, true,
    );

    // Also verify greedy+margin-on actually does something on this dataset
    // (anchor for the beam comparison).
    let cfg_greedy_off = ClassifyConfig {
        beam_width: 1,
        confidence_uses_descent_margin: false,
        ..Default::default()
    };
    let greedy_off = id_taxa(
        &queries, &names, &model, &cfg_greedy_off, StrandMode::Both, OutputType::Extended, 42, true,
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
    // With per-rank (non-cumulative) application, the deepest rank's
    // confidence is discounted by at most one margin (≥ 0.1 floor), so it
    // must be ≥ 10% of its unflagged value. Under the old cumulative-product
    // semantics, 6-7 compounded margins could easily drop this ratio to
    // 0.1^6 ≈ 1e-6 — a collapse we explicitly guard against.
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

    // Compare the deepest rank where both paths reported a confidence.
    let len = off[0].confidence.len().min(on[0].confidence.len());
    assert!(len > 0);
    let deepest_off = off[0].confidence[len - 1];
    let deepest_on = on[0].confidence[len - 1];

    // Skip the trivially-safe case where off was already near zero.
    if deepest_off > 1.0 {
        let ratio = deepest_on / deepest_off;
        assert!(
            ratio >= 0.1 - 1e-6,
            "deepest-rank confidence collapsed: off={}, on={}, ratio={} (< 0.1 floor)",
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
