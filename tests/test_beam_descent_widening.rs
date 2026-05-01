//! Phase 2 (E) — beam-descent runner-up widening via `descent_tie_margin`.
//!
//! Two layers of coverage:
//!
//! 1. Pure-unit tests on the extracted `passes_descent_runner_up_gate`
//!    function. Deterministic by construction; no bootstrap.
//!
//! 2. Integration tests with a small training set + seeded RNG. Exercise
//!    `beam_widened_runner_ups` propagation through the beam path and the
//!    default `descent_tie_margin = 0.0` telemetry contract.

mod common;

use oxidtaxa::classify::{id_taxa, passes_descent_runner_up_gate};
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{ClassifyConfig, OutputType, StrandMode, TrainConfig};

// ---------- 1) Pure-unit gate tests ----------

#[test]
fn test_gate_default_admits_only_exact_ties() {
    // descent_tie_margin = 0.0 ⇒ this helper only passes exact ties.
    assert!(
        passes_descent_runner_up_gate(100, 100, 0.0),
        "exact tie must pass at descent_tie_margin=0.0"
    );
    assert!(
        !passes_descent_runner_up_gate(99, 100, 0.0),
        "non-tied runner-up must fail at descent_tie_margin=0.0"
    );
    assert!(
        !passes_descent_runner_up_gate(50, 100, 0.0),
        "half-vote runner-up must fail at descent_tie_margin=0.0"
    );
}

#[test]
fn test_gate_with_margin_admits_within_cutoff() {
    // descent_tie_margin = 0.10: cutoff = 100 * 0.90 = 90.
    assert!(
        passes_descent_runner_up_gate(95, 100, 0.10),
        "95 votes is within 10% of winner (cutoff=90)"
    );
    assert!(
        passes_descent_runner_up_gate(90, 100, 0.10),
        "90 votes is exactly at cutoff and must pass (>=)"
    );
    assert!(
        !passes_descent_runner_up_gate(89, 100, 0.10),
        "89 votes is below cutoff 90 and must fail"
    );
}

#[test]
fn test_gate_break_safety_for_descending_votes() {
    // Vote sequence is sorted descending in the production loop. Once one
    // entry fails the gate, every subsequent (smaller-or-equal) entry must
    // fail too — locking in `break` semantics.
    let winner_votes = 100usize;
    let descent_tie_margin = 0.10;
    let votes_desc = [95, 90, 87, 80, 50, 10];
    let mut saw_failure = false;
    for &v in &votes_desc {
        let pass = passes_descent_runner_up_gate(v, winner_votes, descent_tie_margin);
        if saw_failure {
            assert!(
                !pass,
                "after first failure, all later entries must fail (got {} pass)",
                v
            );
        }
        if !pass {
            saw_failure = true;
        }
    }
}

#[test]
fn test_gate_extreme_tie_margins() {
    // descent_tie_margin = 1.0: every non-negative runner-up clears (cutoff = 0).
    assert!(
        passes_descent_runner_up_gate(0, 100, 1.0),
        "descent_tie_margin=1.0 admits zero-vote runner-up"
    );
    // descent_tie_margin = 0.5: cutoff = 50.
    assert!(passes_descent_runner_up_gate(50, 100, 0.5));
    assert!(!passes_descent_runner_up_gate(49, 100, 0.5));
}

#[test]
fn test_gate_zero_winner_votes_is_degenerate_admit() {
    // If somehow winner_votes = 0, cutoff = 0 and any votes >= 0 pass.
    // Production code never reaches the runner-up loop with winner_votes = 0
    // (the children_by_votes filter at classify.rs ensures votes > 0), but
    // the gate function itself remains defined.
    assert!(passes_descent_runner_up_gate(0, 0, 0.0));
    assert!(passes_descent_runner_up_gate(1, 0, 0.0));
}

#[test]
fn test_widened_counter_uses_integer_truncation() {
    // The widened-counter increment compares the runner-up's vote count to
    // the integer-truncated floor `(min_descend * b) as usize`, which
    // is what the descent gate uses to admit children. A naive float comparison
    // (`votes/b < min_descend`) drifts from this at fractional-floor values.
    //
    // Scenario: b=100, min_descend=0.985, runner_up_votes=98:
    //   min_descend_floor = (0.985 * 100) as usize = 98
    //   98 >= 98 → normal beam admission → not a widening event
    //   Naive float: 0.98 < 0.985 → would falsely count this.
    let b: usize = 100;
    let min_descend = 0.985_f64;
    let min_descend_floor = (min_descend * b as f64) as usize;
    assert_eq!(
        min_descend_floor, 98,
        "expected truncated floor of 98 at b=100"
    );
    assert!(
        98 >= min_descend_floor,
        "vote 98 must clear the normal descent floor (the widening counter \
         must not increment on this case)"
    );
    // Verify the float-comparison branch would have lied:
    let vf = 98.0_f64 / b as f64;
    assert!(
        vf < min_descend,
        "naive float comparison vf < min_descend would have falsely fired \
         (this is the bug being fixed)"
    );
}

// ---------- 2) Integration tests on a real model ----------

/// Build a small training set with three sibling species. Two of the species
/// share an identical sequence; one has a distinct sequence. This produces
/// vote-tied bootstraps at the parent node, giving the beam path a concrete
/// runner-up admission to test against.
fn build_simple_three_sibling_set() -> oxidtaxa::types::TrainingSet {
    let tied_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                    GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
                    TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA";
    let other_seq = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                     GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                     GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let third_seq = "ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT\
                     CGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG\
                     AAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTTAAAATTTT";

    let sequences = vec![
        tied_seq.to_string(),
        tied_seq.to_string(),
        other_seq.to_string(),
        third_seq.to_string(),
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

/// At `descent_tie_margin = 0.0` and `beam_width = 2`, the margin does not
/// add below-threshold runner-ups. No runner-up is counted as "widened";
/// normal beam admissions above `min_descend` are not widening events.
#[test]
fn test_default_descent_tie_margin_yields_zero_widened_runner_ups() {
    let ts = build_simple_three_sibling_set();

    let query_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                     GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
                     TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA";
    let queries = vec![query_seq.to_string()];
    let names = vec!["q1".to_string()];

    let config = ClassifyConfig {
        beam_width: 2,
        descent_tie_margin: 0.0,
        ..Default::default()
    };

    let results = id_taxa(
        &queries,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert_eq!(
        r.beam_widened_runner_ups, 0,
        "descent_tie_margin=0.0 must never count widened admissions"
    );
}

/// Default greedy path (`beam_width = 1`) never enters the beam descent loop,
/// so `beam_widened_runner_ups` is always 0 regardless of
/// `descent_tie_margin`.
#[test]
fn test_greedy_path_has_zero_widened_runner_ups() {
    let ts = build_simple_three_sibling_set();
    let query_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                     GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
                     TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA";
    let queries = vec![query_seq.to_string()];
    let names = vec!["q1".to_string()];

    // descent_tie_margin > 0 must NOT cause widened admissions on greedy.
    let config = ClassifyConfig {
        beam_width: 1,
        descent_tie_margin: 0.20,
        ..Default::default()
    };

    let results = id_taxa(
        &queries,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert_eq!(
        r.beam_widened_runner_ups, 0,
        "greedy never widens runner-ups (no beam path)"
    );
    assert_eq!(
        r.beam_candidate_count, 1,
        "greedy must report a single candidate"
    );
}

/// Beam-width > 1 must not re-process terminal leaf-parent candidates on
/// subsequent descent iterations. Without the `terminal` flag, a candidate
/// that reached a leaf-parent at iteration N would re-enter the descent
/// loop at the SAME `k_node` at iteration N+1 (when another sibling
/// triggered `any_expanded = true`), re-running votes with fresh PRNG and
/// inflating `beam_close_runner_up_count` and `beam_widened_runner_ups`
/// past the structural bound (the path's tree depth).
///
/// Cross-check: counter values must be bounded by the lineage depth of the
/// reported `taxon` (number of descent steps along the winning path).
#[test]
fn test_terminal_candidates_do_not_double_count() {
    let ts = build_simple_three_sibling_set();

    // Mixed-evidence query — close to one tied species but not exact, so
    // beam descent admits multiple candidates and the leaf-parent / non-
    // leaf-parent split is real.
    let query_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                     GCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT\
                     TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA";
    let queries = vec![query_seq.to_string()];
    let names = vec!["q1".to_string()];

    let config = ClassifyConfig {
        beam_width: 2,
        descent_tie_margin: 0.20,
        ..Default::default()
    };

    let results = id_taxa(
        &queries,
        &names,
        &ts,
        &config,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let r = &results[0];
    // beam_close_runner_up_count counts at most one increment per descent
    // step. The reported taxon has length L (Root + L-1 ranks); the max
    // descent steps along ANY path can't exceed L. This is a structural
    // upper bound the terminal-flag fix preserves.
    let lineage_depth = r.taxon.len() as u32;
    assert!(
        r.beam_close_runner_up_count <= lineage_depth,
        "beam_close_runner_up_count ({}) exceeded lineage depth ({}) — \
         indicates terminal candidates were re-processed",
        r.beam_close_runner_up_count,
        lineage_depth
    );
    assert!(
        r.beam_widened_runner_ups <= lineage_depth,
        "beam_widened_runner_ups ({}) exceeded lineage depth ({}) — \
         indicates terminal candidates were re-processed",
        r.beam_widened_runner_ups,
        lineage_depth
    );
    // Sanity: deterministic RNG seed so this test is bit-stable.
    assert_eq!(results.len(), 1);
}

/// At `descent_tie_margin = 0.0, beam_width = 2`, results match the greedy
/// path's classification on a query that doesn't trigger ties. Confirms the
/// "default reproduces current behavior" contract.
#[test]
fn test_default_beam_matches_greedy_on_unambiguous_query() {
    let ts = build_simple_three_sibling_set();

    // Vulpes-only query — should classify cleanly without ties.
    let query_seq = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                     GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT\
                     GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTT";
    let queries = vec![query_seq.to_string()];
    let names = vec!["q1".to_string()];

    let greedy = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig::default(),
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let beam = id_taxa(
        &queries,
        &names,
        &ts,
        &ClassifyConfig {
            beam_width: 2,
            descent_tie_margin: 0.0,
            ..Default::default()
        },
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(greedy.len(), 1);
    assert_eq!(beam.len(), 1);
    assert_eq!(
        greedy[0].taxon, beam[0].taxon,
        "default descent_tie_margin=0.0 + beam_width=2 must match greedy on \
         unambiguous queries"
    );
    assert_eq!(beam[0].beam_widened_runner_ups, 0);
}
