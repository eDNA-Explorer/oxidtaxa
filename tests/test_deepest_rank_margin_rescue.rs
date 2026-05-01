use oxidtaxa::classify::{
    id_taxa, passes_deepest_rank_margin_rescue, DeepestRankRescueDecisionInput, RescueRejection,
};
use oxidtaxa::training::learn_taxa;
use oxidtaxa::types::{
    ClassificationResult, ClassifyConfig, OutputType, StrandMode, TrainConfig, TrainingSet,
};

fn decision_input(top: f64, runner: f64) -> DeepestRankRescueDecisionInput {
    DeepestRankRescueDecisionInput {
        challenge_original_selection: top,
        challenge_runner: runner,
        selected_is_unique_challenge_top: true,
        challenge_exact_tie: false,
        challenge_near_tie: false,
        candidate_cap_exceeded: false,
        floor: Some(30.0),
        min_delta: 15.0,
        min_ratio: 2.0,
        candidate_count: 2,
    }
}

#[test]
fn deepest_rank_rescue_decision_accepts_exact_boundaries() {
    let input = decision_input(30.0, 15.0);
    assert_eq!(passes_deepest_rank_margin_rescue(&input), Ok(()));
}

#[test]
fn deepest_rank_rescue_decision_rejects_disabled_and_basic_gates() {
    let mut input = decision_input(40.0, 10.0);
    input.floor = None;
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::RescueDisabled)
    );

    let input = decision_input(29.99, 10.0);
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::FloorFailed)
    );

    let input = decision_input(40.0, 26.0);
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::DeltaFailed)
    );

    let mut input = decision_input(40.0, 24.0);
    input.min_delta = 10.0;
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::RatioFailed)
    );
}

#[test]
fn deepest_rank_rescue_decision_handles_zero_runner_with_epsilon() {
    let input = decision_input(30.0, 0.0);
    assert_eq!(passes_deepest_rank_margin_rescue(&input), Ok(()));

    let input = decision_input(30.0, f64::EPSILON / 2.0);
    assert_eq!(passes_deepest_rank_margin_rescue(&input), Ok(()));

    let mut input = decision_input(30.0, 0.0);
    input.candidate_count = 1;
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::CandidateCountLt2)
    );
}

#[test]
fn deepest_rank_rescue_decision_rejects_nonfinite_and_negative_scores_before_math() {
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let input = decision_input(v, 10.0);
        assert_eq!(
            passes_deepest_rank_margin_rescue(&input),
            Err(RescueRejection::NonfiniteChallengeScore)
        );

        let input = decision_input(40.0, v);
        assert_eq!(
            passes_deepest_rank_margin_rescue(&input),
            Err(RescueRejection::NonfiniteChallengeScore)
        );
    }

    let input = decision_input(0.0, 0.0);
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::NonpositiveChallengeTop)
    );

    let input = decision_input(40.0, -0.01);
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::NegativeChallengeRunner)
    );
}

#[test]
fn deepest_rank_rescue_decision_reason_precedence_is_stable() {
    let mut input = decision_input(f64::NAN, -10.0);
    input.candidate_count = 1;
    input.candidate_cap_exceeded = true;
    input.selected_is_unique_challenge_top = false;
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::CandidateCountLt2)
    );

    let mut input = decision_input(f64::NAN, -10.0);
    input.candidate_cap_exceeded = true;
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::ChallengeCandidateCapExceeded)
    );

    let mut input = decision_input(40.0, 10.0);
    input.selected_is_unique_challenge_top = false;
    input.challenge_exact_tie = true;
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::SelectedNotChallengeTop)
    );

    let mut input = decision_input(40.0, 10.0);
    input.challenge_exact_tie = true;
    input.challenge_near_tie = true;
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::ChallengeExactTie)
    );
}

#[test]
fn classify_config_default_and_rescue_validation() {
    assert!(ClassifyConfig::default().validate().is_ok());

    let mut cfg = ClassifyConfig {
        deepest_rank_margin_floor: Some(f64::NAN),
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    cfg = ClassifyConfig {
        deepest_rank_margin_floor: Some(101.0),
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    cfg = ClassifyConfig {
        deepest_rank_margin_min_delta: f64::INFINITY,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    cfg = ClassifyConfig {
        deepest_rank_margin_min_ratio: 0.99,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());

    cfg = ClassifyConfig {
        max_deepest_rank_challenge_candidates: Some(1),
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn deepest_rank_diagnostics_serde_defaults_and_round_trip() {
    let default_json = serde_json::to_string(&ClassificationResult::default()).unwrap();
    assert!(!default_json.contains("deepest_rank_acceptance_mode"));
    assert!(!default_json.contains("deepest_rank_challenge_margin_delta"));
    assert!(!default_json.contains("leaf_margin_candidate_count"));

    let old_json = r#"{"taxon":["Root"],"confidence":[0.0]}"#;
    let old: ClassificationResult = serde_json::from_str(old_json).unwrap();
    assert_eq!(old.deepest_rank_acceptance_mode, None);
    assert_eq!(old.deepest_rank_challenge_candidate_count, None);

    let mut populated = ClassificationResult::default();
    populated.deepest_rank_acceptance_mode = Some("margin_rescue".to_string());
    populated.deepest_rank_effective_confidence = Some(38.0);
    populated.deepest_rank_effective_threshold = Some(60.0);
    populated.deepest_rank_challenge_original_selection_confidence = Some(38.0);
    populated.deepest_rank_challenge_runner_up_confidence = Some(18.0);
    populated.deepest_rank_challenge_margin_delta = Some(20.0);
    populated.deepest_rank_challenge_margin_ratio = Some(38.0 / 18.0);
    populated.deepest_rank_challenge_candidate_count = Some(2);
    let json = serde_json::to_string(&populated).unwrap();
    assert!(json.contains("deepest_rank_acceptance_mode"));
    assert!(!json.contains("Infinity"));
    assert!(!json.contains("NaN"));
    let round_trip: ClassificationResult = serde_json::from_str(&json).unwrap();
    assert_eq!(
        round_trip.deepest_rank_acceptance_mode.as_deref(),
        Some("margin_rescue")
    );
    assert_eq!(round_trip.deepest_rank_challenge_candidate_count, Some(2));
}

fn build_clear_sibling_training_set() -> TrainingSet {
    let target = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                  TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
                  CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
                  GATCGATCGATCGATCGATCGATCGATCGATCGATCGATC";
    let sibling = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC\
                   TTTTAAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAA\
                   ATATATATCGCGCGCGATATATATCGCGCGCGATATATAT\
                   CCCCTTTTGGGGAAAACCCCTTTTGGGGAAAACCCCTTTT";
    let sibling_two = "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCC\
                       CGCGATATCGCGATATCGCGATATCGCGATATCGCGATAT\
                       GGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAA\
                       TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGG";
    let outgroup = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA\
                    AATTAATTAATTAATTAATTAATTAATTAATTAATTAATT\
                    GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC\
                    TATATATATATATATATATATATATATATATATATATATA";
    let sequences = vec![
        target.to_string(),
        sibling.to_string(),
        sibling_two.to_string(),
        outgroup.to_string(),
    ];
    let taxonomy = vec![
        "Root; Kingdom; Phylum; Class; Order; Genus; target".to_string(),
        "Root; Kingdom; Phylum; Class; Order; Genus; sibling".to_string(),
        "Root; Kingdom; Phylum; Class; Order; Genus; sibling_two".to_string(),
        "Root; Kingdom; Phylum; Class; OtherOrder; OtherGenus; outgroup".to_string(),
    ];
    learn_taxa(&sequences, &taxonomy, &TrainConfig::default(), 42, false).unwrap()
}

#[test]
fn deepest_rank_margin_rescue_accepts_clear_terminal_sibling_challenge() {
    let ts = build_clear_sibling_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         GATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
        .to_string()];
    let names = vec!["q".to_string()];
    let config = ClassifyConfig {
        rank_thresholds: Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 101.0]),
        deepest_rank_margin_floor: Some(10.0),
        deepest_rank_margin_min_delta: 1.0,
        deepest_rank_margin_min_ratio: 1.01,
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
    let result = &results[0];
    assert_eq!(
        result.deepest_rank_acceptance_mode.as_deref(),
        Some("margin_rescue")
    );
    assert_eq!(result.deepest_rank_margin_rejection_reason, None);
    assert_eq!(
        result.deepest_rank_challenge_candidate_count,
        Some(3),
        "{:?}",
        result
    );
    assert!(
        result.taxon.contains(&"target".to_string()),
        "{:?}",
        result.taxon
    );
}

#[test]
fn deepest_rank_margin_rescue_candidate_cap_rejects_before_rescue() {
    let ts = build_clear_sibling_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         GATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
        .to_string()];
    let names = vec!["q".to_string()];
    let config = ClassifyConfig {
        rank_thresholds: Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 101.0]),
        deepest_rank_margin_floor: Some(10.0),
        deepest_rank_margin_min_delta: 1.0,
        deepest_rank_margin_min_ratio: 1.01,
        max_deepest_rank_challenge_candidates: Some(2),
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
    let result = &results[0];
    assert_eq!(
        result.deepest_rank_acceptance_mode.as_deref(),
        Some("rejected")
    );
    assert_eq!(
        result.deepest_rank_margin_rejection_reason.as_deref(),
        Some("challenge_candidate_cap_exceeded")
    );
    assert!(!result.taxon.contains(&"target".to_string()));
}

#[test]
fn deepest_rank_margin_rescue_runs_in_beam_mode() {
    let ts = build_clear_sibling_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         GATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
        .to_string()];
    let names = vec!["q".to_string()];
    let config = ClassifyConfig {
        beam_width: 2,
        rank_thresholds: Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 101.0]),
        deepest_rank_margin_floor: Some(10.0),
        deepest_rank_margin_min_delta: 1.0,
        deepest_rank_margin_min_ratio: 1.01,
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
    assert_eq!(
        results[0].deepest_rank_acceptance_mode.as_deref(),
        Some("margin_rescue")
    );
    assert!(results[0].beam_candidate_count >= 1);
}

#[test]
fn deepest_rank_margin_rescue_greedy_and_beam_equivalent_on_clear_fixture() {
    let ts = build_clear_sibling_training_set();
    let query = vec!["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         GATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
        .to_string()];
    let names = vec!["q".to_string()];
    let base = ClassifyConfig {
        rank_thresholds: Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 101.0]),
        deepest_rank_margin_floor: Some(10.0),
        deepest_rank_margin_min_delta: 1.0,
        deepest_rank_margin_min_ratio: 1.01,
        ..Default::default()
    };
    let greedy = id_taxa(
        &query,
        &names,
        &ts,
        &base,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let beam_cfg = ClassifyConfig {
        beam_width: 2,
        ..base
    };
    let beam = id_taxa(
        &query,
        &names,
        &ts,
        &beam_cfg,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    assert_eq!(greedy[0].taxon, beam[0].taxon);
    assert_eq!(greedy[0].confidence, beam[0].confidence);
    assert_eq!(
        greedy[0].deepest_rank_acceptance_mode,
        beam[0].deepest_rank_acceptance_mode
    );
    assert_eq!(
        greedy[0].deepest_rank_challenge_candidate_count,
        beam[0].deepest_rank_challenge_candidate_count
    );
    assert_eq!(
        greedy[0].deepest_rank_challenge_original_selection_confidence,
        beam[0].deepest_rank_challenge_original_selection_confidence
    );
    assert_eq!(
        greedy[0].deepest_rank_challenge_runner_up_confidence,
        beam[0].deepest_rank_challenge_runner_up_confidence
    );
}

#[test]
fn deepest_rank_margin_rescue_disabled_json_is_byte_identical_to_baseline() {
    let ts = build_clear_sibling_training_set();
    let queries = vec![
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
         TTAATTAATTAATTAATTAATTAATTAATTAATTAATTAA\
         CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG\
         GATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
            .to_string(),
        "NNNN".to_string(),
    ];
    let names = vec!["target".to_string(), "too_short".to_string()];
    let baseline = ClassifyConfig {
        threshold: 95.0,
        ..Default::default()
    };
    let explicit_disabled = ClassifyConfig {
        deepest_rank_margin_floor: None,
        deepest_rank_margin_min_delta: 1.0,
        deepest_rank_margin_min_ratio: 1.01,
        max_deepest_rank_challenge_candidates: Some(2),
        ..baseline.clone()
    };

    let baseline_results = id_taxa(
        &queries,
        &names,
        &ts,
        &baseline,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );
    let disabled_results = id_taxa(
        &queries,
        &names,
        &ts,
        &explicit_disabled,
        StrandMode::Top,
        OutputType::Extended,
        42,
        true,
    );

    let baseline_json = serde_json::to_string(&baseline_results).unwrap();
    let disabled_json = serde_json::to_string(&disabled_results).unwrap();
    assert_eq!(baseline_json, disabled_json);
    assert!(!baseline_json.contains("deepest_rank_acceptance_mode"));
    assert!(!baseline_json.contains("deepest_rank_challenge"));
}
