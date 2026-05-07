# Deepest-Rank Original Confidence Gate Implementation Plan

## Overview

Add an ordinary failed-rank confidence gate to deepest-rank margin rescue. The
focused terminal-sibling challenge remains the second-stage evidence, but it can
only change the emitted classification after the original deepest-rank
confidence clears a new configurable floor.

The key distinction is:

```text
diagnostics: may run focused challenge regardless of original confidence
final rescue: original confidence floor must pass before focused challenge can rescue
```

## Current State Analysis

Deepest-rank margin rescue already exists and is disabled by default. The live
decision currently uses focused terminal-sibling challenge confidence only:

- `DeepestRankRescueDecisionInput` contains `challenge_original_selection`,
  `challenge_runner`, tie flags, candidate count, `floor`, `min_delta`, and
  `min_ratio`: `src/classify.rs:887`.
- `passes_deepest_rank_margin_rescue` checks focused challenge top, runner,
  uniqueness, exact/near ties, focused floor, delta, and ratio:
  `src/classify.rs:900`.
- The ordinary threshold-walk failure confidence is already available as
  `confidence` in the deepest-rank threshold failure branch:
  `src/classify.rs:2158`.
- That same ordinary value is already stored in result diagnostics as
  `deepest_rank_effective_confidence`: `src/classify.rs:2169`,
  `src/types.rs:327`.
- The focused challenge is evaluated from the original leaf bootstrap context by
  `evaluate_deepest_rank_challenge`: `src/classify.rs:1621`.
- Final rescue currently appends `deepest_idx` when the challenge decision passes
  and `deepest_rank_margin_floor` is enabled: `src/classify.rs:2193`.

The old design document explicitly says rescue floor/delta/ratio use expanded
challenge values, not original pruned confidence. That is now too permissive for
holdout-like cases where the reduced sibling contest is decisive but the normal
confidence was weak.

## Desired End State

When the deepest eligible rank fails the normal threshold, final classification
rescue requires both:

```text
ordinary deepest-rank confidence >= deepest_rank_margin_min_original_confidence
AND focused terminal-sibling challenge passes existing floor/delta/ratio gates
```

The new ordinary-confidence gate is optional:

```rust
deepest_rank_margin_min_original_confidence: Option<f64>
```

Default is `None` to preserve current behavior unless a caller explicitly turns
on the stricter gate.

When diagnostics are enabled, focused challenge diagnostics still run even if
ordinary confidence is below the new floor. Those diagnostics must not rescue
the classification unless the ordinary gate passes.

### Key Discoveries

- The ordinary confidence value already exists at the exact decision site as the
  `confidence` field of `ThresholdStopReason::FailedThreshold`:
  `src/classify.rs:2158`.
- The current focused confidence is different from ordinary confidence; it is
  recomputed inside the expanded terminal-sibling challenge:
  `src/classify.rs:1747`.
- `deepest_rank_diagnostics` already separates diagnostic collection from final
  rescue behavior: `src/classify.rs:2137`.
- Public Python classify kwargs already expose rescue knobs, so the new knob
  belongs on the same PyO3 signature and `ClassifyConfig` construction path:
  `src/lib.rs:191`, `src/lib.rs:295`.
- Default disabled-rescue behavior has explicit byte-identical JSON coverage:
  `tests/test_deepest_rank_margin_rescue.rs:664`.

## What We're NOT Doing

- We are not renaming `deepest_rank_margin_floor`; it continues to mean focused
  terminal-sibling challenge confidence floor.
- We are not making rescue species-label aware. Eligibility remains the current
  global-deepest terminal-group rule.
- We are not changing normal threshold behavior, `rank_thresholds`, or the
  construction of ordinary confidence.
- We are not changing focused challenge scoring, candidate construction, tie
  handling, or candidate-cap behavior.
- We are not enabling rescue by default.
- We are not changing TSV output. Diagnostics remain available on returned
  `ClassificationResult` objects / JSON, not TSV.
- We are not wiring benchmark-harness sweeps in this repo. The classifier change
  should make that possible as a follow-up in `assignment-tool-benchmarking`.

## Implementation Approach

Add one optional config field and one new rejection reason. The ordinary gate is
evaluated in the deepest-rank threshold-failure branch because that branch has
the original failed-rank confidence available before challenge scoring.

The implementation should use two separate concepts:

```text
diagnostics_requested =
    config.deepest_rank_diagnostics || config.deepest_rank_diagnostic_top_k > 0

ordinary_gate_passed =
    config.deepest_rank_margin_min_original_confidence
        .map_or(true, |floor| confidence >= floor)
```

Focused challenge should run when either:

```text
rescue_enabled && ordinary_gate_passed
OR diagnostics_requested
```

Final rescue should happen only when:

```text
rescue_enabled
AND ordinary_gate_passed
AND focused_challenge_decision == Ok(())
```

If ordinary confidence fails while rescue is enabled, set
`deepest_rank_margin_rejection_reason = "original_confidence_floor_failed"`.
That remains the final classification-blocking reason even if diagnostics also
run the focused challenge.

## Phase 1: Config and Public Surface

### Overview

Add the new optional original-confidence gate to Rust config, defaults,
validation, and Python binding.

### Changes Required

#### 1. ClassifyConfig

**File**: `src/types.rs`

**Changes**:

Add a field beside the existing rescue knobs:

```rust
/// Optional ordinary deepest-rank confidence floor required before deepest-rank
/// margin rescue can change the emitted classification. This uses the same
/// confidence value as the normal threshold walk, before focused sibling
/// challenge rescoring. `None` preserves the existing challenge-only rescue
/// behavior.
pub deepest_rank_margin_min_original_confidence: Option<f64>,
```

Default:

```rust
deepest_rank_margin_min_original_confidence: None,
```

Validation:

```rust
if let Some(v) = self.deepest_rank_margin_min_original_confidence {
    if !v.is_finite() || !(0.0..=100.0).contains(&v) {
        return Err(format!(
            "deepest_rank_margin_min_original_confidence must be finite and in [0, 100], got {}",
            v
        ));
    }
}
```

#### 2. Python Binding

**File**: `src/lib.rs`

**Changes**:

Add a PyO3 kwarg near the existing rescue knobs:

```rust
deepest_rank_margin_min_original_confidence = None,
```

Add the function argument:

```rust
deepest_rank_margin_min_original_confidence: Option<f64>,
```

Set the `ClassifyConfig` field during construction:

```rust
deepest_rank_margin_min_original_confidence,
```

#### 3. Documentation

**Files**:

- `README.md`
- `oxidtaxa_method.html`
- `oxidtaxa_walkthrough.html` if the walkthrough's rescue scenario describes
  decision fields or knobs.

**Changes**:

Document the two confidence gates explicitly:

```text
deepest_rank_margin_min_original_confidence:
  Optional floor on ordinary failed-rank confidence. Final rescue cannot happen
  unless this floor passes.

deepest_rank_margin_floor:
  Floor on the focused terminal-sibling challenge confidence, evaluated after
  the ordinary confidence gate.
```

Make clear that diagnostics can still evaluate the focused challenge below the
ordinary gate when `deepest_rank_diagnostics` or top-k diagnostics are enabled.

### Success Criteria

#### Automated Verification

- [x] `cargo fmt --check`
- [x] `cargo test --test test_deepest_rank_margin_rescue`
- [x] `cargo test`
- [x] `maturin develop --release` or the repo's active local build command
      succeeds after the PyO3 signature change.

#### Manual Verification

- [x] Python help/signature exposes the new optional kwarg.
- [x] Existing classify calls without the new kwarg still work.
- [x] Documentation distinguishes ordinary confidence from focused challenge
      confidence without reusing ambiguous "species rescue" wording.

---

## Phase 2: Decision Rule

### Overview

Teach the rescue decision logic about the ordinary confidence floor and add a
stable rejection reason.

### Changes Required

#### 1. Rejection Reason

**File**: `src/classify.rs`

**Changes**:

Add enum variants:

```rust
NonfiniteOriginalConfidence,
OriginalConfidenceFloorFailed,
```

Map it to:

```rust
"nonfinite_original_confidence"
"original_confidence_floor_failed"
```

Use the same string for `deepest_rank_stop_reason_for_rejection`.

#### 2. Decision Input

**File**: `src/classify.rs`

**Changes**:

Extend `DeepestRankRescueDecisionInput`:

```rust
pub original_confidence: f64,
pub min_original_confidence: Option<f64>,
```

#### 3. Decision Helper

**File**: `src/classify.rs`

**Changes**:

After confirming rescue is enabled, check the ordinary confidence floor before
any focused challenge candidate-count, score, floor, delta, or ratio gate. This
preserves the intended ordering for final classification:

```rust
if let Some(floor) = input.min_original_confidence {
    let c = input.original_confidence;
    if !c.is_finite() {
        return Err(RescueRejection::NonfiniteOriginalConfidence);
    }
    if c < floor {
        return Err(RescueRejection::OriginalConfidenceFloorFailed);
    }
}
```

Keep existing focused challenge checks after that. The integration layer will
handle the production fast path so low ordinary-confidence rows avoid challenge
scoring when diagnostics are off, but the helper must still encode the final
classification ordering for tests and diagnostic runs where challenge scoring
was performed anyway.

### Success Criteria

#### Automated Verification

- [x] Unit test: `None` original floor preserves existing passing/failing
      decision behavior.
- [x] Unit test: `Some(50.0)` rejects when `original_confidence = 49.99` even
      if focused challenge top/delta/ratio pass.
- [x] Unit test: `Some(50.0)` accepts when `original_confidence = 50.0` and
      focused challenge gates pass.
- [x] Unit test: nonfinite original confidence rejects with
      `NonfiniteOriginalConfidence` when the original floor is enabled.
- [x] Unit test: validation rejects NaN, infinity, negative, and >100 original
      floors.

#### Manual Verification

- [x] Rejection string is stable and readable in JSON result diagnostics.
- [x] The old `floor_failed` reason still refers only to focused challenge
      confidence floor failure.

---

## Phase 3: Leaf-Phase Integration

### Overview

Wire the ordinary failed-rank confidence into final rescue while preserving
diagnostic-only focused challenge collection.

### Changes Required

#### 1. Compute Gate State

**File**: `src/classify.rs`

**Current location**: deepest-rank threshold failure branch around
`src/classify.rs:2168`.

**Changes**:

After recording `deepest_rank_effective_confidence` and
`deepest_rank_effective_threshold`, compute:

```rust
let original_confidence_gate_passed = config
    .deepest_rank_margin_min_original_confidence
    .map_or(true, |floor| confidence >= floor);

let diagnostics_requested =
    config.deepest_rank_diagnostics || config.deepest_rank_diagnostic_top_k > 0;

let should_run_challenge =
    (rescue_enabled && original_confidence_gate_passed) || diagnostics_requested;
```

If `rescue_enabled && !original_confidence_gate_passed`, immediately set:

```rust
deepest_rank_margin_rejection_reason =
    Some(RescueRejection::OriginalConfidenceFloorFailed.as_str().to_string());
deepest_rank_stop_reason = Some(
    deepest_rank_stop_reason_for_rejection(
        RescueRejection::OriginalConfidenceFloorFailed,
    )
    .to_string(),
);
```

#### 2. Challenge Execution

**File**: `src/classify.rs`

**Changes**:

Only call `evaluate_deepest_rank_challenge` when `should_run_challenge` is true.

Pass both original confidence fields into the challenge decision input. This
keeps unit tests and diagnostic-mode challenge evaluation on the same decision
helper used for final classification.

Preferred final rescue branch:

```rust
match challenge.decision {
    Ok(()) if rescue_enabled && original_confidence_gate_passed => {
        above.push(deepest_idx);
        deepest_rank_acceptance_mode = Some(DEEPEST_MODE_MARGIN_RESCUE.to_string());
        deepest_rank_margin_rejection_reason = None;
    }
    Ok(()) => {
        if rescue_enabled && !original_confidence_gate_passed {
            deepest_rank_margin_rejection_reason = Some(
                RescueRejection::OriginalConfidenceFloorFailed.as_str().to_string(),
            );
        }
    }
    Err(reason) => {
        if original_confidence_gate_passed {
            deepest_rank_margin_rejection_reason = Some(reason.as_str().to_string());
        }
    }
}
```

The important rule is that an ordinary-confidence failure keeps the final
classification truncated even when diagnostic challenge fields show a strong
focused challenge.

#### 3. Diagnostic Semantics

**File**: `src/classify.rs`

**Changes**:

When diagnostics are enabled and ordinary confidence fails:

- populate `deepest_rank_challenge_original_selection_confidence` if the
  challenge scores;
- populate runner, delta, ratio, candidate count, selected path, runner-up path,
  and top-k exactly as current challenge code allows;
- keep `deepest_rank_acceptance_mode = Some("rejected")`;
- keep `deepest_rank_margin_rejection_reason =
  Some("original_confidence_floor_failed")`;
- keep final `taxon` truncated.

### Success Criteria

#### Automated Verification

- [x] Integration test: original confidence below floor, diagnostics off,
      challenge would otherwise pass, final taxon remains truncated, challenge
      fields are absent.
- [x] Integration test: original confidence below floor, diagnostics on,
      challenge would otherwise pass, final taxon remains truncated, challenge
      fields are populated.
- [x] Integration test: original confidence equals floor, focused challenge
      passes, final taxon is rescued.
- [x] Integration test: original confidence passes floor, focused challenge
      fails delta/ratio, final taxon remains truncated with the focused failure
      reason.
- [x] Existing diagnostics-only test still passes or is updated to account for
      the new default field without changing behavior.

#### Manual Verification

- [x] Inspect one JSON result where ordinary confidence fails but diagnostics are
      on; it should contain both the ordinary confidence and the focused
      challenge confidence, with final rejection reason pointing at the ordinary
      gate.
- [x] Confirm no extra challenge scoring is performed for low ordinary-confidence
      rows when diagnostics are off.

---

## Phase 4: Tests and Regression Coverage

### Overview

Extend the focused rescue test suite to pin default behavior, new gate behavior,
and diagnostic behavior.

### Changes Required

#### 1. Unit Tests

**File**: `tests/test_deepest_rank_margin_rescue.rs`

**Changes**:

Update `decision_input` helper to set:

```rust
original_confidence: top,
min_original_confidence: None,
```

Add focused decision tests:

```rust
#[test]
fn deepest_rank_rescue_decision_rejects_original_confidence_below_floor() {
    let mut input = decision_input(80.0, 10.0);
    input.original_confidence = 49.99;
    input.min_original_confidence = Some(50.0);
    assert_eq!(
        passes_deepest_rank_margin_rescue(&input),
        Err(RescueRejection::OriginalConfidenceFloorFailed)
    );
}
```

#### 2. Integration Fixtures

**File**: `tests/test_deepest_rank_margin_rescue.rs`

**Changes**:

Reuse `build_clear_sibling_training_set()` and the existing forced deepest-rank
threshold failure fixture. Choose original-confidence floors relative to the
fixture's actual `deepest_rank_effective_confidence` instead of relying on exact
absolute values:

1. Run diagnostics once to obtain `deepest_rank_effective_confidence`.
2. Set floor to `effective_confidence + 1.0` for rejection.
3. Set floor to `effective_confidence` or lower for acceptance.

This avoids brittle expectations if bootstrap confidence shifts slightly.

#### 3. Byte-Identity Coverage

**File**: `tests/test_deepest_rank_margin_rescue.rs`

**Changes**:

Extend the existing byte-identical disabled-rescue test so that:

```rust
deepest_rank_margin_min_original_confidence: None
```

is part of explicit disabled config and JSON remains identical to baseline.

### Success Criteria

#### Automated Verification

- [x] `cargo test --test test_deepest_rank_margin_rescue`
- [x] `cargo test deepest_rank_rescue_decision`
- [x] `cargo test`

#### Manual Verification

- [x] The tests prove the new parameter is inert at `None`.
- [x] The tests prove diagnostics can still run when ordinary confidence fails.
- [x] The tests prove final rescue cannot happen unless ordinary confidence
      passes.

---

## Phase 5: Benchmark Follow-Up Contract

### Overview

This repo change should expose enough classifier surface for later benchmark
sweeps, but the harness wiring lives elsewhere.

### Expected Sweep Values

For follow-up tuning, sweep the new ordinary-confidence gate independently from
focused challenge params:

```text
deepest_rank_margin_min_original_confidence:
  None, 35, 40, 45, 50, 55

deepest_rank_margin_floor:
  existing focused challenge floor values

deepest_rank_margin_min_delta:
  existing focused challenge delta values

deepest_rank_margin_min_ratio:
  existing focused challenge ratio values
```

Do not default to a floor near the normal threshold unless benchmark evidence
supports it. If the normal threshold is 60 and this gate is also 60, rescue
mostly collapses into "ordinary threshold pass," which removes the point of the
focused challenge.

### Success Criteria

#### Automated Verification

- [x] Python `oxidtaxa.classify(..., deepest_rank_margin_min_original_confidence=50.0)`
      accepts the new kwarg after local rebuild.
- [ ] A stale installed build that lacks the kwarg fails clearly when explicit
      non-default benchmark configs use it. This is a benchmark-harness follow-up
      outside this repo.

#### Manual Verification

- [ ] Benchmark configs can distinguish:
      ordinary original confidence floor from focused challenge confidence floor.
- [ ] Decision analysis can compare unsafe rescued rows where focused challenge
      confidence is high but ordinary confidence is low.

---

## Testing Strategy

### Unit Tests

- Rescue decision accepts old behavior when original floor is `None`.
- Rescue decision rejects below original floor.
- Rescue decision accepts at exact original floor boundary.
- Config validation rejects invalid original floor values.
- Rejection reason string is stable.

### Integration Tests

- Clear terminal-sibling challenge rescues only when original floor passes.
- Clear terminal-sibling challenge does not rescue when original floor fails.
- Diagnostics-on mode still fills focused challenge fields when original floor
  fails.
- Diagnostics-off mode avoids focused challenge fields when original floor fails.
- Beam mode follows the same gate semantics.

### Manual Testing Steps

1. Build locally with `maturin develop --release`.
2. Run a small query set with rescue disabled; compare output with current
   baseline.
3. Run with existing rescue knobs and no original floor; behavior should match
   current rescue-enabled behavior.
4. Run with `deepest_rank_margin_min_original_confidence` set above the observed
   `deepest_rank_effective_confidence`; verify the result truncates.
5. Re-run the same case with `deepest_rank_diagnostics=true`; verify focused
   challenge fields populate while final classification remains truncated.

## Performance Considerations

The new gate can reduce production rescue overhead because low ordinary-confidence
rows can skip the expanded terminal-sibling challenge when diagnostics are off.

Diagnostic runs intentionally keep the current cost profile or higher: if
`deepest_rank_diagnostics` or top-k diagnostics are enabled, focused challenge
scoring still runs even below the ordinary-confidence gate.

No extra hot-path allocations are required for ordinary gate evaluation because
the ordinary failed-rank confidence is already present in the threshold failure
branch.

## Migration Notes

Rust callers using full `ClassifyConfig` struct literals will need to add the
new field, as with prior config additions. Python callers are backward
compatible because the kwarg defaults to `None`.

Existing JSON output is unchanged when the new field is `None` and diagnostics
are disabled. JSON output for explicit original-floor rescue configs may include
the new rejection reason `original_confidence_floor_failed`.

## References

- Rescue decision helper: `src/classify.rs:887`
- Focused challenge evaluator: `src/classify.rs:1621`
- Ordinary threshold failure branch: `src/classify.rs:2158`
- Rescue append site: `src/classify.rs:2193`
- `ClassificationResult` diagnostics: `src/types.rs:324`
- `ClassifyConfig` rescue knobs: `src/types.rs:744`
- Python classify kwargs: `src/lib.rs:191`
- Focused rescue tests: `tests/test_deepest_rank_margin_rescue.rs`
- Original rescue plan: `thoughts/shared/plans/2026-05-01-species-margin-rescue-confidence.md`
- Diagnostics plan: `thoughts/shared/plans/2026-05-04-oxidtaxa-decision-diagnostics-paths.md`
