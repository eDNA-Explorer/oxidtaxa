---
date: 2026-04-30
researcher: Ryan Martin
git_commit: 7eed283
branch: main
repository: eDNA-Explorer/oxidtaxa
status: in_progress
last_updated: 2026-04-30
last_updated_by: Ryan Martin
topic: "Consolidated follow-up actions from the 15-parameter audit"
tags: [plan, parameter-audit, follow-ups]
---

# Parameter audit follow-ups — consolidated action list

This document consolidates decisions from the 15-parameter deep-dive audit at `thoughts/shared/research/2026-04-30-param-*-deep-dive.md`. Each section captures the audit's headline findings, the decision made, and the resulting action items.

Status legend: ✅ no action needed · 🔧 action item agreed · 🟡 deferred · ❌ rejected

## Reviewer feedback applied (2026-04-30 round 2)

A colleague review identified seven issues with the initial plan. Verified against repo state and corrected:

1. **`tie_margin` negative-`max_tot` math.** Original "drop the `max_tot > 0` guard" was wrong — current `cutoff = max_tot * (1 - tie_margin)` produces an *unreachable* cutoff when `max_tot < 0` (e.g., `-10 * 0.95 = -9.5`, but `-10 < -9.5`, winners empty, panic at `winners[0]`). Replaced with `cutoff = max_tot - max_tot.abs() * tie_margin`. Same fix applied to the broad share-split relaxation.
2. **Validation API surface.** `id_taxa` returns `Vec<ClassificationResult>` (not `Result<_>`). Validation that needs to surface errors must go in the Python binding (`src/lib.rs:185-208` `classify` pyfunction returns `PyResult<...>`, can raise `PyValueError`). Updated all "throw a config error" items to be Python-side; `debug_assert!` is the Rust-side fallback.
3. **Python validation premise.** Repeated claims that "Python wrapper validates `(0, 1]`" referenced an external benchmark harness, NOT the pyo3 binding. Real state: zero validation in `src/lib.rs:185-208`. So the validation tasks are *additions*, not harmonizations.
4. **`full_length` references.** Removed all `length_normalize` items that referenced `full_length` (which was deleted in commit `f097f05`).
5. **`use_idf_in_descent` rename drift.** Demoted to ✅ — verified zero remaining `use_idf_in_training` occurrences in user-facing docs at HEAD.
6. **README parameter table backfill.** Demoted to ✅ — `README.md:134-136` already has rows for `tie_margin`, `confidence_uses_descent_margin`, `sibling_aware_leaf`.
7. **`correlation_aware_features` testing gap overstated.** Demoted three of four sub-properties (determinism, parallel-vs-sequential, Bhattacharyya valid-model) to "already covered" — they exist at `tests/test_algorithmic_improvements.rs:613, 648, 681, 717`. Real gaps are only the `n_children=2` regression and all-zero-profile cases.
8. **`model.n_ranks` accessor.** Verified `TrainingSet` is NOT a pyclass (only `PreparedData` and `BuiltTree` are). Standalone helper `model_n_ranks(model_path)` is the minimal path; pyclass-wrapping is overkill.

---

## Cross-cutting research findings

### `beam_width` × `sibling_aware_leaf` interaction

Two audits (`beam_width`, `sibling_aware_leaf`) surfaced the same asymmetry from different angles. The user-facing claim in `oxidtaxa_method.html` §7c is that the two flags are "orthogonal." That holds at the mid-tree level (beam widens by adding alternative paths; sibling-aware widens by adding sibling leaves at the same path) but breaks at the leaf-parent step, where they were designed to compose.

What the audits found:

- **Greedy and beam diverge at the leaf-parent success path.** When the top child cleanly clears `min_descend` at a leaf-parent node:
  - Greedy at `src/classify.rs:341-351` honors `sibling_aware_leaf` — if true, widens `w_indices` to include any sibling with `vote_counts[j] >= 0.5 * b`.
  - Beam at `src/classify.rs:560-566` unconditionally builds `w_indices = vec![winner_idx]` and never reads `config.sibling_aware_leaf`.

- **The halt path is mirrored, so the asymmetry only fires on success.** When no child clears `min_descend` (Mechanism 1 fallback), both paths use the same `> 0 votes` widening — greedy at `:314-335`, beam at `:608-629`. The tied-bootstrap case the walkthrough's S7 depicts goes through this path and behaves correctly under both descent strategies.

- **The flag is silently inert under `beam_width > 1`.** A user who sets `beam_width=2, sibling_aware_leaf=true` and lands on a node where Stage 1 succeeds (top child cleanly clears the gate) gets `w_indices = vec![winner_idx]` regardless of the flag. The flag has no effect on the success path under beam.

- **Why this matters for `tie_margin`.** The intended design (per method.html §8a Mechanism 2 + §8c) is `sibling_aware_leaf + tie_margin` together: sibling-aware brings the runner-up species into `keep`, and tie_margin lets it count as a co-winner for LCA-cap reporting. Under `beam_width > 1`, the runner-up never enters `keep` on the success path, so `tie_margin` has nothing to detect. The pair fails to compose — silently. This means the `tie_margin` audit's "designed pair" claim (one without the other is inert) is itself partially inert under beam.

- **Latent greedy ≠ beam-width-1.** The dispatcher at `src/classify.rs:211` gates beam on `beam_width > 1`, so a `beam_width=1` user always hits the greedy path. If the dispatcher were ever relaxed (or beam=1 were used as a unified code path), the leaf-parent asymmetry would surface immediately. The asymmetry is masked by the dispatcher today, not absent.

- **Historical context.** No plan or research doc in `thoughts/` explains why the beam path was written without the `sibling_aware_leaf` widening. The `oxidtaxa_oat_ablation` plan catalogs both flags as separate sweep axes but doesn't probe their combined behavior. The 2026-04-22 walkthrough plan promised an S7 scenario depicting beam rescue of a 58/40 split, but post-622ca33 the 58/40 case under `min_descend=0.98` no longer rescues anything — the actual S7 was rewritten around tied-bootstraps, which (per the second bullet above) doesn't exercise the asymmetry either.

---

## Classify-time parameters

### `beam_width`

Wiring is clean. Findings:

- 🔧 **Add a positive beam test.** A fixture-based test that locks in two contracts: (1) `beam=2` produces a different `taxon` than `beam=1` on a tied-bootstrap fixture (Felidae → {Felis, Lynx}, both clear `min_descend`); (2) the 622ca33 runner-up gate — `beam=2 + min_descend=0.98` on a 100/97 split (top passes, runner-up doesn't) yields the same taxon as `beam=1`. Today only one regression test references `beam_width`, and it guards an unrelated descent-margin plumbing bug.
- 🔧 **Refresh line numbers in `oxidtaxa_method.html` §7c and `oxidtaxa_walkthrough.html` S7.** Mechanical drift sweep: `:289→:211`, `:307→:399`, `:430→:556`, `:458-462→:584-588`, `:503→:640-645`, `:512→:656`, plus `:481-499→:607-636`, `:380-387→:501-512`, `:451-457→:577-588`. Narrative content unchanged.
- 🟡 **Beam vs greedy `sibling_aware_leaf` divergence (A in discussion).** Decision deferred to the cross-cutting research entry above.
- 🔧 **Expand field doc-comment with "when to flip" guidance.** Current doc-comment at `src/types.rs:340-343` is structurally accurate but doesn't tell a user when to enable it. Add a paragraph that diverges-from-greedy only when ≥2 children clear `min_descend` (rare under default 0.98 — typically requires tied bootstraps or relaxed `min_descend`), and note that sub-threshold runner-ups are not rescued (cross-ref `classify.rs:577-583`).
- 🟡 **`beam_strict_runner_ups: bool` opt-out (E in discussion).** Deferred — speculative, only worth adding if OAT ablation surfaces a regime where the rescue helps.

### `sibling_aware_leaf`

Audit: `thoughts/shared/research/2026-04-30-param-sibling_aware_leaf-deep-dive.md`. Single greedy read site at `src/classify.rs:341-351`; beam-path divergence captured under cross-cutting research above. The flag was designed to compose with `tie_margin` — sibling-aware brings the runner-up into `keep`, tie_margin lets it count as a co-winner.

- 🔧 **Add a positive sibling-aware test (folded into the beam test).** The same near-tied terminal fixture used for the beam test (Felidae → {Felis, Lynx}) doubles as a `sibling_aware_leaf` regression: with `sibling_aware_leaf=true, tie_margin=0.05`, assert `alternatives.len() >= 2` and a one-rank-shorter LCA-clamped `taxon`. Also exercises the cross-cutting beam-path divergence when run with `beam_width=2`.
- 🔧 **Expose `sibling_aware_min_vote_frac: f64` as a `ClassifyConfig` knob, default `0.5`.** Read at both Mech 1 (`src/classify.rs:317`) and Mech 2 (`src/classify.rs:342`) sites. Replaces the hardcoded `0.5` magic number; lets the OAT sweep test other thresholds. Note that this knob couples the two mechanisms — they currently use the same constant by accident, and exposing it preserves that.
- 🔧 **Reject `sibling_aware_leaf=true && tie_margin == 0.0` at the Python binding** (Python-only — the Rust `id_taxa` returns `Vec<ClassificationResult>`, not `Result<_>`, so error-return at the Rust entry point would require a breaking signature change). Add the check in `src/lib.rs:185-208` (`classify` pyfunction) before constructing `ClassifyConfig`; raise `PyValueError`. Direct Rust callers are not protected, but every production caller goes through Python. Optionally add a parallel `debug_assert!` inside `id_taxa` for development-time defense.
- ✅ **README parameter table backfill.** Verified DONE — `README.md:134-136` already has rows for `tie_margin`, `confidence_uses_descent_margin`, `sibling_aware_leaf`. Tier 3 sweep guide pairing reminder remains a small doc tweak; folded into the cross-cutting docs sweep.
- 🔧 **Unify Mech 1 and Mech 2 under the same flag.** Currently `sibling_aware_leaf=true` only tightens the success path (Mech 2, `:341-351`); the halt-at-leaf-parent fallback (Mech 1, `:314-334`) always uses the broad `> 0 votes` filter regardless of the flag. Change Mech 1's halt-at-leaf-parent branch to use the strict `≥ sibling_aware_min_vote_frac·B` filter when `sal=true`, preserving the empty-`w50` fallback to ALL-children for scattered cases. Net: one flag governs both paths, no new knob. Tied-decisive cases (lupus=98, latrans=98, familiaris=4) emit `{lupus, latrans}` instead of `{lupus, latrans, familiaris}`; scattered cases are unaffected.
- ✅ **Cast-truncation oddity.** `((b as f64) * 0.5) as usize` is one less than strict 0.5·B for odd `b`. Same truncation at both sites; behavior consistent. Leaving as-is.

### `tie_margin`

Audit: `thoughts/shared/research/2026-04-30-param-tie_margin-deep-dive.md`. Wiring is clean; single read site at `src/classify.rs:1016-1031`. Designed to compose with `sibling_aware_leaf` (captured under sibling_aware_leaf section).

- 🔧 **Fix the cutoff formula to handle negative `max_tot`.** Current code at `:1016-1031` uses `cutoff = max_tot * (1 - tie_margin)` gated on `max_tot > 0`. Removing the guard alone is **wrong**: with `max_tot = -10, tie_margin = 0.05`, current formula gives `cutoff = -9.5`, but `-10 >= -9.5` is FALSE → winners empty → `winners[0]` panics later. Replace the formula with `cutoff = max_tot - max_tot.abs() * tie_margin` (gives `-10 - 0.5 = -10.5`, `-10 >= -10.5` TRUE). Drop the `max_tot > 0` guard. Add tests for: (a) positive-IDF case (regression unchanged), (b) negative `max_tot` with non-trivial `tie_margin`, (c) `max_tot = 0` corner (cutoff=0; only zero-score groups qualify, identical to the legacy exact-equality branch).
- 🔧 **Range-validate `tie_margin` at the Python boundary.** Python wrapper at `src/lib.rs:185-208` constructs `ClassifyConfig` directly with no checks (the audit's "Python validates" claim referenced an external benchmark harness, not the pyo3 binding — confirmed). Add `if !(0.0..=1.0).contains(&tie_margin) { return Err(PyValueError::new_err(...)); }` in the `classify` pyfunction before struct construction. Optionally add a `debug_assert!` mirror in Rust core for direct-Rust-API misuse.
- 🔧 **Broad share-split relaxation (whenever `tie_margin > 0`).** Share-split at `:971-999` currently fires only on exact `==` per-rep ties. Extend the per-replicate `is_tied` test to use the same relaxed cutoff as the post-bootstrap winner stage. Use the negative-safe formula from the previous bullet: `cutoff = max_val - max_val.abs() * tie_margin`, then `is_tied[j] = hits_flat[ti*b+rep] >= cutoff`. Rationale: `tie_margin`'s design intent is "relax exact equality," and applying it in only one of the two places that test for ties is inconsistent. Canonical IDTAXA's `ties.method = "random"` for exact ties already gives partial credit in expectation; oxidtaxa's deterministic share-split is mathematically equivalent. Extending to near-ties is the same logic with a relaxed test — not a new semantic. Tests must include negative-`hits_flat` fixtures.
- 🔧 **Add missing tests.** Three: (1) `tie_margin > 0` combined with `suppress_ancestor_only_groups=true` on a near-tied (non-byte-identical) ancestor/descendant pair; (2) `tie_margin` on the mid-tree halt path (Mech 1) where `sibling_aware_leaf` is irrelevant; (3) `tie_margin = 1.0` (every group with `tot_hits >= 0` qualifies, LCA cap at root).
- 🔧 **Refresh stale line ref in method.html §8c.** Cited `src/classify.rs:742-749`; current is `:1016-1031`. Folds into the cross-cutting line-drift sweep.
- ✅ **README backfill.** Already locked in under `sibling_aware_leaf` (E) — `tie_margin`, `sibling_aware_leaf`, `confidence_uses_descent_margin` rows added together.

### `suppress_ancestor_only_groups`

Audit: `thoughts/shared/research/2026-04-30-param-suppress_ancestor_only_groups-deep-dive.md`. Implementation is clean post-7eed283 (Stage 1 share-split filter + Stage 2 winner-stage filter + cross-rank universal fix from 8328c2e). Eight tests passing; docs current across all four surfaces. No code-logic changes to this flag — only test additions and doc-comment refinement. The semantic does evolve via the broad `tie_margin` change locked under `tie_margin` (Stage 1's "tied" definition becomes relaxed by `tie_margin > 0`).

- 🔧 **Add multi-branch-descendant test.** Fixture: ancestor `Family` + descendants under two different genera (e.g., `Family;GenusA;Species_1` and `Family;GenusB;Species_2`). Assert Stage 1 drops the Family ancestor when any descendant in any branch is tied — not just same-parent descendants. Implementation already handles this correctly by construction; test makes the contract explicit.
- 🔧 **Add negative-`hits_flat` probe test.** The `:958-962` comment notes the `max_val > 0` guard was removed because IDF can produce negative scores. No fixture currently probes the tied-prefix branch when scores are negative. Add a test using a negative-IDF-skewed fixture (or a synthetic one) to lock in that the suppression still fires correctly with negative `hits_flat`.
- 🔧 **Lock the "drop outright-win ancestor from winners" semantic.** `test_suppress_ancestor_outright_wins_unaffected` allows either outcome for whether the ancestor appears in `alternatives` when its post-bootstrap `tot_hits` ties with a descendant's. Add a strict assertion: the ancestor is dropped from `winners` regardless of how its `tot_hits` accumulated (outright wins included), because the flag's design intent is "prefer descendants over ancestor-only entries." Pins the contract.
- 🔧 **Tick stale Phase-3 boxes in the share-split plan doc.** `thoughts/shared/plans/2026-04-30-prefix-aware-bootstrap-share-split.md` Phase 3 claims README is undocumented; `README.md:129` actually does describe both stages. Update the plan tracker to reflect current state.
- 🔧 **Doc-comment update at `src/types.rs:359-378`** (comment-only, no logic change). Add a sentence noting that Stage 1's tie definition tracks `tie_margin` once the broad relaxation lands — i.e., when `tie_margin > 0`, Stage 1 fires on near-ties, not just exact ties.
- 🟡 **Stage 1 / Stage 2 split flag.** UX gripe: a user can't opt into only Stage 1 or only Stage 2 — single flag controls both. By design (both share the same prefix-test logic and conservation reasoning). Defer unless a real use case emerges.

### `confidence_uses_descent_margin`

Audit: `thoughts/shared/research/2026-04-30-param-confidence_uses_descent_margin-deep-dive.md`. Implementation is correct at HEAD after three bug-fix commits (79ecdb9 off-by-one, 17b492b beam path + non-cumulative, e287eb7 affine remap). Current semantics: per-rank, non-cumulative, affine-remapped via `MARGIN_FLOOR + (1 - MARGIN_FLOOR) * m` with `MARGIN_FLOOR = 0.8`. Effective discount range `[0.82, 1.0]` per rank. No code-logic changes.

- 🔧 **Rewrite stale field doc-comment at `src/types.rs:349-352`.** Currently describes pre-e287eb7 cumulative-product semantics ("running product of per-node descent margins... floored at 0.1"). Replace with current affine-remap description: per-rank, non-cumulative, multiplier in `[0.82, 1.0]`, Root untouched.
- 🔧 **Add a §7d body in `oxidtaxa_method.html`.** Parallel to §8d for `suppress_ancestor_only_groups`. The flag deserves a body section similar to other opt-in features. Either place it as §7d (between beam_width §7c and §8 leaf-phase) or as §8e (post tie_margin §8c, post suppress §8d). §7d is more accurate since the data is collected during Stage 1 descent. Update §12 index cross-reference (currently dangling at "§7b/§8c"). Add the flag as a new row in the §3 differences-from-IDTAXA table.
- 🔧 **Tighten test doc-comments and assertion threshold.** `tests/test_margin_aware.rs:212-216, 438-442` doc-comments still describe cumulative-product semantics — update to affine. The assertion at line 492 (`ratio >= 0.1`) passes under either old cumulative or new affine semantics — cannot distinguish a regression. Tighten to `ratio >= 0.5` (current affine `0.82^L` for typical L=4-6 comfortably exceeds 0.5; cumulative `0.1^L` would fail).
- 🔧 **Rewrite `test_descent_margin_active_with_beam_width_3` to use a synthetic fixture.** Currently silently skips if `benchmarks/data/bench_1000_*` is absent (`tests/test_margin_aware.rs:289`). CI without benchmark data has zero coverage of the 17b492b beam-path fix. Synthetic fixture eliminates the dependency entirely.
- 🔧 **Add direct off-by-one regression test for 79ecdb9.** No test currently asserts that margin recorded at descent-step `k` discounts `confidences[k+1]` specifically. Construct a fixture where the margin at step 2 is intentionally extreme (e.g., 0.0 → effective 0.82) and verify only `confidences[3]` is affected, not `confidences[2]` or `confidences[4]`.
- 🔧 **Document unary-rank index-alignment + add test.** When `subtrees.len() == 1` (e.g., monotypic family), no margin is recorded; `descent_margins.get(i-1)` returns None → that rank is undiscounted. Correct behavior, undocumented. Add a brief doc-comment at `classify.rs:355-361` explaining the alignment, plus a test fixture exercising a taxonomy with a unary intermediate node.
- ✅ **README backfill.** Covered under `sibling_aware_leaf` (E) — `tie_margin`, `sibling_aware_leaf`, `confidence_uses_descent_margin` rows added together.
- ✅ **Walkthrough S8 line-number drift.** Covered under cross-cutting line-drift sweep.
- 🟡 **Hardcoded magic numbers.** `MARGIN_FLOOR = 0.8`, `(1 - MARGIN_FLOOR) = 0.2`, raw-margin floor `0.1` at `:301, 523`. Three rounds of bug-fixes already occurred; adding tunability invites more failure modes. Defer until concrete use case appears.
- 🟡 **Margin defaults to 1.0 when all children have zero votes.** Practically unreachable (descent halts via w50 fallback). Leave alone.

### `length_normalize`

Audit: `thoughts/shared/research/2026-04-30-param-length_normalize-deep-dive.md`. Single read site at `src/classify.rs:825-839`. Implementation is light on documentation (one-line comment) and has no test coverage. Sqrt exponent is heuristic with no derivation. The most important issue is the per-query keep-local averaging, which is structurally inert at single-keep — exactly the singleton-leaf gotcha case. F2 (train-time fixed average) is the right fix.

- 🔧 **Add tests.** No `cargo test` covers `length_normalize=true`. Add: (1) a positive test that the divisor changes hit values when length variance is present; (2) a single-keep test that captures current no-op behavior (so the F2 fix has a regression target).
- 🔧 **Fix doc-comment imprecision at `src/types.rs:334-336`.** Says "training sequence length"; actual statistic is unique-k-mer count after seed/dedup (`ls[i] = ts.kmers[i].len()`). Update to match reality.
- 🔧 **Add method.html coverage.** No mention of `length_normalize` anywhere in `oxidtaxa_method.html`. Add a §3 row, a body section in §8 (parallel to §8d for suppress_ancestor_only_groups), and a §12 entry. Note: this is one of three flags also missing from method.html (`length_normalize`, `confidence_uses_descent_margin`, `seed_pattern`).
- 🔧 **Fix walkthrough S11 caption.** Currently implies fixed `avg=5.5` across all rows; actual behavior is per-query keep-local recomputation. Update caption to reflect per-leaf-phase scoping.
- 🔧 **Add in-source rationale at `classify.rs:824-839`.** Acknowledge the sqrt is heuristic ("compromise between no correction and linear; not derivable from a sampling model"). Compare to `confidence_uses_descent_margin`'s 8+ line justification at `:1136-1160`. Pin the sqrt as a known choice rather than a derived one.
- 🔧 **F2: train-time fixed average per node.** Replace per-query keep-local averaging with a stable train-time `avg_n_unique_per_node: Vec<f64>` cached on `TrainingSet`. Fixes the single-keep no-op. Plan sketch:
  - Add field to `TrainingSet` (and `PreparedData` if needed upstream); index by node id; one entry per internal node.
  - Compute during `create_tree`: at each internal node, `mean(ls[i] for i in subtree sequences)`. Recurse naturally with the existing 5-tuple return.
  - At classify, replace `avg_unique = mean(ls[idx] for idx in keep)` at `:826-827` with `avg_unique = ts.avg_n_unique_per_node[stop_node_id]`.
  - Add `avg_n_unique_global: f64` as backstop for opaque `max_children` nodes (no cached avg available).
  - Bincode schema change → bump version (we've taken bincode breaks already this audit cycle for leave_one_out's `raw_counts`).
  - Tests: singleton-leaf fixture (verify divisor != 1.0); cross-query stability (same training seq normalized identically regardless of routing).
  - Edge case: if stop-node has only one descendant, cached avg is single-ref → same no-op. Mitigation: walk up to nearest ancestor with n_descendants ≥ 2, OR accept (very small subtrees don't benefit).
- 🔧 **G1 doc-only: surface the bidirectional behavior in README.** "length_normalize is symmetric — long refs are demoted AND short refs are promoted (divisor < 1.0 scales hits up). If short-ref promotion is undesirable for your data, see G2 below as a future option."
- 🟡 **G2 (cap divisor at 1.0) — defer.** Behavior change: G2 caps `divisor.max(1.0)` so short refs are never promoted, only long refs are demoted. Trade-off: matches eDNA-user intuition ("trust longer refs more") but introduces an asymmetric bias that retains part of the natural length advantage of longer refs (long refs accumulate more raw hits → demotion alone doesn't fully equalize match-density-based scoring; no compensating short-ref boost). Defer until empirical data shows G1's promotion is hurting. Reversible from G1 → G2 with a single `.max(1.0)` if needed.
- 🟡 **Sqrt exponent change.** No derivation, but alternatives (linear, expected-hit count, per-group scaling) are bigger formula redesigns. Defer; pair with F2 evaluation if both end up being touched.
- ✅ **Method.html / README / walkthrough drift.** Folded into per-item actions above and the cross-cutting line-drift sweep.

### `rank_thresholds`

Audit: `thoughts/shared/research/2026-04-30-param-rank_thresholds-deep-dive.md`. Single read site at `src/classify.rs:1214-1231`. Wiring is clean; the threshold-walk loop has no bugs. Surface area is permissive (silent edge-cases) and undertested. Two items involve real code changes (validation + Python accessor), the rest are docs/tests.

- 🔧 **Expand doc-comment at `src/types.rs:339`.** Cover all four edge cases: empty list (currently silently behaves like None), longer list (extras ignored), shorter list (last value reused — already in README, mirror in source), and the fact that `threshold` is fully overridden when `rank_thresholds=Some(non-empty)` (no per-element fallback today).
- 🔧 **Add input validation at the Python binding** (real code change, Python-only). Verified: `src/lib.rs:185-208` `classify` pyfunction returns `PyResult<Vec<...>>` and can raise `PyValueError`; `id_taxa` cannot. Validate before struct construction:
  - Reject `Some(vec![])` with a clear error message ("rank_thresholds = Some([]) is ambiguous; pass None to disable per-rank thresholds").
  - Bound each element to `[0, 100]` (current confidence is on this scale).
  - No length check needed (longer-than-path is harmless).
- 🔧 **Add tests for the override path.** Positive case (per-rank values applied at correct depths), shorter-list reuse, lca_cap interaction (LCA-cap supersedes threshold walk), and the new validation errors from the previous bullet.
- 🔧 **Standardize example lengths across README and walkthrough.** README's `[90, 80, 70, 60, 50, 40, 40]` (7 elements) and walkthrough S9's `[50,50,50,50,95]` (5 elements) imply different tree depths. Pick one canonical depth (probably 7, matching vert-12S Domain → Species) and use it consistently in both surfaces. Update walkthrough S9 to match.
- 🔧 **Expose rank-count in the Python binding** (real code change). Verified: only `PreparedData` and `BuiltTree` are pyclasses; `TrainingSet` is NOT exposed, so this is **not** a "small additive getter." Two paths:
  - **Path A (preferred):** add a standalone helper `model_n_ranks(model_path: str) -> int` (and optionally `model_ranks(model_path: str) -> list[str] | None`) that loads the bincode model and returns the count. Minimal API surface, no new pyclass.
  - **Path B:** expose `TrainingSet` as a new `#[pyclass]` with `n_ranks` / `ranks` accessors. More invasive; only worth it if other model-introspection use cases pile up.
- ✅ **Line-number drift in method.html §9a and walkthrough S13.** Folded into the cross-cutting line-drift sweep.
- 🟡 **Per-element fallback to `threshold`** (`Option<Vec<Option<f64>>>` API, or sentinel like `-1.0` meaning "use global"). Currently a user must repeat the global threshold to partial-override (`[60, 60, 60, 60, 60, 60, 95]`). Verbosity is real but small in practice; defer until a concrete user complaint surfaces.
- 🟡 **Defaults helper (`rank_thresholds_for_depth(d, top_floor=80, leaf_strict=95)`-style constructor).** Pure UX sugar.
- 🟡 **Rename for directional hint** (`rank_thresholds_root_down`). Public API breakage; defer.
- ✅ **Not exercised in `classifier_benchmark.py` sweep harness.** External harness; out of scope for this repo.

### `full_length`

**Removed** (commit `f097f05` "Remove full_length classification parameter"). The audit at `thoughts/shared/research/2026-04-30-param-full_length-deep-dive.md` predates the removal. No orphaned references in `README.md`, `oxidtaxa_method.html`, or `oxidtaxa_walkthrough.html`. No follow-up actions needed.

### `sample_exponent`

Audit: `thoughts/shared/research/2026-04-30-param-sample_exponent-deep-dive.md`. **Important correction from the audit:** the 0.47 default is **not** oxidtaxa-novel — it's inherited verbatim from canonical IDTAXA (DECIPHER's R signature is `samples=L^0.47`; the "0.5" in Murali et al. is shorthand for `√L`). Oxidtaxa is faithful here. Single read site at `src/classify.rs:106-115` plus two parallel abstention guards. No code-logic changes.

- 🔧 **Fix doc-comment at `src/types.rs:331-333`** (comment-only). Currently says "L = unique k-mers in query." Reality is asymmetric: `S` is computed against `not_nas` (raw stream count after NA filter, before dedup) at `:107-115`, while the `too_few_kmers` abstention at `:210` and `:384` checks `my_kmers.len()` (dedup'd). The asymmetry is intentional (S sized against apparent query complexity; abstention checks unique-signal floor) and catches low-complexity queries correctly. Document the asymmetry rather than change the code.
- 🔧 **Harmonize range docs across surfaces.** Currently inconsistent: Rust core has no validation; Python wrapper validates `(0, 1]`; README says "0.2-0.8" without enforcement. Loosen rather than tighten — drop the range claim from `README.md:125`'s parameter table and frame the `[0.35, 0.40, 0.47, 0.55, 0.65]` values as the Tier 1 sweep recommendation only. Leave Python's `(0, 1]` as a sanity check; leave Rust permissive.
- 🔧 **Document the `5L/S` bootstrap cap + `bootstraps` coupling.** At `classify.rs:138`, `b = min(5L/S, bootstraps)` is the implicit Pareto knob — lowering `sample_exponent` lowers `S` and lets `b` climb toward the configured cap. Currently invisible in user-facing docs. Add one paragraph in method.html §7a explaining the interaction and cross-reference from §12 and `types.rs:331-333` doc-comment. Note the `5` multiplier is a magic number with no in-source justification.
- 🔧 **Add Rust tests for the abstention edge cases.** Particularly the asymmetric stream-vs-dedup case: a low-complexity fixture (e.g., 12 raw 8-mers, 3 unique) should fire `too_few_kmers` regardless of `sample_exponent` value. Lock the contract.
- ✅ **Walkthrough S12 line drift** (`classify.rs:303` → `:210` / `:384`). Folded into cross-cutting line-drift sweep.
- 🟡 **Range validation in Rust core.** Could add `debug_assert!(0.0 < config.sample_exponent && config.sample_exponent <= 1.0)`. Defer; Python clamps and there's no concrete bug from out-of-range values.

## Train-time parameters

### `descendant_weighting`

Audit: `thoughts/shared/research/2026-04-30-param-descendant_weighting-deep-dive.md`. Wiring is clean. All three modes (`Count`/`Equal`/`Log`) are genuinely distinct at the dispatch site `src/training.rs:975-981` — `q` differs at every internal node with unequal-sized children, and that propagates through cross-entropy ranking to the kept k-mer set. No no-op risk. No doc drift across README, method.html, walkthrough. No TODO/FIXME near read sites.

- 🔧 **Add three-mode-distinctness test.** Tests today only exercise `Count` (in `test_staged_training_equivalence`). Build three trees on a deliberately imbalanced fixture (e.g., 5-seq vs 2-seq sibling) with `Count` / `Equal` / `Log` and assert pairwise inequality of `decision_kmers[k].keep`. Closes the same coverage gap that hid the old uniform-scaling `leave_one_out` no-op.
- 🔧 **Expand doc-comment at `src/types.rs:270-272`** (comment-only, in source). Currently one line ("Strategy for weighting child profiles during feature selection. Default: Count (original behavior).") — terser than `leave_one_out`'s 8-line doc-comment or `suppress_ancestor_only_groups`'s 18-line one. Brief description of each mode (Count = canonical IDTAXA, raw descendant count; Equal = uniform 1/n; Log = ln(1+d) compressed).
- 🔧 **Document the `total_weight > 0` invariant at `merge_sparse_profiles`** (comment-only at `src/training.rs:802-834`). The function divides by `total_weight` without a guard at line 831. Currently unreachable on valid inputs (every leaf has `descendant_count = 1`), but a 1-line comment pinning the invariant hardens it against future refactors.
- 🔧 **Refactor `create_tree` signature** (cleanup, no behavior change). At `src/training.rs:345-352`, `_build_tree_inner` repacks `BuildTreeConfig` back into a `TrainConfig` with `..Default::default()` filler so it can call `create_tree(... &TrainConfig)`. Works today but masks build-config/train-config bleed if a future `TrainConfig` field is added without being copied here. Change `create_tree`'s signature to accept `&BuildTreeConfig` directly; drop the repack.
- 🔧 **Persist `descendant_weighting` on `TrainingSet`** (cleanup, bincode schema break). Currently models trained with different modes are indistinguishable from the classification side. Add `descendant_weighting: DescendantWeighting` to `TrainingSet` (analogous to the recent `use_idf_in_descent` addition at `src/types.rs:65`). Lets downstream tooling label models by their training-time weighting. Bumps bincode incompatibility — consistent with the breaks we've already taken for `leave_one_out`'s `raw_counts` and the planned `length_normalize` F2.
- 🟡 **Sqrt / continuous-alpha alternative modes.** `descendant_weight = d^alpha` with alpha=1.0 (Count), alpha=0.0 (Equal), alpha=0.5 (sqrt) — would let Optuna sweep continuously instead of discrete-categorical. Defer; current enum covers the canonical/innovation contrast.
- 🟡 **Default switch to `Equal`.** Real reference databases are imbalanced, so `Count` lets large subtrees dominate `q`. Audit makes the case for `Equal` as a more robust default. Defer — keeping `Count` honors the "every opt-in off ≈ stock IDTAXA" contract.

### `correlation_aware_features`

Audit: `thoughts/shared/research/2026-04-30-param-correlation_aware_features-deep-dive.md`. Wiring clean. Bhattacharyya metric on L1-normalized sqrt profiles is the deliberate fix for the original Pearson `n_children=2` degeneracy (flagged in `2026-04-15-new-parameter-audit.md`). The a2cd22a perf optimization (cached `max_corr`, parallel argmax+update gated at `n_cand >= 2048`, struct-of-arrays with flat row-major `profiles_flat`) is in place. No TODO/FIXME near read sites.

- 🔧 **Add the two missing tests for `correlation_aware_features`.** The audit overstated "zero direct tests": `tests/test_algorithmic_improvements.rs` already covers determinism (line 681), sequential-vs-parallel equivalence (line 648), Bhattacharyya valid-model end-to-end (line 717), and keep-set distinctness (line 613). Real gaps: (1) explicit `n_children=2` non-degeneracy fixture (locks the Bhattacharyya-vs-Pearson regression); (2) all-zero profile candidates aren't selected. Add these two tests; consider colocating in a new `tests/test_correlation_aware.rs` for grouping.
- 🔧 **Add a worked example in method.html §6d.** Currently `correlation_aware_features` is mentioned only as a parenthetical at line 781 — no worked numerical example showing redundancy penalty in action. Inconsistent with §6d's worked H matrix for the round-robin path. Borrow from walkthrough S18 (which has the conceptual example) and lift the `entropy * (1 - max_corr)` math into §6d.
- 🔧 **Fix §12 cross-reference.** Method.html §12 line 1679 cites §6d as the introduction; §6d does not actually introduce the parameter. Either expand §6d (covered by previous bullet) or relocate the §12 cross-ref.
- 🔧 **Expand walkthrough S18.** Currently shows the round-robin vs corr-aware kept-set comparison but doesn't mention the L1-normalized sqrt step, entropy-descending early-exit, or the a2cd22a cache. Add one explanatory paragraph linking the user-facing concept to the implementation.
- 🔧 **Expand doc-comment at `src/types.rs:290-294`** (comment-only). Add a one-line cost note: "O(R · C) per node where R = `record_kmers`, C = candidate pool size; parallelized when `n_cand >= 2048`."
- 🔧 **Document tie-break behavior at `src/training.rs:1119`** (comment-only). `max_corr` initialized to 0.0 means first pick is purely-entropy-driven; tie-breaks fall to permutation-sort stability. Worth a 1-line comment near `:1119`.
- 🔧 **Expand README.md line 41 inline comment.** Currently bare ("greedy feature selection with redundancy penalty"); no mention of Bhattacharyya or the cost. One sentence: "uses Bhattacharyya similarity on L1-normalized profiles; slower training, no impact on classify speed."
- 🟡 **`correlation_penalty_strength: f64` knob.** Would expose the implicit α=1.0 in `entropy * (1 - α * max_corr)`. Speculative — current behavior matches design intent. Defer until OAT data shows partial penalty would help.
- 🟡 **`corr_candidate_multiplier: f64` knob.** Hardcoded 2× headroom on the candidate pool at `:1040`. Defer.
- 🟡 **HashMap precompute for entropy lookup.** `:1067-1075` walks each child's sorted `H` list per candidate — O(R · C · n_children). HashMap would reduce to O(R · n_children + C). Defer until benchmarks show this is a bottleneck.

### `leave_one_out`

Audit: `thoughts/shared/research/2026-04-30-param-leave_one_out-deep-dive.md`. Per-kmer LOO landed in `84dd411`. Wiring clean (TrainConfig → LearnFractionsConfig → `_learn_fractions_inner` at `src/training.rs:481-498`). Formula is exact at leaf siblings, exact at internal siblings under default `Count` weighting, and approximate at internal siblings under `Equal`/`Log` weighting (noted in source at `:477-479`). Tests cover the no-op regression. No doc drift specific to LOO.

- 🔧 **Surface the internal-sibling × non-Count approximation in user-facing docs.** Source comment at `src/training.rs:477-479` calls it out; README/method.html/walkthrough don't. Add a one-line caveat to README's `leave_one_out` row at `:144` and a brief cross-reference between the `descendant_weighting` and `leave_one_out` rows: "LOO is exact at leaf siblings and internal siblings under `descendant_weighting=Count`; approximate at internal siblings under `Equal` or `Log`."
- 🔧 **Add an end-to-end byte-diff regression test.** Existing `test_leave_one_out_reshapes_profile` proves the formula isn't a uniform-scaling no-op, but no test asserts `fraction[]` differs between `LOO=true` and `LOO=false` on a real-shaped fixture. The bugfix plan deferred this on the grounds that small test datasets classify perfectly under any LOO setting (fractions stay at `max_fraction`). A fixture-backed assertion on a slice of vert12s known to fail under default-LOO would close the gap.
- 🔧 **Document multi-LOO scope.** Add a brief note (README or types.rs doc-comment) that LOO holds out one sequence at a time per descent iteration, matching the IDTAXA paper. No batched/multi-holdout exists; not in scope.
- 🟡 **OAT/Optuna sweep rerun.** Pre-fix sweep results at `thoughts/shared/research/2026-04-21-three-marker-sweep-findings.md` ran against the no-op and are now noise. Rerun is recommended but external (`assignment-tool-benchmarking` harness). Out of scope for this repo.
- 🟡 **Cargo.toml version bump.** Deferred — will be done as a single coordinated bump after all bincode-breaking changes (this audit cycle's `raw_counts`/`raw_totals`, the planned `descendant_weighting` persistence, and `length_normalize` F2's per-node averages) land together.
- ✅ **`use_idf_in_training` → `use_idf_in_descent` rename drift.** Visible at the same lines as LOO callouts but unrelated; covered under the `use_idf_in_descent` audit section (next).

### `use_idf_in_descent`

Audit: `thoughts/shared/research/2026-04-30-param-use_idf_in_descent-deep-dive.md`. End-to-end wiring is clean post `e5320a4`. Train and classify descent are unified: flag is persisted on `TrainingSet.use_idf_in_descent` (`src/types.rs:65`); train/classify-side weighted-profile construction blocks at `training.rs:500-515`, `classify.rs:262-275` (greedy), and `classify.rs:481-497` (beam) are byte-identical structural copies. Default stays `false` for canonical IDTAXA compatibility.

- ✅ **`use_idf_in_descent` rename in user-facing docs.** Verified DONE — zero remaining occurrences of `use_idf_in_training` in `README.md`, `oxidtaxa_method.html`, or `oxidtaxa_walkthrough.html` at HEAD. The audit's claim was stale (auditor was reading the pre-rename state). No action needed.
- 🔧 **Add an automated test that prediction differs under flag toggle.** Today's six tests are smoke + persistence + bit-identical-when-off; nothing locks in "flag=true changes predictions." The implementation note in `2026-04-30-unify-idf-in-descent.md:292-298` flagged this as manual-verification-only ("predictions differ" was confirmed via diagnostic instrumentation that was removed before commit). Construct a fixture where iterative fraction-learning misclassifies (e.g., a deliberate train/classify mismatch fixture) and assert `flag=true` vs `flag=false` produce different `taxon`/`confidence` values. Locks the contract that the unify did real work.

### `seed_pattern`

Audit: `thoughts/shared/research/2026-04-30-param-seed_pattern-deep-dive.md`. Wiring is clean. The model file is the single source of truth for the pattern (`src/classify.rs:94-97`), so train↔classify mismatch is impossible at the API level. Lazy validation via `parse_seed_pattern` (`src/kmer.rs:19-45`) catches empty/all-zeros/non-binary chars at training time. No TODO/FIXME near read sites.

- 🔧 **Add direct tests for spaced seeds.** Currently zero coverage for any non-None pattern. Add: (1) unit tests for `parse_seed_pattern` covering the success path (e.g. `"11011011011"` → weight 8, span 11) and all four rejection paths (empty, all-zeros, invalid char, whitespace); (2) an integration test that trains with `seed_pattern=Some(...)`, saves and reloads the model, classifies and compares against a golden reference. The walkthrough §T1/S16 SNP-collapse example (`k=5, seed_pattern="11011"` collapsing a 1-base mutation) is a natural template.
- 🔧 **Add spaced-seed coverage to `oxidtaxa_method.html`.** §6a (line ~613) discusses only contiguous k-mer enumeration; spaced seeds are absent end-to-end. The §3 differences-from-IDTAXA table also omits `seed_pattern` despite it being one of the 12 oxidtaxa-only training parameters. Add a §3 row + a body sub-panel in §6a (parallel to §8d for `suppress_ancestor_only_groups`). Cross-reference from README's §Spaced K-mers and walkthrough §T1.
- 🔧 **Fix walkthrough §T1 label confusion.** S16 uses `k=5` as the label, but the code would set `k = weight = 4` from `"11011"`. The label refers to span, not weight — relabel as `weight=4, span=5` to match the code's actual k semantics.
- 🔧 **Expand doc-comment at `src/types.rs:264-265`** (comment-only). Currently one line. Mention: validation rules (caught at first call to `train()`/`prepare_data()`); that classify reads it from the model (no override knob); that `weight = #1s` and `span = pattern length`; that passing both `seed_pattern` and `k` silently drops `k`.
- 🔧 **Reject weight > 15 in `parse_seed_pattern`** (real correctness gap). At weight > 15, `pwv[w-1] * 4` at `kmer.rs:489` silently overflows `i32` and produces corrupt k-mer IDs. The contiguous path is clamped to `max_k = 13` at `training.rs:124`; spaced has no analogue. Add a check in `parse_seed_pattern` (`src/kmer.rs:19-45`) that rejects `weight > 15` with a clear error, mirroring the existing empty/all-zeros/non-binary rejections. Small additive validation, no logic change.
- 🔧 **Validate `k == pattern.weight` when both are provided** (real consistency gap). Currently the code at `training.rs:115-132` silently drops `k` when `seed_pattern` is set. The Optuna harness and the README's framing treat `k = pattern.weight` as a paired invariant — make Rust enforce it. New match shape:
  ```
  (Some(pat), Some(k)) if k == pat.weight  → k = pat.weight   # consistent
  (Some(pat), Some(k)) if k != pat.weight  → ERROR             # mismatch
  (Some(pat), None)                        → k = pat.weight   # use pattern's weight
  (None, Some(k))                          → k = k             # contiguous
  (None, None)                             → k = auto
  ```
  No-op for the existing Optuna sweep (all configs already satisfy `k == pat.weight`); surfaces hand-written or harness-bug mismatches loudly instead of silently using the pattern's weight.
- 🟡 **Spaced-path performance optimization.** Per-window cost is O(weight) at `kmer.rs:437-449` vs contiguous's O(1) rolling update — ~6-9× slowdown for typical patterns. Unmeasured. Defer until benchmarks (or production complaint) show this matters at training scale.
- 🟡 **Add a bench for spaced vs contiguous enumeration.** Would let us characterize the slowdown above. Defer.
- 🟡 **Recommended pattern catalog.** README has 8 example patterns advisory only. A `const RECOMMENDED_SEED_PATTERNS: &[&str]` array (or Python helper) would surface known-good patterns. Defer; README guidance is sufficient for now.
- 🟡 **Min-weight check.** Weight=2 collapses index space to 16 entries (useless for classification) but isn't rejected. Defer; users pasting a weight-2 pattern is unlikely.
- 🟡 **Span-aware quality filter.** `lib.rs:374` filters seqs shorter than 30 bp; a span-50 pattern silently drops most short seqs to empty. Defer.

## Always-on architecture

### `per_rank_idf`

Audit: `thoughts/shared/research/2026-04-30-param-per_rank_idf-deep-dive.md`. Architectural always-on decision (no `TrainConfig` field, no kwarg). Wiring is clean. Compute site at `src/training.rs:306`, formula at `:741`, four read sites (`training.rs:462`, `classify.rs:245`, `:462`, `:691`). Memory cost is negligible (3.7-14.7 MB at production scale).

- 🔧 **Add test coverage for shallower IDF rows.** Today's only assertion is `compare_training_set` at `tests/test_training.rs:117-137` comparing the *deepest* row byte-equivalent to R's single global IDF. No test exercises a synthetic case where row 0 differs from row K-1 (e.g., a k-mer that's universal at species but only partly-shared at kingdom — the walkthrough §T2 ACG example would lift directly into a fixture). Also: no test for the depth-selection index math at the four read sites; no test for the negative-IDF case (all-shared k-mer at shallow rank).
- 🔧 **Expand doc-comments at `src/types.rs:55-58` and `src/training.rs:741`** (comment-only). Mention: (a) the negative-IDF case for fully-shared k-mers is **intentional** — `c ≈ N_r` gives `ln(N_r/(1+N_r)) < 0`, and universal-at-this-rank k-mers SHOULD actively downweight the score because they carry no discriminative signal; (b) the prefix-truncation behavior for variable-depth lineages (`prefixes_at_rank` caps `r` at `parts.len()`, so shallow lineages contribute to deep IDF rows as if they were species-resolved).
- 🔧 **Extract depth-selection helper** (cleanup, no behavior change). The expression `(ts.levels[k_node] - 1).max(0) as usize; depth.min(ts.idf_weights_by_rank.len().saturating_sub(1))` is replicated verbatim at four sites: `training.rs:462`, `classify.rs:245`, `:462`, `:691`. Extract as a single helper (e.g., `idf_row_for_depth(&[Vec<f64>], i32) -> &[f64]`) and call from all four sites. Prevents future drift if the rank-encoding convention ever changes.
- ✅ **Method.html §6b stale file:line refs** (`training.rs:313` → `:306` for compute, `classify.rs:552` → `:691-693` for leaf-phase, plus the new descent reads at `:244-247` and `:461-464`). Folded into cross-cutting line-drift sweep.
- ✅ **`use_idf_in_training` rename drift.** Covered under `use_idf_in_descent` section.
- 🟡 **Sparsity exploitation.** Compute uses dense `vec![0.0f64; n_kmers]` per chunk. Most rare-everywhere k-mers carry `ln(N_r)` redundantly across rows. Compression possible but sub-megabyte gain at current scale. Defer.
- 🟡 **In-process memoization.** Per-rank compute is parallelized but not memoized. Staged API's disk cache substitutes. Defer.
- 🟡 **Make per-rank tunable** (`use_per_rank_idf: bool` flag). No proposal in `thoughts/`. Defer; "always on" is the current commitment.
- 🟡 **`max_rank = 0` defensive guard.** Structurally unreachable (early-out at `training.rs:27` when `l < 2`), but the IDF code itself has no check. Defer; not worth a guard for a structurally-unreachable case.

---

## Cross-cutting actions
*(populated as themes emerge from per-parameter discussion)*
