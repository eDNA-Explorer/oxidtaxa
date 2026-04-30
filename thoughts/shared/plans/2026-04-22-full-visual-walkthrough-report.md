# Oxidtaxa Full Visual Walkthrough Report — Implementation Plan

## Overview

Two artifacts:

1. **Corrections to `oxidtaxa_method.html`** — line-number drift in ~7 code references, and nine parameters missing from the §12 appendix. Fix in place.
2. **New `oxidtaxa_walkthrough.html`** — a sibling document to the method doc, modeled on §10 but covering the *entire* training+classification pipeline end-to-end, with every knob called out at its point of influence, and every notable scenario depicted with concrete numbers. Where §10 shows one query running through classification, this new doc shows the model being built (§6-style knobs visually) and every qualitatively-distinct classification outcome, each on its own scenario panel, culminating in a single "whole pipeline" trace that includes training and classification.

## Current State Analysis

`oxidtaxa_method.html` (1621 lines) is broadly correct and well-structured. Cross-check against `src/training.rs` (1185 lines) and `src/classify.rs` turned up:

### Accurate (verified against code)

- §6b IDF formula `ln(N_r / (1 + c))` with weighted `c` — matches `compute_idf_at_rank` (`training.rs:663–708`).
- §7b intra-bootstrap ties give every tied child a vote — matches `classify.rs:224–226`.
- §8a widening rule `vote_counts[j] >= 0.5·B || j == winner` — matches `classify.rs:265–268`.
- §8c co-winner rule `tot_hits[g] >= max_tot · (1 − tie_margin)` — matches `classify.rs:743`.
- §9a contiguous-prefix break-on-fail — matches `classify.rs:853–868`.
- §8b rollup from `tot_hits` / B — matches `classify.rs:773–786`.
- All defaults in the doc match `types.rs:275–353`.

### Line-number drift (must fix)

| HTML anchor | Cited | Actual | Note |
|---|---|---|---|
| §6b param-box | `training.rs:685` | `training.rs:706–707` | IDF `ln(N_r/(1+c))` formula |
| §6b param-box | `classify.rs:504` | `classify.rs:552–554` | Per-rank IDF row lookup |
| §7c param-box | `types.rs:315` | `types.rs:318` | `beam_width` field |
| §7c param-box | `classify.rs:502` | `classify.rs:503–504` | beam sort + truncate |
| §7c param-box | `classify.rs:510` | `classify.rs:511–524` | leaf-phase across beam |
| §8c param-box | `classify.rs:706` | `classify.rs:742–743` | `tie_margin` cutoff |
| §6e | `training.rs:542–558` | `training.rs:550–570` | nudge-down block |

### Missing from §12 parameter index (nine fields)

From `TrainConfig` (`types.rs:237–294`):

- `n` (default 500.0) — drives auto-k calculation when `k=None`.
- `min_fraction` (0.01) — fraction-learning floor.
- `max_fraction` (0.06) — fraction-learning initial / ceiling.
- `max_iterations` (10) — outer loop cap.
- `multiplier` (100.0) — scales `delta` (per-failure fraction decrement).
- `max_children` (200) — feature-selection cap; above this, node becomes opaque.

From `ClassifyConfig` (`types.rs:298–334`):

- `full_length` (0.0) — leaf-phase length-band filter.
- `length_normalize` (false) — `1/sqrt(n_unique/avg)` leaf-phase scaling.
- `confidence_uses_descent_margin` (false) — per-rank margin discount (commit `622ca33`, `e287eb7`, `17b492b`, `79ecdb9`).

### Scenario coverage already in §10 (keep)

- Clean descent (Root→Animalia, 98/2).
- Strict-gate halt (Canidae 95/5, fails 0.98 gate).
- "Leading but not decisive" halt (Canis 92/8).
- Terminal three-way with close runner-up (62/30/8).
- Convergence over bootstrap count (10c grid).
- End-to-end trace (10d).

### Scenarios *not yet* depicted in §10 (must add in new walkthrough)

1. **Fraction-learning visualization** — how `training_threshold`, `min_fraction`, `max_fraction`, `multiplier` move `fraction[node]` over iterations; when a node gets masked.
2. **Feature selection under `descendant_weighting`** — how Count/Equal/Log change the parent profile `q` at a lopsided node.
3. **Correlation-aware selection** — picking non-redundant k-mers vs round-robin at a node with correlated top-entropy features.
4. **Max-children opaque node** — what happens when a genus has > 200 children.
5. **Beam descent rescuing a runner-up** — 58/40 mid-tree split, greedy picks 58 and misclassifies; beam (K=2) keeps both, leaf-phase picks the correct 40-side.
6. **Sibling-aware + tie-margin composition** — the "one without the other is inert" case.
7. **Confidence-margin discount** — near-tied descent (60/40) raises margin 0.2 → effective multiplier `0.8 + 0.2·0.2 = 0.84`, so reported confidence 90 → 75.6 at that rank.
8. **Full-length filter drops all candidates** — query L=400 with `full_length=1.2`; a reference with 600 unique k-mers gets filtered out → `reject_reason="no_training_match"`.
9. **Length-normalize damping** — long reference (600 unique) vs avg (400) is divided by `sqrt(1.5) ≈ 1.225`, demoting it out of a false tie.
10. **Singleton-leaf gotcha visualized** — a species with one training seq produces meaninglessly high `tot_hits`.
11. **Abstention paths** — `too_few_kmers` (query too short), `no_training_match` (everything filtered), `below_threshold` (prefix truncated to Root only).
12. **`tied_species` alternative-reporting path** — winners = 2 groups, LCA clamp at genus, `alternatives=[A,B]`, `reject_reason=None`.
13. **IDF training/classify alignment** — node fails to train under `use_idf_in_training=false`, trains correctly under `true`.
14. **Leave-one-out actually firing** — group_size ∈ [2,5]; scale 0.5–0.8 applied; effect on fraction learned.
15. **Spaced seed collapses a SNP** — seed `110110110` causes two references differing at a "0" position to collide into the same k-mer ID.

## Desired End State

After this plan is executed:

- `oxidtaxa_method.html` has every line reference verified against HEAD, and its §12 appendix lists all 23 fields across `TrainConfig` + `ClassifyConfig` (excluding `processors`, which has no algorithmic effect and can be footnoted).
- A new file `/Users/ryanmartin/oxidtaxa/oxidtaxa_walkthrough.html` exists, containing a scenario-driven pipeline trace that depicts all 15 scenarios above plus the six already-covered §10 scenarios, all running against a single expanded toy dataset whose arithmetic is tractable by hand.
- Both files render cleanly in a browser with no broken internal links and shared visual style (reuse the existing `--accent-*` palette and SVG markers).

### Verification

- `grep -nE 'src/(classify|training|types)\.rs:[0-9]+' oxidtaxa_method.html` — every hit resolves to a live line that implements the claim.
- Spot-check each numeric example in the new walkthrough by running the toy dataset through `cargo run --example walkthrough` (see Phase 3).
- Open both HTML files in a browser — no missing icons, no CSS bleed, scenarios land where called.

### Key discoveries

- **`confidence_uses_descent_margin` is per-rank, not cumulative** (commit `17b492b`). The multiplier is `MARGIN_FLOOR + (1 − MARGIN_FLOOR) · m` with `MARGIN_FLOOR = 0.8` (`classify.rs:804–812`). The beam path inherits margins per candidate (`classify.rs:392–407`); pass-through beam steps do not push a new margin.
- **`sibling_aware_leaf` + `tie_margin` are the designed pair.** Widening the candidate set without loosening the tie test keeps the siblings exact-matched out; loosening the tie test without widening the set has nothing to tie against. Baseline (both off) is strict IDTAXA.
- **`tied_species` is NOT a `reject_reason` value.** Tied-leaf behavior is signaled via the `alternatives` list plus LCA clamp; `reject_reason` remains `None`. The current doc §9b example is therefore *incorrect* (shows `reject_reason = "tied_species"`). This is a bug to fix in the existing doc, not just the new one.
- **Singleton-leaf + `sibling_aware_leaf` cannot produce ties by themselves.** The LCA clamp requires `winners.len() > 1`, which needs a real sibling group in `keep` — `sibling_aware_leaf` widens to include them, but only `tie_margin > 0` promotes a close score to co-winner. Without tie_margin, exact equality is the only path.
- **`max_children` makes a node opaque.** Genus with 250 species → no `DecisionNode` emitted, k-mers lumped into one sparse profile seen by the parent family node. Classify cannot descend through an opaque node.
- **`full_length` is a hard filter.** Out-of-band references are dropped *before* leaf-phase match. Empty `keep` after filtering → `reject_reason="no_training_match"`.

## What We're NOT Doing

- Not rewriting §§1-9 of `oxidtaxa_method.html`. That doc already has the mental model; corrections are surgical.
- Not removing or restructuring §10. The new file *extends* §10's idea at full scope; §10 stays as the introductory walkthrough.
- Not making the walkthrough interactive (no JS toggles, no hover tooltips). Static HTML + inline SVG, like the method doc.
- Not refactoring the toy dataset in §5 (ACGTACGTAC et al.). The existing dataset stays for §§6-9 numeric continuity; the new walkthrough uses its own larger extension.
- Not documenting `processors` or `seed` (implementation plumbing, no user-visible algorithmic effect).
- Not adding parameters that don't exist yet. Scope is what's in HEAD.

## Implementation Approach

Build as two independent work streams. Stream 1 patches the existing doc (low-risk, mechanical edits). Stream 2 produces the new walkthrough file (larger, creative work). No merge conflicts — they touch disjoint sets of lines / files.

Toy dataset for Stream 2 is an *extension* of §5, not a replacement: 6 more sequences are added to push the tree past the scenarios §5 can demonstrate (a 5-child genus to show lopsided `descendant_weighting`, a singleton species for the gotcha, a near-sibling pair for `tie_margin`, a long reference for `full_length`/`length_normalize`).

---

## Phase 1: Correct `oxidtaxa_method.html`

### Overview

Fix the seven line-number drifts, add the nine missing appendix rows, and correct the `tied_species` reject_reason example in §9b.

### Changes Required

#### 1.1 Line-number corrections

**File**: `oxidtaxa_method.html`

Edit the `<code class="inline">…</code>` references listed in the Current State table. Each is a single-string replacement. Apply `replace_all` where the old string occurs once.

| Section | Find | Replace |
|---|---|---|
| §6b | `src/training.rs:685` | `src/training.rs:707` |
| §6b | `src/classify.rs:504` | `src/classify.rs:552` |
| §6e | `src/classify.rs:210–245` | `src/classify.rs:210–245` *(verify, likely correct)* |
| §6e | `src/training.rs:542–558` | `src/training.rs:550–570` |
| §7c | `src/types.rs:315` | `src/types.rs:318` |
| §7c | `src/classify.rs:289` | `src/classify.rs:289` *(correct, leave)* |
| §7c | `src/classify.rs:307` | `src/classify.rs:307` *(correct, leave)* |
| §7c | `src/classify.rs:502` | `src/classify.rs:503` |
| §7c | `src/classify.rs:510` | `src/classify.rs:512` |
| §8c | `src/classify.rs:706` | `src/classify.rs:743` |

#### 1.2 Fix §9b `reject_reason` error

**File**: `oxidtaxa_method.html` (lines ~1117, ~1124)

The Scenario B output shows `reject_reason = "tied_species"`, but the code never emits that string — tied-species behavior is signaled by non-empty `alternatives` with `reject_reason = None`. Change the Python and TSV examples to:

```
taxon         = ["Root", "Animalia", "Canidae", "Canis"]
confidence    = [100, 98, 95, 92]
alternatives  = ["Canis_latrans", "Canis_lupus"]
reject_reason = null
```

TSV row: drop the `tied_species` value in the `reject_reason` column (leave empty or omit).

#### 1.3 Extend §12 parameter index

**File**: `oxidtaxa_method.html` (table ~1594–1613)

Add nine rows in alphabetical order. Each row: `<tr><td><code class="inline">name</code></td><td>phase</td><td>§ref or new</td><td>description</td></tr>`.

- `confidence_uses_descent_margin` — classify — new §, "Per-rank confidence multiplied by the descent margin observed at each split."
- `full_length` — classify — new §, "Relative length band filter for leaf-phase references; 0.0 disables."
- `length_normalize` — classify — new §, "Scale leaf-phase hits by √(avg/n) to dampen longer-reference inflation."
- `max_children` — train — §6d, "Max child count at which feature selection runs; above, the node becomes opaque."
- `max_fraction` / `min_fraction` — train — §6e, "Ceiling/floor on per-node learned sampling fraction."
- `max_iterations` — train — §6e, "Outer-loop cap for fraction learning."
- `multiplier` — train — §6e, "Per-failure fraction decrement scale."
- `n` — train — §6a, "Target reference count used only for auto-k estimation."

#### 1.4 Note `use_idf_in_training` matrix usage

**File**: `oxidtaxa_method.html` (§6e note on `use_idf_in_training`)

Small clarification: when on, the per-rank IDF row matching the descent node's depth is used (`training.rs:462–466`), not a global vector — this aligns training and classify scoring. Current text implies a single vector; revise to match matrix behavior.

### Success Criteria

#### Automated

- [x] All line refs in `oxidtaxa_method.html` resolve to live code: `python3 -c "$(cat <<'EOF'
import re, pathlib
html = pathlib.Path('oxidtaxa_method.html').read_text()
for m in re.finditer(r'src/(classify|training|types)\.rs:(\d+)', html):
    f, ln = m.group(1), int(m.group(2))
    lines = pathlib.Path(f'src/{f}.rs').read_text().splitlines()
    assert ln <= len(lines), f'{f}.rs:{ln} out of range'
print('OK')
EOF
)"` — prints `OK`.
- [x] `grep -c 'tied_species' oxidtaxa_method.html` — prints 0.

#### Manual

- [ ] Browser-render — layout unchanged, no broken code spans.
- [ ] Every line ref points to code that substantively matches the surrounding claim (skim each in turn).
- [ ] §12 table contains all 23 fields in alphabetical order.

---

## Phase 2: Design toy dataset &amp; scenario matrix

### Overview

Define the extended toy dataset and the scenario-to-panel mapping, so Phases 3–5 know what each visualization depicts. No code changes in this phase — just an anchor in the plan.

### Extended toy dataset

Starting from §5's 6 sequences, add 6 more to expose every scenario. Target `k=3` still (tractable arithmetic, 64-k-mer universe).

| ID | Sequence | Taxonomy | Purpose |
|---|---|---|---|
| seq1–6 | (as §5) | (as §5) | Baseline scenarios |
| seq7 | `ACGTACGATT` | Animalia;Canidae;Canis;*Canis_familiaris* | 3rd Canis species for three-way terminal + sibling_aware_leaf |
| seq8 | `GGCCGGCCGT` | Animalia;Felidae;Felis;*Felis_silvestris* | Near-tied F. catus sibling for tie_margin |
| seq9 | `TTAATTAACC` | Plantae;Rosaceae;Rosa;*Rosa_gallica* | 2nd Rosa species so Rosa isn't singleton |
| seq10 | `CGATCGATCGATCGATCG...` (≥60 bp) | Animalia;Canidae;Vulpes;*Vulpes_vulpes* | Long reference for `full_length` / `length_normalize` scenarios |
| seq11 | `ACGTACGTAG` (singleton) | Animalia;Canidae;Canis;*Canis_aureus* | Singleton-leaf gotcha — one seq under species |
| seq12 | `GGCCTTAACG` (chimeric) | Animalia;Felidae;Lynx;*Lynx_rufus* | A Felidae non-Felis to create Felidae→{Felis, Lynx} split for beam rescue |

Scenarios can be worked out with pen/paper against the 3-mer presence matrix — the new walkthrough's arithmetic panels lift directly from those counts.

### Scenario matrix

Each row defines one visualization panel in Phase 4.

| # | Scenario | Params | Panel type |
|---|---|---|---|
| S1 | Clean descent, leaf reached | defaults | Histograms (like §10b) |
| S2 | Mid-tree halt (scattered votes) | defaults | Histograms + `w_indices` annotation |
| S3 | Mid-tree halt (leading-but-not-decisive) | defaults | Histograms |
| S4 | Terminal near-tie w/o sibling_aware_leaf | defaults | Bootstrap grid |
| S5 | Terminal near-tie w/ sibling_aware_leaf on | `sibling_aware_leaf=true` | Bootstrap grid, side-by-side w/ S4 |
| S6 | Terminal near-tie → LCA clamp | `sibling_aware_leaf=true, tie_margin=0.05` | Tree w/ LCA cap drawn |
| S7 | Beam rescues runner-up | `beam_width=2` | Two-column tree |
| S8 | Descent-margin discount | `confidence_uses_descent_margin=true` | Per-rank confidence bars, before/after |
| S9 | Singleton-leaf gotcha | defaults | `tot_hits` bar chart |
| S10 | `full_length` filters everything → `no_training_match` | `full_length=1.2`, long query | Reference-length band diagram |
| S11 | `length_normalize` demotes long reference | `length_normalize=true` | Before/after hits chart |
| S12 | `too_few_kmers` abstention | very short query | Input box + unclassified output |
| S13 | `below_threshold` abstention | high `threshold`, weak query | Rollup bars w/ red cutoff |
| S14 | IDF training/classify alignment | `use_idf_in_training` off vs on | Fraction-learning curves |
| S15 | `leave_one_out` scale factor | group_size=3, `leave_one_out=true` | Profile diff chart |
| S16 | Spaced seed collapses SNP | `seed_pattern="110110110"` | K-mer id table |
| S17 | `descendant_weighting` flips feature pick | Count vs Equal at 5-child node | Profile vector diagram |
| S18 | Correlation-aware selection de-dupes | `correlation_aware_features=true` | Two selected-k-mer lists |
| S19 | `max_children` opaque node | genus w/ 250 children (synthetic) | Tree w/ opaque box |
| S20 | Fraction-learning convergence | default loop | Fraction-over-iterations line chart |
| S21 | Convergence over bootstrap count | (as §10c) | Grid, extended to cover S13 |

### Success Criteria

#### Automated

- [ ] (none; this phase is a design anchor)

#### Manual

- [ ] Every scenario has a unique parameter state + a unique expected-output story.
- [ ] Toy dataset sequences produce arithmetic-tractable k-mer counts (each ≤ 12 bp).

---

## Phase 3: Build training-side walkthrough

### Overview

Sections 1–4 of the new `oxidtaxa_walkthrough.html`: reuse the color palette and param-box style from the method doc; cover training params visually.

### Changes Required

#### 3.1 File scaffold

**File**: `oxidtaxa_walkthrough.html` (new, sibling to method doc)

Copy the `<style>` block and CSS variables from `oxidtaxa_method.html` verbatim. Header: "Oxidtaxa — Full Visual Walkthrough". Subtitle: references §10 of method doc, positions this as the full-scope scenario catalog.

#### 3.2 Section T1 — dataset

Extended toy dataset table (Phase 2). Presence/absence matrix for all 64 possible 3-mers, but grouped by column family so the reader can scan.

#### 3.3 Section T2 — k-mer enumeration + IDF matrix (scenario S16)

Two panels side-by-side: contiguous k-mers vs spaced-seed `110110110` (weight 6 → would need k=6; instead use `1101101` with weight 5 for a small demo). Show one pair of sequences collapsing to the same k-mer ID under the spaced seed.

Add a per-rank IDF table (Kingdom / Family / Genus / Species rows × k-mer columns) computed from the extended dataset — so the reader sees the *matrix*, not just a vector.

#### 3.4 Section T3 — feature selection (scenarios S17, S18, S19)

- **S17**: at the Root→Kingdom→Family tree where Animalia has 9 seqs, Plantae has 2, show profile `q` under each `descendant_weighting`. Bar chart of `q` for ACG under Count (Animalia dominates), Equal (each child weights 1), Log (compressed).
- **S18**: synthesize a 5-child node, hand-pick two highly-correlated top-entropy k-mers (identical profiles). Show round-robin keeping both; show correlation-aware keeping only one because BC = 1 zeros out the second's gain.
- **S19**: cartoon 3-panel ("normal node → split", "opaque node → concat"). Label the DecisionNode slot as `None`.

#### 3.5 Section T4 — fraction learning (scenarios S14, S15, S20)

- **S20**: line chart of `fraction[Canis]` across 10 iterations under defaults. Show it dropping from 0.06 toward 0.01, crossing `min_fraction` on iteration 7, becoming `None` (masked).
- **S14**: same fraction evolution, on vs off `use_idf_in_training`. Show that with IDF on, the node converges by iteration 3 and stays at max; off, it bounces more.
- **S15**: one node with group_size=3, LOO scale 2/3. Show profile weight column (before / after scaling). Note that LOO only fires at group_size ∈ [2,5].

### Success Criteria

#### Automated

- [x] File `oxidtaxa_walkthrough.html` exists at repo root.
- [x] HTML validates: `python3 -c "import html.parser; html.parser.HTMLParser().feed(open('oxidtaxa_walkthrough.html').read())"` → no exceptions.

#### Manual

- [ ] All five scenarios (S14–S20) appear exactly once.
- [ ] Numbers on panels match hand-computed values against the extended dataset.
- [ ] Style matches method doc — no visual drift.

---

## Phase 4: Build classification-side walkthrough

### Overview

Sections 5–10 of the new HTML. One scenario per panel, grouped by which of the three stages the knob lives in.

### Changes Required

#### 4.1 Section C1 — Stage 1 scenarios (S1, S2, S3, S7, S8)

Histogram panels (reuse `.hist` CSS). For S7 (beam), draw two trees side-by-side as in method §7c. For S8 (descent margin), bar chart of rank confidences before/after the multiplier, with the 0.8 floor drawn as a dashed baseline.

#### 4.2 Section C2 — Stage 2 scenarios (S4, S5, S6, S9, S10, S11)

- **S4/S5** are a single comparison panel (sibling_aware_leaf off vs on).
- **S6** overlays an LCA clamp on the tree SVG.
- **S9** (singleton-leaf): bar chart of `tot_hits` shows the lone-species group at ~60 purely because every bootstrap trivially "picks" the only reference — follow with a labeled arrow "this 60 is not supported signal."
- **S10**: reference-length band diagram (min, max annotated from `(1/1.2, 1.2) · L`); a reference sitting outside the band is greyed out; `keep` becomes empty → `reject_reason="no_training_match"`.
- **S11**: two bars per reference (raw vs normalized by `1/sqrt(n/avg)`); show the long reference demoted out of a tie.

#### 4.3 Section C3 — Stage 3 scenarios (S12, S13)

- **S12** (`too_few_kmers`): input panel showing a 4-bp query + S=2 gate; `my_kmers.len() <= s` → abstention. Result box shows `ClassificationResult::unclassified("too_few_kmers")`.
- **S13** (`below_threshold`): stack of rank bars with red threshold line crossing between rank 1 and rank 2; truncated taxon `["Root", "Animalia"]`, `reject_reason="below_threshold"`.

#### 4.4 Section C4 — alternatives output (S6 continued)

A TSV/JSON side-by-side showing the exact shape: `alternatives` non-empty, `reject_reason` null. Correct from the very start (this is what §9b of the method doc *should* have shown).

### Success Criteria

#### Automated

- [x] Every `ClassifyConfig` field (except `processors`) appears in at least one scenario panel: grep-check on field names.

#### Manual

- [ ] Each scenario's numbers trace back to the toy dataset.
- [ ] The three color bands (blue/orange/purple from method §10) are preserved.
- [ ] Color-blind legibility: every color carries a text label too.

---

## Phase 5: "Whole pipeline" mega-trace

### Overview

One full-width SVG stitching Training → Classification, similar in spirit to method §10d but wider: adds the training column on the left, shows the learned `fraction[Canis]` as a gauge feeding into the descent gate on the right.

### Changes Required

#### 5.1 SVG layout

**File**: `oxidtaxa_walkthrough.html` (new bottom section)

Columns (left→right):

1. **Training dataset** — seq1–12 listed; taxonomy tree sketch.
2. **Prepare** — k-mer enumeration + per-rank IDF matrix (miniature).
3. **Build tree** — feature-selection panel per node (shows `record_kmers_fraction`, `descendant_weighting`, `correlation_aware_features` as active knob labels on the arrow into this column).
4. **Fraction learn** — one node highlighted with its `fraction` trajectory and final value; `training_threshold`, `use_idf_in_training`, `leave_one_out` on the incoming arrow.
5. **Query input** — the toy query k-mer pool.
6. **Stage 1** — the same mini-tree from §10d, with the learned fraction visibly plugged into the Canis node.
7. **Stage 2** — tot_hits bars + LCA cap.
8. **Stage 3** — threshold cutoff.
9. **Output box** — final taxon/confidence/alternatives.

Knob callouts on every arrow, colored purple (`--accent-4`) like the existing param boxes. The legend at the bottom maps colors to stages.

#### 5.2 Footer

Cross-link back to `oxidtaxa_method.html` §10 ("for the abstract three-filter framing, see method doc §4; for classification-only pipeline, §10d").

### Success Criteria

#### Automated

- [ ] SVG renders in headless Chromium: `chromium --headless --screenshot=out.png file:///.../oxidtaxa_walkthrough.html` and `out.png` > 0 bytes (if Chromium available; skip otherwise).

#### Manual

- [ ] Every training and classify parameter from Phase 1.3 appears as a knob callout on exactly one arrow (no double-counting, no omissions).
- [ ] The learned fraction from the training column visibly connects to the Stage 1 gate in the classify column (an actual drawn line, not just text).
- [ ] Page prints landscape legibly (width ≤ 1400, height ≤ 900 for the main trace).

---

## Phase 6: Review &amp; sync pass

### Overview

Final cross-check between method doc and walkthrough doc: defaults, formulas, scenario numbers, and cross-links.

### Changes Required

- For each scenario in the walkthrough, verify the claimed outcome by running the toy dataset through an ad-hoc Rust test (Phase 6.1 below).
- Ensure every parameter in §12 of method doc has a corresponding scenario in the walkthrough (and vice versa).
- Cross-link: add a one-line note at the top of method §10 pointing readers to the walkthrough for "the full scenario catalog."

#### 6.1 Optional: driver test

**File**: `examples/walkthrough.rs` (new, `examples/` already exists)

A small Rust binary that builds the extended toy `TrainingSet` at `k=3` and classifies each scenario's query. Prints each scenario's expected output in a format the author can diff by eye against the HTML. Not wired into CI — it's a reproducibility aid for the doc author.

### Success Criteria

#### Automated

- [ ] Format / lint passes on any new Rust: `cargo fmt --check`, `cargo clippy -- -D warnings`.
- [ ] `cargo run --example walkthrough` prints non-empty output with every scenario label.

#### Manual

- [ ] Spot-check five scenarios: numbers in the HTML match `examples/walkthrough.rs` output.
- [ ] No scenario is claimed in Phase 4 but absent from the extended toy dataset's reachable space.
- [ ] Both HTML files open in Safari and Firefox without layout bugs.

---

## Testing Strategy

No unit tests to add. Verification is visual + arithmetic:

- **Arithmetic**: for every number that appears in the new walkthrough, there is a matching derivation (either a reproducible calculation in `examples/walkthrough.rs` or a pen-and-paper derivation in a comment in the HTML source).
- **Visual**: side-by-side browser render of method doc vs walkthrough to confirm style parity.
- **Parameter coverage**: automated grep that every `TrainConfig` / `ClassifyConfig` field name from `types.rs` appears at least once in `oxidtaxa_walkthrough.html` (or is documented as deliberately omitted, currently only `processors`).

## Performance Considerations

Not applicable — documentation only. The walkthrough HTML should be < 500 KB gzipped.

## Migration Notes

Not applicable — no code migration; only docs and an optional example binary.

## References

- Existing method doc: `oxidtaxa_method.html`
- Training code: `src/training.rs`
- Classification code: `src/classify.rs`
- Config types: `src/types.rs:237–353`
- Recent margin-handling commits: `e19df51`, `622ca33`, `e287eb7`, `17b492b`, `79ecdb9`
- Prior plan on tied-species reporting: `thoughts/shared/plans/2026-04-08-tied-species-reporting.md`
- Prior plan on oxidtaxa features: `thoughts/shared/plans/2026-04-06-oxidtaxa-features.md`
