---
date: 2026-04-19T11:30:00-04:00
researcher: Claude
git_commit: 36eed37a8f8a8028a5ec3d1549f270e48e4189ed
branch: main
repository: oxidtaxa
topic: "Algorithmic improvements to oxidtaxa for robustness to species/genus holdout and mixed-depth (LCA) reference scenarios"
tags: [research, classify, training, holdout, robustness, abstention, lca, tied-species, confidence, benchmark]
status: complete
last_updated: 2026-04-19
last_updated_by: Claude
---

# Research: oxidtaxa Logic Improvements for Holdout-Tier Robustness

**Date**: 2026-04-19
**Researcher**: Claude
**Git Commit**: 36eed37
**Branch**: main
**Repository**: oxidtaxa

## Research Question

Continue searching for algorithmic / logic improvements in **oxidtaxa itself** (not the benchmark harness).
Context: `~/assignment-tool-benchmarking/projects/assignment_benchmarks` builds synthetic eDNA communities
with four holdout tiers — Normal, Haplotype, Species-holdout, Genus-holdout — plus mixed-depth
`[genus]X` / `[family]Y` pseudo-taxa from LCA-truncated reference entries. The benchmark forces the
classifier to handle queries whose true taxon is progressively less represented in the reference. The
user is seeing weak behavior on the species- and genus-holdout tiers and wants the classifier to be
more robust to those scenarios.

Prior research documents the benchmark harness side of this problem
(`thoughts/shared/research/2026-04-17-species-genus-holdout-tier-improvements.md`). This document
focuses on oxidtaxa-internal changes.

## Summary

Thirteen algorithmic improvements are identified, ranked by expected impact on holdout-tier metrics. The
dominant issue is **information loss at the abstention boundary**: the output layer collapses two
distinct classifier states (no signal at all vs. real leaf-phase signal that didn't clear threshold
at any rank) into the same `\t\t0\t<alts>` TSV row. The in-memory `ClassificationResult` carries
`confidence[0] = c0` where `c0 > 0` for the latter case, but the TSV writer hardcodes `0` for both.
The benchmark's empty-path rates (37.5% of species-holdout rows, 63.4% of genus-holdout rows in one
recent sweep) therefore aggregate two different phenomena. Three further issues compound it:

1. **Tie detection is exact-equality only** (`src/classify.rs:648-649`). The LCA cap and
   `alternatives` field never fire for near-ties, which is the ordinary shape of species-holdout
   evidence (all surviving congeners score within ~1 % of one another). No margin concept exists.
2. **Descent ambiguity does not propagate to reported confidence** in the greedy path. A node that
   descended at 98/100 vs 97/100 (below-min_descend sibling) is reported identically to one that
   descended at 100/100. A species-holdout query typically descends via this kind of thin margin
   because the true species is absent.
3. **LCA-truncated reference entries ("NA") become pseudo-taxa** in the trained tree because the
   training code has no NA guard (`src/training.rs:140-156`, `src/fasta.rs:31-57`). The benchmark's
   new mixed-depth augmentation injects these on purpose; oxidtaxa treats `;NA;NA` accessions as
   a fake "NA" genus with a singleton species, which pulls real sibling k-mers through it.

The remaining improvements cover the already-known bugs from the 2026-04-15 parameter audit
(`rank_thresholds` non-contiguous lineage, `correlation_aware_features` degenerating on binary
splits, `use_idf_in_training` creating a hybrid scoring path that matches neither descent nor leaf
phase) plus three new proposals: sibling-aware leaf scoring, a novelty score independent of
`confidence`, and per-rank IDF.

Detailed findings and code-level proposals follow.

---

## How the Benchmark Stresses oxidtaxa — Tier by Tier

### Normal tier
All target accessions remain in the reference. oxidtaxa should classify to species with high
confidence. This tier exercises the happy path and is a regression baseline.

### Haplotype tier
(`species_selector.py:1165-1178` — one accession becomes the query, the rest of that species'
accessions stay in the reference.) The classifier should still descend to species because
conspecific sequences are present. This tier specifically exercises **within-species k-mer
tolerance** — the query's exact k-mer set is no longer in any training sequence, but very similar
ones are.

Stress point for oxidtaxa: if the kept-k-mer profile at the species' decision node
(`src/types.rs:6-11`) was shaped by the removed accession in training, the query may not vote
decisively for the correct child. Training-time `leave_one_out`
(`src/types.rs:243`, fixed to skip singletons in commit `3d6cb91`) is the relevant calibration knob.

### Species-holdout tier
(`species_selector.py:1118-1151` — entire species removed; at least one congeneric species retained.)
Evaluation at **genus rank** (`ground_truth.py:747-749`). Scoring is strict — any non-null
species-rank prediction turns the row into FP (`metrics.py:1594-1596`).

Stress points for oxidtaxa:
- At the genus node, descent sees multiple remaining sibling species. Votes concentrate on the
  sibling whose k-mer profile happens to be closest to the query. Unless all vote_counts are within
  the fallback band, **descent continues into that sibling's subtree** and the leaf-phase produces
  a species-level prediction for a species that is not the true one.
- The `min_descend = 0.98` threshold (`src/types.rs:303`) is easily met when one sibling has a
  handful more votes than the others. At the genus node, with three surviving siblings whose vote
  splits are e.g. 70/20/10, the greedy filter at `src/classify.rs:225-227` rejects all three (<=98)
  — good, we fall into the 50% branch at `src/classify.rs:229-238`. But at 99/1/0 the classifier
  commits to the single winner even though that winner is only a ~1 % margin over "no signal".
- Exact-tie LCA cap (`src/classify.rs:648-649`) rarely fires because `tot_hits` values are continuous
  floats; bit-identical ties are only produced when the reference contains duplicate sequences,
  which is uncommon on LCA-built reference databases (the production format — see
  `2026-04-17-species-genus-holdout-tier-improvements.md` Addendum 1).

### Genus-holdout tier
(`species_selector.py:1069-1116`, write path at `benchmark_shared.py:1275-1281` removes the entire
genus from the reference, **including all its species in the full DB not just the community**.)
Evaluation at **family rank** (`ground_truth.py:750-752`). Both species and genus must be null
(`metrics.py:1650-1652`).

Stress point for oxidtaxa: same as species-holdout but one rank deeper. At the family node, the
remaining genera's k-mer profiles are now being asked to arbitrate a query whose true genus is
missing entirely. The 63.4 % empty-path rate on this tier reported in
`2026-04-17-abstention-path-output-handling.md` shows the classifier correctly recognizes this as
low-confidence, but the output format loses the signal.

### Mixed-depth tier
(`species_selector.py:197-266` + `:1194-1208`.) Pseudo-labels `[genus]X` / `[family]Y` are LCA-truncated
reference entries. The reference contains `;Genus;NA;` or `;NA;NA;` suffixes. Evaluation expects
the classifier to emit a truncated path (no species / no genus+species). Current oxidtaxa builds
"NA" as a pseudo-taxon and trains against it as if it were a real class — so queries routed through
the "NA" decision node receive its profile matches as legitimate k-mer votes.

---

## Detailed Findings

### F1. Abstention output collapses three distinct states

**Files**: `src/classify.rs:741-748`, `src/fasta.rs:78-105`, `src/types.rs:168-175`.

Three paths produce "unclassified":

1. **Path A — too few k-mers after filter** (`src/classify.rs:186-188`): returns `None` →
   `ClassificationResult::unclassified()` with hardcoded `confidence=[0.0, 0.0]`.
2. **Path B — no training match survives `full_length` filter** (`src/classify.rs:483`): same
   `None` → same `::unclassified()`.
3. **Path C — below-threshold after real scoring** (`src/classify.rs:741-748`): the full
   `predicteds` lineage and `confidences` vector were computed. If `above` is empty, falls back to
   `w = vec![0]` — emits `[Root, unclassified_Root]` with `confidence = [c0, c0]` where `c0` is the
   real Root confidence.

Downstream, `write_classification_tsv` at `src/fasta.rs:78-105`:
- Drops entries whose name starts with `unclassified_` (line 85).
- If `filtered_taxa` is empty (all entries were `unclassified_*` or only Root remained), writes
  `{read_id}\t\t0\t{alts}` (line 101).

So Path C — which carries a real `c0 > 0` confidence that the query's k-mers landed under Root at
all — is reported identically to Path A/B (`c0 == 0`, no signal). Every downstream consumer
(including `assignment_benchmarks`' `tools/oxidtaxa/parser.py:43-106`) sees `taxonomic_path == ""`
and score `0`. Tier-specific metrics cannot distinguish "confidently assigned root and abstained at
kingdom" from "query did not overlap reference at all" — both are scored as FN at the eval rank
(`metrics.py` truth-depth branch).

Impact is concentrated in the genus-holdout and species-holdout tiers because those are where Path C
fires most often: the classifier has genuine Root/Kingdom/Phylum signal but nothing that passes the
single `threshold = 60` (or per-rank threshold).

### F2. No margin-based tie detection anywhere; descent uses band thresholds, leaf uses exact equality

**Files**: `src/classify.rs:225-238` (descent), `:647-656` and `:696-718` (leaf-phase LCA cap).

oxidtaxa has two distinct points where "is this a tie?" could be asked. Neither asks
`(top_1 - top_2) < margin`. They test different things.

**(a) Descent-time: absolute vote thresholds against a fixed bar** (`src/classify.rs:225-231`):

```rust
let w: Vec<usize> = vote_counts.iter().enumerate()
    .filter(|(_, &c)| c >= (config.min_descend * b as f64) as usize)   // >= 98/100
    .map(|(i, _)| i).collect();
if w.len() != 1 {
    let w50: Vec<usize> = vote_counts.iter().enumerate()
        .filter(|(_, &c)| c >= ((b as f64) * 0.5) as usize)             // >= 50/100
        .map(|(i, _)| i).collect();
    ...
}
```

Each child is tested **against a fixed bar, not against its runner-up**. A child with 98 votes
passes the 0.98 filter whether the runner-up had 2 votes or 97. There is no subtraction and no
ratio between them. What this gets you is **band semantics**:

- `vote_counts = [99, 1]` → child 0 passes 0.98, child 1 fails 0.98, `w = [0]` → descend into child 0.
- `vote_counts = [99, 98]` → both pass 0.98, `w = [0, 1]`, `w.len() != 1` → fall into the 50 %
  fallback branch. Here the runner-up's strength incidentally blocks a confident descent, which
  *looks* margin-aware but is actually a consequence of two absolute thresholds firing
  simultaneously.
- `vote_counts = [99, 97]` → only child 0 passes 0.98, `w = [0]` → descend into child 0 with no
  signal that child 1 was one vote away from forcing the fallback.

So the descent filter catches the 99/98 case incidentally but not the 99/97 case — the distinction
between those two is exactly the "is the gap small?" question a margin check would resolve, and
the code does not ask it.

**(b) Leaf-phase LCA cap and `alternatives`: exact floating-point equality on `tot_hits`**
(`src/classify.rs:647-656`):

```rust
let max_tot = tot_hits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let winners: Vec<usize> = tot_hits.iter().enumerate()
    .filter(|(_, &v)| v == max_tot).map(|(i, _)| i).collect();
```

The `lca_cap` block at `src/classify.rs:696-718` is gated `if winners.len() > 1`. `v == max_tot`
requires bit-equal f64 values between aggregated bootstrap sums. That happens when:

- Two training sequences have bit-identical k-mer sets (rare; sequence deduplication upstream
  usually removes these). The `_species` flavor of rcrux-built reference does have such
  duplicates, but the `_lca` flavor used in production has already collapsed them (see
  2026-04-17 Addendum 1).
- Bootstrap replicates happen to award identical credit to multiple groups — possible in principle
  but vanishingly unlikely for more than a handful of test sequences.

Two `tot_hits` of 45.2 and 45.1999999 are NOT tied here; the first strictly wins, `winners.len()`
is 1, the LCA cap does not fire, and `alternatives` stays empty.

**(c) Per-replicate max (both descent and leaf)** (`src/classify.rs:220-223`, `:634-643`):

```rust
if hits_flat[ti * b + rep] == max_val { n_tied += 1; }
```

Same exact-equality story at replicate granularity. A replicate where two children scored 0.72 and
0.719 awards the vote to the 0.72 child alone.

**Why this matters for species-holdout**: the expected shape of evidence is **near-ties** — e.g.,
three surviving siblings of the removed species with `tot_hits` values like `[45.2, 44.8, 44.3]`.
None of these trigger `winners.len() > 1` at the leaf phase, so no LCA cap applies. And the
descent filter two ranks earlier may have already committed to a subtree via a 99/97 split that
the band-threshold check treated as a clean win. Neither filter can look at "the top candidate's
lead is only 1 %" and respond.

The recent per-replicate tie-splitting change (commit `0a684f5` / `cbc1a35`, documented in
`plans/2026-04-08-tied-species-reporting.md`) ensures bit-identical training pairs **do** surface
at the leaf-phase exact-equality filter, but that's a fix to make genuinely identical sequences
aggregate to identical floats — still exact equality, just reliably so. It's not a margin check.

A true margin check (what I1/I5 would add) replaces `v == max_tot` with
`v >= max_tot * (1.0 - tie_margin)` at the leaf phase, and adds a symmetric
`top_vote - second_vote >= margin_b` filter at descent time. Today neither exists.

### F3. Descent-time ambiguity does not enter `confidence`

**File**: `src/classify.rs:190-248` (greedy), `:371-378` (beam).

The greedy loop at `src/classify.rs:225-243` makes a Boolean descent decision: one child passing
`vote_counts[j] >= min_descend * b` → descend, else break and leaf-score the fallback `w_indices`.
The scalar vote counts that drove the decision are discarded. Downstream `base_confidence =
tot_hits[selected] / b * 100` (`src/classify.rs:673`) is the leaf-phase aggregate only and is
**constant across every rank in `predicteds`** (line 674). Non-selected-winners' `tot_hits` are
folded into ancestor confidences (`src/classify.rs:675-686`) — but again, only for exact-tie
non-selected winners (F2).

A query that descended 99/1/0 at the genus node and then scored 40 at leaf leaves the classifier
claiming `confidence = [40, 40, 40, 40, 40, 40, 40]` (identical across Kingdom…Species) — no
signal that the genus node was a near-singleton win. A query that descended 60/39/1 would have
fallen into the 50% fallback branch and gotten a very different classification, but the per-rank
reported confidence carries no memory of this.

The beam path at `src/classify.rs:371, 378` multiplicatively accumulates `candidate.score *=
top_vote_frac`, but that score is used only for beam pruning (`src/classify.rs:422-423`) and to
choose between competing leaf results (`src/classify.rs:430-442`). It does not flow into the
reported `confidences` vector either.

### F4. `rank_thresholds` non-contiguous lineage bug

**File**: `src/classify.rs:720-733`, `src/types.rs:289-291`.

The `above` filter collects indices whose confidence passes their per-rank threshold, but does not
require the set to be a contiguous prefix of `predicteds`. With `rank_thresholds = [90, 80, 70]`
and `confidences = [100, 75, 72]`, `above = [0, 2]` — the classifier emits `[Root,
<Class>]` skipping Kingdom. The truncation-fallback branch at `src/classify.rs:741-748` takes
`last_w = *w.last().unwrap()` and appends `unclassified_<last_w>` which labels this as
"unclassified at the Class rank" — a taxonomically incoherent statement.

This directly contaminates species-holdout abstention because the natural configuration of strict
species + lenient family thresholds (the kind of per-rank schedule users would choose to exploit
holdout-tier evaluation rules) activates this bug most often.

### F5. `correlation_aware_features` degenerates on binary splits

**File**: `src/training.rs:857-1025`, per-commit history `a2cd22a`.

The greedy forward-selection loop at `src/training.rs:943-983` scores each candidate as
`entropy_i * (1 - max_corr_i)`. Pearson correlation between two 2-element vectors
(n_children = 2) is always ±1, so after the first feature is selected, every remaining candidate
has `max_corr = 1` and `gain = 0`. The algorithm then falls back to selection order (entropy rank),
which is equivalent to the round-robin default.

Binary splits are **by far the most common shape** at fine ranks (genus with 2 species, family with
2 genera). So the knob that is intended to produce more-discriminating features at exactly those
nodes silently reverts to the baseline.

### F6. LCA-truncated reference entries become pseudo-taxa

**Files**: `src/training.rs:140-156`, `src/fasta.rs:31-57`.

Taxonomy parsing is string-based. `read_taxonomy` at `src/fasta.rs:31-57` joins the TSV taxonomy
value to the accession with no sentinel checks. `learn_taxa` at `src/training.rs:140-156` strips
`Root;`, normalizes separators, and ensures trailing `;`. Any distinct string at any rank —
including the literal `"NA"` — creates a distinct taxon.

For a reference row `gi|...|  Eukaryota;Craniata;Mammalia;Catarrhini;Homininae;NA;NA`:
- The tree gains a pseudo-taxon `"NA"` at the genus rank under Homininae.
- A second `"NA"` pseudo-taxon exists at the species rank under it.
- The `sequences[node]` list at the genus-"NA" node aggregates every accession with `family !=
  "NA" AND genus == "NA"`, which in production vert12S LCA DB is ~5000 entries spanning unrelated
  clades (cf. addendum 1 of the 2026-04-17 research).
- IDF weights computed at `src/training.rs:309-351` treat this pseudo-class as one of `n_classes`,
  diluting the IDF denominator and biasing weights toward pseudo-class discrimination.

The benchmark's new mixed-depth augmentation feature deliberately injects `[genus]X` / `[family]Y`
pseudo-labels into the community and writes their taxonomy with empty species / genus fields. When
the reference is concatenated, the TSV taxonomy is `...Family;Genus;;` or `...Family;;;` — blank
rather than literal "NA". The current `read_taxonomy` `split(';').collect()` behavior maps blank to
the empty string, which then becomes a pseudo-taxon named `""`. This is an even more pathological
case.

### F7. IDF is computed globally, not per-rank

**File**: `src/training.rs:309-351`.

```rust
idf_weights[k] = ln(n_classes / (1 + idf_counts[k]))
```

`n_classes` is the number of unique leaf taxonomy strings; `idf_counts[k]` is the class-weighted
sum of sequences containing k-mer k. A k-mer that is diagnostic at the phylum rank (highly
conserved within phylum, highly variable across phyla) receives the same weight as a k-mer
diagnostic at the species rank (idiosyncratic to one species).

For species-holdout evaluation, the correct classifier behavior is to **reward genus-level
discriminators** and **downweight species-level discriminators** (since the species-level signal
is about to be tested on a species that does not exist in the reference). A per-rank IDF would
enable this; a global IDF does not.

### F8. `use_idf_in_training` creates a third scoring variant

**File**: `src/training.rs:510-512` (training-time `vector_sum`), `src/classify.rs:550-554`
(classify-time `parallel_match*`), prior research doc `2026-04-15-new-parameter-audit.md`.

When `use_idf_in_training = true`, the fraction-learning descent scores each child with
`profile[j] * idf_weights[keep[i]]` instead of `profile[j]`. The intent is "make training scoring
match classification scoring". But classification descent (tree-descent phase at
`src/classify.rs:193-248`) uses `profile[j]` only — not IDF — via `vector_sum`. IDF only enters at
the leaf phase (`src/classify.rs:546-548`). So `use_idf_in_training` produces a third hybrid that
matches **neither** classification path, which is why the 2026-04-15 audit flagged it as "design
flaw". Today this knob is off by default; enabling it would degrade holdout-tier accuracy rather
than improve it.

### F9. Training fraction decrement diverges from R

**File**: `src/training.rs:570-590`, prior research doc `2026-04-15-new-parameter-audit.md`.

oxidtaxa applies fraction decrements once per iteration in a parallel batch, after aggregating all
failures; R applies them sequentially per sequence. oxidtaxa also caps each decrement at half the
remaining headroom (`training.rs:575`). For deep or imbalanced trees these diverge — the trained
`fraction[node]` vector controls how many k-mers are sampled at descent time
(`src/classify.rs:206-208`), which directly affects holdout-tier behavior because sampling density
drives the variance of vote counts.

### F10. Haplotype-tier sensitivity to training memorization

**File**: `src/training.rs:1088-1106` (leaf profile aggregation).

A leaf's emitted profile is the sum of normalized per-sequence k-mer counts across all sequences at
that leaf. For a species with N accessions, removing one accession (haplotype tier) shifts the
leaf profile by ~1/N of its mass. For singletons (N = 1, held out and then only one conspecific
accession in the reference — the haplotype-holdout scenario effectively), the profile was shaped
by the removed sequence; the surviving sibling's k-mers produce a degraded match.

This is what `leave_one_out` was supposed to compensate for. The current implementation
(`src/training.rs:477-508`) skips singletons entirely and only adjusts groups of size 2–5, scaling
by `(n-1)/n`. The logic is biased: it helps moderate-sized groups but leaves singletons and large
groups untouched. For the haplotype-holdout tier this is roughly the right calibration for the
middle case; for the species-holdout tier (where the sibling is a different species) it does
nothing.

---

## Proposed Improvements

Categorized by expected impact on holdout-tier robustness. The estimates assume the current
benchmark framing where Normal/Haplotype/Species-holdout/Genus-holdout/Mixed-depth each represent
~25 % of the ASV population and scoring uses truth-depth eval ranks per tier.

### Tier 1 — Dominant: preserve real signal through abstention

#### I1. Write the real leaf-phase confidence when the path is empty

**Targets**: F1.

**The problem precisely**: `confidences[i]` is monotonically non-decreasing as `i` walks from
Species toward Root (only non-selected-winner ancestor contributions are added, never subtracted —
see `src/classify.rs:675-686`). So if Root's confidence fails threshold, every deeper rank fails
too. `above.is_empty()` genuinely means "no rank has enough confidence to report." Emitting `Root`
as a taxon label in that case would be tautological (every organism is Root) and not useful.

But the **scalar confidence value itself** — specifically `base_confidence = tot_hits[selected] /
b * 100` at `src/classify.rs:673` — is lost by the TSV writer even though it carries real
information. Two classifier states currently produce identical TSV output:

| State | In-memory `confidence` | Current TSV output |
|-------|------------------------|--------------------|
| Path A/B (no signal) | `[0.0, 0.0]` hardcoded (`src/types.rs:169-175`) | `{read_id}\t\t0\t` |
| Path C, below threshold (real leaf scoring, all ranks fail cutoff) | `[c0, c0, ..., c0]` where `c0 > 0` | `{read_id}\t\t0\t` |

A Path C row with `c0 = 45` is unrecoverable from a Path A row with `c0 = 0` once it hits disk,
even though in-memory they are plainly distinguishable.

**Change**: In `src/fasta.rs:78-105`, when `filtered_taxa` is empty, write the real
`result.confidence[0]` (which is `c0` for every Path C case and `0.0` for Path A/B) instead of the
hardcoded `0`. Keep `taxonomic_path` empty — there is no defensible taxon label to emit. The
confidence column itself becomes the signal:

```rust
// Today (src/fasta.rs:101, :104):
output.push_str(&format!("{}\t\t0\t{}\n", read_id, alternatives_field));

// Revised:
let c0 = result.confidence.first().copied().unwrap_or(0.0);
output.push_str(&format!("{}\t\t{}\t{}\n", read_id, c0, alternatives_field));
```

**Impact**: Path A/B rows write `confidence = 0`; Path C rows write `confidence = c0 > 0`.
Downstream calibration (benchmark notebooks, Optuna sweeps) can now:
- Distinguish "no evidence at all" from "weak evidence below threshold."
- Compute precision/recall curves across thresholds without re-running the classifier — just
  re-apply the threshold to the TSV confidence column.
- Spot when a holdout tier's empty-row rate is dominated by Path C (a calibration problem, fixable
  by tuning threshold) vs Path A/B (a reference-coverage problem, fixable only by expanding the
  reference).

The benchmark parser at `tools/oxidtaxa/parser.py:43-106` reads `confidence / 100.0` and already
handles any value in `[0, 100]`, so no downstream changes are required.

#### I2. Optional: explicit `reject_reason` column to separate Path A from Path B

**Targets**: F1 (refinement).

I1 collapses Path A and Path B into the same "confidence = 0" output. If the distinction between
"too few k-mers after filter" (`src/classify.rs:186-188`) and "no training match survives
`full_length` filter" (`src/classify.rs:483`) matters for a specific analysis, add an optional
column:

```
read_id \t taxonomic_path \t confidence \t alternatives \t reject_reason
```

Values: `""` (not rejected), `"too_few_kmers"`, `"no_training_match"`, `"below_threshold"`.

**Impact**: Marginal. Most analyses don't care which `None`-returning branch fired; they just need
"signal vs no signal," which I1 alone provides. Ship only if Path A vs Path B turns out to be
material after I1 lands.

#### I3. Fix `rank_thresholds` non-contiguous bug

**Targets**: F4.

**Change**: In `src/classify.rs:720-733`, after collecting `above`, enforce contiguity — if index
`i` fails threshold, drop indices `> i` regardless of their own thresholds. Proposed replacement:

```rust
let mut above: Vec<usize> = Vec::new();
for (i, &c) in confidences.iter().enumerate() {
    if let Some(cap) = lca_cap {
        if i > cap { break; }
    }
    let thresh = rank_threshold_at(i, config);
    if c >= thresh {
        above.push(i);
    } else {
        break;
    }
}
```

This guarantees `above` is always `[0, 1, 2, ..., k]` for some `k`, eliminating the `[0, 2]`
pathology. The truncation fallback at `src/classify.rs:741-748` then honestly reports "classified
through rank k, unclassified at rank k+1".

**Impact**: Unlocks per-rank threshold schedules as a useful knob for species-/genus-holdout tiers.
Today, users avoid `rank_thresholds` because of this bug; with the fix, a schedule like `[0, 20,
30, 40, 50, 60, 80]` would let the classifier emit family-rank assignments for genus-holdout
queries confidently while abstaining at genus and species — which is **exactly** the evaluation
rule the benchmark uses.

#### I4. Handle LCA-truncated reference entries at training time

**Targets**: F6.

**Change**: In `src/fasta.rs:31-57` and `src/training.rs:140-156`, treat `NA` (case-insensitive) and
empty-string taxonomy fields as sentinels. 

**Truncate the taxonomy at the sentinel**: `"Family;NA;NA"` becomes `"Family;"` — a leaf at
the family rank. Downstream tree construction naturally places this accession at the family
decision node. No pseudo-taxa are created. Classification of queries that match this accession
then cannot descend past family, giving honest rank-truncated output.

**Impact**: The mixed-depth tier becomes a fair evaluation rather than a pseudo-classification
benchmark. Also cleans up production real-data runs on LCA-built references (`_lca.fasta` /
`_lca_taxonomy.txt`, currently the production format per 2026-04-17 Addendum 1).

### Tier 2 — High: better representation of ambiguity

#### I5. Margin-based LCA cap

**Targets**: F2.

**Change**: At `src/classify.rs:647-656`, in addition to the exact-equality filter, compute a
margin-based winners set:

```rust
let margin = config.tie_margin.unwrap_or(0.0);  // new config field
let max_tot = tot_hits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let threshold = max_tot * (1.0 - margin);
let winners: Vec<usize> = tot_hits.iter().enumerate()
    .filter(|(_, &v)| v >= threshold).map(|(i, _)| i).collect();
```

Default `tie_margin = 0.0` preserves today's behavior. Setting to e.g. 0.05 would treat any group
within 5 % of the maximum as a "tied" winner for the purposes of LCA cap and `alternatives`.

**Impact**: Species-holdout queries with `tot_hits = [45.2, 44.8, 44.3]` now emit genus-rank LCA
cap + `alternatives = [sibling_A, sibling_B, sibling_C]`, a far more useful output than the
current "45.2 wins, commit to sibling_A's species". The benchmark's strict species-holdout metric
(`metrics.py:1569` — require species == null for TP) rewards this directly.

Care: `alternatives` currently serializes alphabetically sorted short-labels of tied leaves
(`src/classify.rs:710-714`). With `margin > 0`, the alternatives can include genuinely weaker
candidates — callers should be aware. Adding a second column `alternative_scores` with per-alt
scores would preserve information.

#### I6. Propagate descent ambiguity into confidence

**Targets**: F3.

**Change**: In the greedy descent at `src/classify.rs:218-243`, record per-rank descent margins
(e.g., `top_vote_frac - second_vote_frac`) and multiply `confidences[rank]` by the accumulated
margin product before the above-threshold check at `src/classify.rs:720-733`. A minimum margin
floor prevents zeroing.

Implementation sketch:

```rust
let mut descent_margins: Vec<f64> = Vec::new();  // one entry per descended rank
// ... inside the descent loop:
let top = vote_counts.iter().max().unwrap() as f64;
let runner_up = {
    let mut v = vote_counts.clone();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.get(1).copied().unwrap_or(0) as f64
};
let margin = ((top - runner_up) / b as f64).max(0.1);  // floor at 10%
descent_margins.push(margin);
```

Then after leaf scoring, multiply `confidences[i]` for each rank `i` by the product of descent
margins from Root to `i`.

**Impact**: A 99/1 descent contributes ×0.98 to downstream confidences; a 60/40 descent contributes
×0.2. The classifier that accidentally descended into a sibling during species-holdout now has its
species-level confidence discounted by the near-tie it had at the genus node — that confidence
often falls below `threshold = 60`, triggering abstention at the correct rank.

This is a behaviorally meaningful change and should be gated behind a config flag
`confidence_uses_descent_margin: bool` defaulting to `false` so existing calibration sweeps are not
invalidated.

#### I7. Sibling-aware leaf scoring for single-winner descent

**Targets**: F2, F3.

**Change**: At the point where greedy descent commits to a single winner (`src/classify.rs:241-243`),
instead of setting `w_indices = vec![winner]`, set `w_indices` to the winner **plus any sibling
whose vote share was within a margin** (e.g., >= 50 % of winner's, already part-way computed at
`src/classify.rs:229-231`). Then let `leaf_phase_score` compute `tot_hits` against all of them.

This restores the sibling-comparison step that the greedy path elides. When true species is absent,
genuine siblings score equally well at leaf phase, and the exact-equality LCA cap (or the
margin-based cap from I5) then fires.

**Impact**: Covers the "descent was confident but leaf evidence is ambiguous" case. Combined with
I5, this gives the classifier two independent paths to detect absent-taxon queries — one at descent
time, one at leaf time.

#### I8. Add a novelty score independent of `confidence`

**Targets**: F1 (provides calibration signal), F3.

**Change**: Compute `similarity` (already exists at `src/classify.rs:657-660`) as a field on
`ClassificationResult`. Expose it via TSV and Python.

```rust
#[cfg_attr(feature = "python", pyo3::pyclass(get_all))]
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
    pub alternatives: Vec<String>,
    pub similarity: f64,   // new
}
```

TSV gains a column `similarity`. Benchmark consumers can then score "novel query" (low similarity,
possibly low confidence) distinctly from "ambiguous query" (high similarity, low confidence —
meaning query looks like reference but reference has ambiguous labels).

**Impact**: Low code risk (`similarity` is already computed and currently unused). Opens the door
to the post-hoc "reject below similarity S" filter that a user could apply to separate holdout
from in-reference queries without needing to re-train.

### Tier 3 — Medium: correctness fixes that help holdout tiers specifically

#### I9. Replace Pearson redundancy with Bhattacharyya coefficient / Hellinger distance

**Targets**: F5.

**Framing**: The issue isn't a binary-split edge case to patch around — it's that Pearson is the
wrong *kind* of metric for this problem, and n=2 is where the wrongness becomes visible as a
mathematical degeneracy (for any two 2D points, Pearson r = ±1, so `1 − |r| = 0` for every
non-first candidate and the redundancy term annihilates the gain function). For n > 2 Pearson
still misbehaves more subtly: profiles `[0.01, 0.01, 0.01]` (rare everywhere) and `[0.99, 0.99,
0.99]` (common everywhere) have Pearson = 1 despite carrying very different amounts of information.

**Literature position**:
- The mRMR foundation (Peng/Long/Ding 2005, IEEE TPAMI) uses **mutual information** for discrete
  features, Pearson correlation only as a computational proxy for MI and only when per-feature
  sample size is large (their microarray applications had hundreds of samples per feature).
  Brown et al. 2012 (JMLR) surveys the information-theoretic feature-selection family
  (mRMR/CMIM/JMI/DISR) — all use MI/conditional MI, none use JSD or Pearson-as-primary.
- Jensen-Shannon divergence is standard for **genome-vs-genome** alignment-free phylogenetics
  (Sims et al. 2009 PNAS, "Feature Frequency Profiles") but **not** for feature-vs-feature
  redundancy inside a classifier. No major k-mer classifier (IDTAXA, Kraken, CLARK, SINTAX, RDP)
  uses JSD for internal k-mer deduplication. IDTAXA itself doesn't do any redundancy filtering —
  it just takes top-N by cross-entropy. oxidtaxa's `correlation_aware_features` mode is already
  an extension beyond IDTAXA.
- **Hellinger distance** as a redundancy metric for bioinformatics feature selection has direct
  precedent: Fu, Wu, Liu 2020 (BMC Bioinformatics 21:121) — "Hellinger distance-based stable
  sparse feature selection for high-dimensional class-imbalanced data." Proper metric on
  probability distributions, bounded in [0, 1], triangle inequality, robust to class imbalance.

**Profile vector semantics** (confirmed by code analysis of `src/training.rs:886-919`): each
candidate's profile vector is `[freq(kmer | child_0), …, freq(kmer | child_{n-1})]` with entries
in `[0, 1]`, non-negative, zero for absent children. Values do NOT sum to 1 across children (each
entry is an independent conditional frequency), so any divergence-on-distributions metric
requires an L1-normalization preprocessing step.

**Change**: At `src/training.rs:857-1025`, replace Pearson-based redundancy with
**Bhattacharyya coefficient on L1-normalized sqrt-transformed profiles** (equivalent to
`1 − H²` where H is Hellinger distance).

Recipe, one-time per candidate (precomputation):
```rust
fn prepare_bhattacharyya_profile(p: &[f64]) -> Vec<f64> {
    let sum: f64 = p.iter().sum();
    if sum == 0.0 { return vec![0.0; p.len()]; }  // filter at caller
    p.iter().map(|&v| (v / sum).sqrt()).collect()
}
```

Pairwise (replaces `pearson_with_stats` at `src/training.rs:714-722`):
```rust
fn bhattacharyya(s_a: &[f64], s_b: &[f64]) -> f64 {
    s_a.iter().zip(s_b.iter()).map(|(a, b)| a * b).sum()
    // = 1 when profiles are identical (after L1 norm), 0 when disjoint.
}
```

Gain function unchanged in shape:
```rust
gain_i = entropy_i * (1.0 - max_bc_i);
```
where `max_bc_i` is the max Bhattacharyya coefficient between candidate `i` and any
already-selected feature. The `max_corr` cache structure (commit `a2cd22a`) carries over
verbatim — rename to `max_bc`.

**Edge cases**:
- All-zero profile: drop from candidate pool before entropy ranking (these k-mers are
  uninformative by construction).
- Numerical stability: no log evaluations means no log(0) issues.

**Cost relative to current Pearson** (per pair, after precomputation):
- Current Pearson: `O(n)` mul + 2 sub + 1 div
- Bhattacharyya on sqrt-profiles: `O(n)` mul + `O(n)` add (dot product)
- JSD (for reference): `O(n)` mul + `2n log` + additions → ~5–15× slower per pair

Bhattacharyya is **as fast or faster than Pearson**. Precomputation of sqrt-profiles is
`O(n_candidates * n_children)` — the same scale as the existing candidate-pool construction.

**Impact**: (1) Resolves the n=2 degeneracy cleanly — `[0.9, 0.1]` vs `[0.1, 0.9]` now gives
BC = 2·sqrt(0.09) = 0.6 (meaningful redundancy), not r = −1 → redundancy = 1. (2) Replaces a
metric that's subtly wrong across the whole tree with one that's mathematically justified for
probability-distribution-like profile vectors. (3) Enables a principled upgrade path to JSD or
mutual-information-based redundancy if future work calls for it (Bhattacharyya is within the
same metric family as JSD; swapping in the full JSD is a local change).

**Open item**: The research agent noted that Zielezinski et al. 2019 (Genome Biology) found
simpler metrics often outperform sophisticated ones in genomic benchmarks. Empirical sweep on
the benchmark corpus (Pearson vs cosine-on-raw-profiles vs Bhattacharyya) would confirm the
theoretical argument with numbers; this is a one-PR change to the feature-selection core with
direct F1 impact measurable on the same config sweep infrastructure that sweeps `threshold`/`k`.

#### I10. Per-rank IDF weights

**Targets**: F7.

**Change**: At `src/training.rs:309-351`, additionally compute `idf_weights_by_rank[rank][k]`
where `n_classes` at rank `r` is the number of distinct taxa at that rank. Store as
`Vec<Vec<f64>>` on `TrainingSet`. At classify time, use the rank corresponding to the current
depth in the descent.

Simpler variant: compute IDF once, but at depth `d` during the leaf phase use
`weights[k] = global_idf[k] * rank_weight[d]` where `rank_weight[]` is a user-configurable
schedule defaulting to uniform.

**Impact**: Allows training to recognize genus-diagnostic k-mers as such and upweight them when
the classifier's leaf phase is effectively asking a genus-rank question (because the descent got
stuck at the genus node due to species holdout).

Higher risk: changes the model bincode format. Would need a format version bump and load-compat
path.

### Tier 4 — Low-risk hygiene

#### I11. Disable or replace `use_idf_in_training`

**Targets**: F8.

**Change**: Either (a) deprecate and remove the knob (it makes no scenario better), or (b) redefine
it to actually match one of the two scoring paths (descent-time profile-only, or leaf-time
IDF-only) instead of the current hybrid. The prior audit flagged (a) as the simpler path.

**Impact**: Eliminates a misleading knob from Optuna sweeps. No behavior change with the default
`false`.

#### I12. Fix `seed_pattern` (and `k`) weight overflow

**Targets**: separate audit finding.

**Root cause**: `pwv: Vec<i32>` at `src/kmer.rs:474-490` stores `4^i` up to `4^(weight-1)`;
max k-mer index then ranges up to `4^weight - 1`. `weight = 15` gives `4^15 - 1 = 2^30 - 1` which
fits in i32; `weight = 16` gives `4^16 - 1 = 2^32 - 1` which overflows (silent wrap in release,
panic in debug). The same overflow applies to the contiguous-k path at `src/kmer.rs:505-517` when
`k > 15`. No bounds check exists in either `parse_seed_pattern` (`src/kmer.rs:19-42`) or on the
`k`/`word_size` entry points.

**Change**: Validate at parse time. In `parse_seed_pattern`:

```rust
const MAX_KMER_WEIGHT: usize = 15;

if match_positions.len() > MAX_KMER_WEIGHT {
    return Err(format!(
        "Seed pattern weight {} exceeds maximum {} (4^{} overflows i32). \
         Use a shorter pattern or fewer '1' positions.",
        match_positions.len(), MAX_KMER_WEIGHT, MAX_KMER_WEIGHT
    ));
}
```

Symmetric guard on `k` at the Python entry points in `src/lib.rs` (`train`/`classify`), rejecting
`k > 15` with the same message. The ceiling of **15** is correct because `4^15 = 2^30` leaves a
full bit of headroom for signed-int arithmetic (intermediate sums like `3 * (4^k - 1) / 3` must
stay below `2^31`).

**Impact**: Prevents silent corruption when users try longer spaced seeds or k values. If
weight ≥ 16 is ever needed (~1B distinct k-mers at weight 15 is already more than any realistic
reference), the remedy is widening the k-mer path from i32 to u32/i64 — a significant refactor
across `src/matching.rs`, `src/types.rs::DecisionNode`, and the bincode format. Not worth doing
until there's a concrete need.

#### I13. Document the R-divergent training decrement as intentional

**Targets**: F9.

**Decision**: The parallel batch snapshot + capped decrement + aggregated per-node update at
`src/training.rs:570-590` is a deliberate performance trade-off (accepts non-bit-identical
`fraction` vectors vs R in exchange for training parallelism). Not worth reverting to R's
sequential behavior.

**Change**: Document the divergence rather than fix it.

1. Add a comment block at `src/training.rs:570-590` explicitly stating what differs from R
   (parallel snapshot, capped decrement, aggregated update) and why (training-time speedup).
2. Add a regression test in `tests/` that captures current `fraction` vector values on a small
   fixture dataset so the behavior doesn't silently drift.
3. Note in `README.md` that `learn_taxa` is classifier-equivalent to R's `LearnTaxa` but
   produces different internal `fraction` vectors due to parallel training. Downstream
   classification output remains behaviorally comparable.

**Impact**: Prevents future maintainers (or audits) from re-raising the "diverges from R" flag
without understanding the trade-off. No runtime effect.

---

## Impact and Prioritization Table

| ID | Improvement | Files | Category | Holdout-tier impact |
|----|-------------|-------|----------|---------------------|
| I1 | Write real `c0` to TSV confidence column when path is empty | `src/fasta.rs:78-105` | Correctness | **Dominant** — distinguishes Path A/B from Path C without fake taxon labels |
| I2 | Optional `reject_reason` column (A vs B distinction) | `src/fasta.rs`, `src/types.rs` | Clarity | Low — only if Path A vs B matters after I1 |
| I3 | Fix `rank_thresholds` contiguity | `src/classify.rs:720-733` | Correctness | High — unlocks per-rank thresholds for holdout tiers |
| I4 | NA / LCA-truncated handling | `src/training.rs:140-156`, `src/fasta.rs:31-57` | Correctness | High — mixed-depth tier becomes fair |
| I5 | Margin-based LCA cap | `src/classify.rs:647-656` | Algorithmic | High — species-holdout near-ties produce LCA+alternatives |
| I6 | Descent-margin in confidence | `src/classify.rs:218-243, :720-733` | Algorithmic | High — confidence reflects descent ambiguity |
| I7 | Sibling-aware leaf scoring | `src/classify.rs:241-250` | Algorithmic | Medium-High — catches leaf-phase ambiguity single-winner descent misses |
| I8 | Expose `similarity` on result | `src/classify.rs:657-660`, `src/types.rs:154-166` | Clarity | Medium — calibration handle for downstream |
| I9 | Replace Pearson redundancy with Bhattacharyya coefficient (Hellinger-based) | `src/training.rs:714-722, :857-1025` | Correctness | Medium-High — mathematically justified, faster than Pearson, resolves n=2 degeneracy |
| I10 | Per-rank IDF | `src/training.rs:309-351` | Algorithmic | Medium — model format change |
| I11 | Remove `use_idf_in_training` | `src/types.rs:238`, `src/training.rs:510-512` | Hygiene | Low — hides misleading knob |
| I12 | Validate seed_pattern weight and k ≤ 15 at parse time | `src/kmer.rs:19-42`, `src/lib.rs` | Correctness | Low — unrelated but open |
| I13 | Document R-divergent training decrement as intentional | `src/training.rs:570-590`, README | Clarity | Low — comment + regression test |

---

## Recommended Rollout

1. **I1 + I3 + I4** — three file-local fixes that together transform the output semantics for the
   three holdout tiers the user cares about. No training format change, no API change (I1/I3) or
   only sentinel-handling addition (I4). Ship as one PR with regression tests covering the
   empty-path row conversion and the `;NA;NA` pseudo-taxon elimination.

2. **I8** — expose `similarity` on `ClassificationResult` + TSV. Gated on the caller; benchmark
   notebook can opt-in. (I2 is now optional — only ship if Path A vs Path B turns out to matter
   after I1 lands.)

3. **I5 + I6 + I7** — core algorithmic changes. Ship behind config flags
   (`tie_margin`, `confidence_uses_descent_margin`, `sibling_aware_leaf`) defaulting to current
   behavior. Run benchmark sweep on the existing config corpus to confirm they help species- and
   genus-holdout tiers without regressing Normal / Haplotype.

4. **I9 + I10** — training-side. I9 is a small fix; I10 needs format version bump and compat
   loader. Sequence after core algorithmic changes have settled.

5. **I11 + I12 + I13** — polish pass.

---

## Code References

### Classification algorithm
- `src/classify.rs:33-83` — `id_taxa` entry
- `src/classify.rs:168-251` — `classify_one_pass` greedy descent
- `src/classify.rs:190-248` — greedy descent loop (F3)
- `src/classify.rs:225-243` — descent threshold and winner selection
- `src/classify.rs:229-238` — 50% fallback band
- `src/classify.rs:449-752` — `leaf_phase_score`
- `src/classify.rs:601-643` — bootstrap aggregation with tie-splitting
- `src/classify.rs:647-656` — winner selection and exact-tie filter (F2)
- `src/classify.rs:673-686` — confidence propagation
- `src/classify.rs:696-718` — LCA cap and `alternatives`
- `src/classify.rs:720-733` — per-rank threshold filter (F4)
- `src/classify.rs:741-748` — below-threshold collapse (F1)
- `src/classify.rs:657-660` — `similarity` scalar (I8)

### Output layer
- `src/fasta.rs:31-57` — `read_taxonomy` (F6, I4)
- `src/fasta.rs:60-110` — `write_classification_tsv` (F1, I1, I2)

### Training
- `src/training.rs:19-40` — `learn_taxa`
- `src/training.rs:140-156` — taxonomy normalization (F6, I4)
- `src/training.rs:309-351` — IDF computation (F7, I10)
- `src/training.rs:415-591` — fraction learning (F9, F10)
- `src/training.rs:477-508` — `leave_one_out` weight scaling (F10)
- `src/training.rs:510-512` — training-time scoring (F8)
- `src/training.rs:570-590` — fraction decrement and cap (F9)
- `src/training.rs:747-1108` — `create_tree`
- `src/training.rs:857-1025` — correlation-aware selection (F5, I9)
- `src/training.rs:1088-1106` — leaf profile aggregation

### Types and config
- `src/types.rs:6-11` — `DecisionNode`
- `src/types.rs:22-46` — `TrainingSet` (I10 impact)
- `src/types.rs:153-175` — `ClassificationResult` (I8)
- `src/types.rs:215-272` — `TrainConfig`
- `src/types.rs:274-312` — `ClassifyConfig`
- `src/types.rs:298-311` — default threshold=60, min_descend=0.98, sample_exponent=0.47

### Benchmark tier construction (for context)
- `assignment_benchmarks/src/assignment_benchmarks/infrastructure/species_selector.py:1033-1067` — haplotype selection
- `:1069-1116` — genus-holdout selection
- `:1118-1151` — species-holdout selection
- `:197-266` — mixed-depth augmentation
- `:1194-1208` — normal-tier absorption of mixed-depth pseudo-labels
- `infrastructure/benchmark_shared.py:1243-1290` — reference writing
- `domain/metrics.py:1569-1621` — strict species-holdout TP rule
- `domain/metrics.py:1624-1681` — strict genus-holdout TP rule
- `domain/ground_truth.py:747-760` — per-tier eval rank overrides
- `infrastructure/tools/oxidtaxa/executor.py:45-140` — classify invocation with defaults

---

## Historical Context (from thoughts/)

- `thoughts/shared/research/2026-04-17-abstention-path-output-handling.md` — documents the TSV
  collapse that this research proposes fixing with I1. Reports the 37.5 % / 63.4 % empty-row rates
  on species- and genus-holdout tiers for config `t60_se0.65_md0.95_k9_rkf0.1`.
- `thoughts/shared/research/2026-04-17-species-genus-holdout-tier-improvements.md` — companion
  research focusing on the benchmark harness. The N1–N7 findings there about NA handling in the
  benchmark's selection code are resolved independently of oxidtaxa's own NA handling (I4); both
  are needed.
- `thoughts/shared/research/2026-04-15-new-parameter-audit.md` — source for F4 (`rank_thresholds`
  bug), F5 (`correlation_aware_features` binary-split degeneracy), F8 (`use_idf_in_training`
  hybrid), and the `leave_one_out` no-op history (fixed in commit `3d6cb91`, referenced in F10).
- `thoughts/shared/research/2026-04-14-confidence-scores-for-unclassified-asvs.md` — catalogs the
  three "unclassified" paths (A, B, C) and proposes option 4 (`classify(threshold=0)` +
  post-filter) as a no-code workaround. I1/I2 make that workaround unnecessary.
- `thoughts/shared/research/2026-04-08-tied-species-reporting-at-truncated-rank.md` — documents the
  per-replicate tie-split change that makes bit-identical training pairs surface as ties. I5
  generalizes that exact-equality filter to margin-based.
- `thoughts/shared/research/2026-04-05-rust-idtaxa-parameter-space.md` — parameter sweep history.
  Confirms `threshold`, `k`, and `sample_exponent` are the dominant knobs; I5 and I6 add new knobs
  orthogonal to these.
- `thoughts/shared/plans/2026-04-13-algorithmic-improvements.md` — rollout plan for the
  `leave_one_out`, `descendant_weighting`, `beam_width`, `correlation_aware_features` knobs, each
  behind a flag. Same pattern applies to I5/I6/I7.

## Related Research

- `thoughts/shared/research/2026-04-17-correlation-aware-training-bottlenecks.md` — training-time
  perf work, orthogonal to this robustness analysis.
- `thoughts/shared/research/2026-04-13-algorithmic-improvements.md` — the original 14 speedup /
  algorithmic improvements research that preceded this work.

## Open Questions

1. **I1 vs `ClassificationResult::unclassified()`**: if we emit Root-confidence rows for Path C, do
   we also need to change `::unclassified()` (`src/types.rs:169-175`) so Path A/B explicitly
   distinguish themselves at the Rust layer? Or is the TSV-layer signal enough? Answer affects
   whether the in-memory Python API also changes.

2. **I4 semantics for mixed taxonomy depths**: should the training tree treat two accessions,
   one `Family;Genus;Species` and one `Family;NA;NA`, as siblings at the family rank (both parking
   at the family leaf), or should the NA accession be a separate leaf at the family rank alongside
   the genus subtree? The first is simpler and matches the LCA-taxonomy semantic; the second
   preserves the accession's sequence as its own decision point.

3. **I5 tie_margin default**: empirically, what margin best separates genuine ties from
   noise-driven near-ties? A benchmark sweep over `tie_margin in [0.0, 0.02, 0.05, 0.10]` against
   species-holdout F1 would answer this. Default should stay 0.0 for bit-identical legacy until
   the sweep lands.

4. **I6 interaction with R-compatibility**: R's IDTAXA does not apply descent-margin weighting to
   confidence. If confidence-uses-descent-margin is ever made default, it invalidates R-golden
   tests. Stay off-by-default is the sensible path.

5. **I10 per-rank IDF storage cost**: for a 6-rank model with N = 100k unique k-mers,
   `Vec<Vec<f64>>` is 6 × 100k × 8 bytes = 4.8 MB — negligible. But for k = 12 spaced-seed models
   with N = 10M, it's 480 MB, too large. A sparse representation is needed for large-k models.

6. **Coupling with 4-tier benchmark metric wiring (P1 from companion research)**: If the benchmark
   starts passing `species_lca_rank` maps to `evaluate_accuracy_unified`, oxidtaxa's output needs
   to survive round-tripping through `expand_taxonomic_path` (`ground_truth.py:57-96`) as
   "truncated with known LCA rank". I1 + I3 + I4 together ensure the TSV path carries that
   information cleanly.

7. **Calibration anomaly (2026-04-17 report §6)**: TOS plausibility decreases as `threshold`
   increases on real data. Does any combination of I1/I5/I6 produce confidence values whose
   threshold relationship to TOS is monotonic? If not, a Platt/isotonic calibration layer on top
   of the raw confidence may be needed — separate from this research but pairs naturally with I8.
