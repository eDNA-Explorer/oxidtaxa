use serde::{Deserialize, Serialize};

/// Decision node in the taxonomic tree.
/// Maps to R's `decision_kmers[[k]]` = list(keep_indices, profile_matrix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    /// K-mer indices used for classification decisions at this node.
    pub keep: Vec<i32>,
    /// Profile matrix: rows = child subtrees, cols = kept k-mers.
    pub profiles: Vec<Vec<f64>>,
    /// Raw sequence counts per child subtree, aligned with `keep`.
    /// `raw_counts[j][k]` = number of leaf-sequences in child subtree `j`
    /// that contain `keep[k]`. Used by leave-one-out at fraction-learning
    /// to subtract a held-out sequence's contribution from the matching
    /// sibling's profile counts and renormalize.
    pub raw_counts: Vec<Vec<f64>>,
    /// Total k-mer presence count per child subtree (summed over all k-mers
    /// across every leaf-sequence in the subtree, not just kept k-mers).
    /// Used as the LOO denominator: `new_total = raw_totals[j] - |kmers[i]|`.
    pub raw_totals: Vec<f64>,
}

/// A sequence that was misclassified during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSequence {
    pub index: usize,
    pub expected: String,
    pub predicted: String,
}

/// Trained IDTAXA model. Output of `learn_taxa()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSet {
    pub taxonomy: Vec<String>,
    pub taxa: Vec<String>,
    pub ranks: Option<Vec<String>>,
    pub levels: Vec<i32>,
    pub children: Vec<Vec<usize>>,
    pub parents: Vec<usize>,
    pub fraction: Vec<Option<f64>>,
    pub sequences: Vec<Option<Vec<usize>>>,
    /// Sorted unique k-mer indices per training sequence (1-indexed, matching R).
    pub kmers: Vec<Vec<i32>>,
    pub cross_index: Vec<usize>,
    pub k: usize,
    pub decision_kmers: Vec<Option<DecisionNode>>,
    pub problem_sequences: Vec<ProblemSequence>,
    pub problem_groups: Vec<String>,
    /// Spaced seed pattern used during training (e.g., "11011011011").
    /// None = contiguous k-mers (default).
    pub seed_pattern: Option<String>,
    /// Inverted k-mer index: for each k-mer id (0-indexed), sorted list of
    /// training sequence indices that contain it.
    pub inverted_index: Option<Vec<Vec<u32>>>,
    /// Per-rank IDF matrix. `idf_weights_by_rank[r][k]` is the IDF weight of
    /// k-mer `k` computed across distinct taxonomic prefixes at depth `r + 1`
    /// (so row 0 is Kingdom-level grouping, the deepest row is species-level).
    /// Classification picks the row matching the descent node's depth.
    ///
    /// Negative IDF values (`ln(N_r/(1+c)) < 0`) are intentional: when `c ≈
    /// N_r` a k-mer is universal at this rank, carries no discriminative
    /// signal, and SHOULD actively downweight the score. Variable-depth
    /// lineages contribute via `prefixes_at_rank` capping `r` at `parts.len()`
    /// — shallow lineages contribute to deep IDF rows as if they were
    /// species-resolved.
    pub idf_weights_by_rank: Vec<Vec<f64>>,
    /// Whether descent scoring (at both training and classify time) was
    /// configured to multiply per-child profile values by the rank-appropriate
    /// IDF row. Set from `TrainConfig.use_idf_in_descent` at the end of
    /// fraction learning. Read by `classify_one_pass` and
    /// `classify_one_pass_beam` to match the train-time descent algorithm.
    pub use_idf_in_descent: bool,
    /// The `descendant_weighting` mode used at tree-building time. Carried
    /// through from the originating `BuiltTree`. Metadata only — not used
    /// in any classify-time decision; surfaces so downstream tooling can
    /// label models.
    pub descendant_weighting: DescendantWeighting,
    /// Mean unique-k-mer count of training sequences in each node's subtree.
    /// `avg_n_unique_per_node[node]` is the average across all training
    /// sequences rooted at `node`. Used by classify-time `length_normalize`
    /// as a stable train-time average (replaces the per-query keep-local
    /// averaging that becomes a no-op at single-keep stop nodes).
    pub avg_n_unique_per_node: Vec<f64>,
    /// Mean unique-k-mer count across all training sequences (root-level
    /// average). Backstop for opaque nodes (those exceeding `max_children`).
    pub avg_n_unique_global: f64,
}

/// Intermediate training data: k-mer enumeration, taxonomy tree, and IDF weights.
///
/// Output of the "prepare" phase — everything computed from
/// (sequences, taxonomy, k, seed_pattern). Cache keyed on these params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedData {
    pub k: usize,
    pub n_kmers: usize,
    pub kmers: Vec<Vec<i32>>,
    pub inverted_index: Vec<Vec<u32>>,
    pub classes: Vec<String>,
    pub taxonomy: Vec<String>,
    pub taxa: Vec<String>,
    pub levels: Vec<i32>,
    pub children: Vec<Vec<usize>>,
    pub parents: Vec<usize>,
    pub end_taxonomy: Vec<String>,
    pub sequences_per_node: Vec<Option<Vec<usize>>>,
    pub n_seqs: Vec<usize>,
    pub cross_index: Vec<usize>,
    /// Per-rank IDF matrix: row `r` is the IDF computed across distinct
    /// taxonomic prefixes at depth `r + 1`. Always used at classify time in
    /// the leaf phase. Also used by descent (training and classification)
    /// when `use_idf_in_descent = true` on the trained model.
    pub idf_weights_by_rank: Vec<Vec<f64>>,
    pub seq_hashes: Vec<u64>,
    pub seed_pattern: Option<String>,
}

/// Decision tree nodes produced by feature selection.
///
/// Output of the "build tree" phase. Does NOT embed PreparedData —
/// learn_fractions() takes both as separate arguments. This keeps
/// serialized tree files small (~5-10 MB vs ~45 MB if embedded).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltTree {
    pub decision_kmers: Vec<Option<DecisionNode>>,
    /// The `descendant_weighting` mode used at feature-selection time.
    /// Persisted so downstream tooling (and `learn_fractions`) can label or
    /// branch on the weighting that produced this tree without re-deriving
    /// it from elsewhere.
    pub descendant_weighting: DescendantWeighting,
}

impl PreparedData {
    pub fn save(&self, path: &str) -> Result<(), String> {
        let encoded = bincode::serialize(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, encoded).map_err(|e| format!("write {path}: {e}"))
    }
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        bincode::deserialize(&data).map_err(|e| format!("deserialize {path}: {e}"))
    }
}

impl BuiltTree {
    pub fn save(&self, path: &str) -> Result<(), String> {
        let encoded = bincode::serialize(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, encoded).map_err(|e| format!("write {path}: {e}"))
    }
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        bincode::deserialize(&data).map_err(|e| format!("deserialize {path}: {e}"))
    }
}

/// Config for the tree-building phase.
pub struct BuildTreeConfig {
    pub record_kmers_fraction: f64,
    pub descendant_weighting: DescendantWeighting,
    pub correlation_aware_features: bool,
    pub max_children: usize,
    pub processors: usize,
}

/// Config for the fraction-learning phase.
pub struct LearnFractionsConfig {
    pub training_threshold: f64,
    pub use_idf_in_descent: bool,
    pub leave_one_out: bool,
    pub min_fraction: f64,
    pub max_fraction: f64,
    pub max_iterations: usize,
    pub multiplier: f64,
    pub processors: usize,
}

impl From<&TrainConfig> for BuildTreeConfig {
    fn from(c: &TrainConfig) -> Self {
        Self {
            record_kmers_fraction: c.record_kmers_fraction,
            descendant_weighting: c.descendant_weighting,
            correlation_aware_features: c.correlation_aware_features,
            max_children: c.max_children,
            processors: c.processors,
        }
    }
}

impl From<&TrainConfig> for LearnFractionsConfig {
    fn from(c: &TrainConfig) -> Self {
        Self {
            training_threshold: c.training_threshold,
            use_idf_in_descent: c.use_idf_in_descent,
            leave_one_out: c.leave_one_out,
            min_fraction: c.min_fraction,
            max_fraction: c.max_fraction,
            max_iterations: c.max_iterations,
            multiplier: c.multiplier,
            processors: c.processors,
        }
    }
}

/// Classification result for a single query sequence.
#[cfg_attr(feature = "python", pyo3::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
    /// Short-labels of all reference groups tied at the maximum `tot_hits`
    /// score during classification. Empty for non-tied classifications.
    /// When non-empty, the classifier was unable to distinguish between these
    /// leaves and has truncated `taxon` at their lowest common ancestor.
    /// Entries are sorted alphabetically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternatives: Vec<String>,
    /// Reason the classifier abstained (no/low signal). Values:
    /// `None` (classified), `Some("too_few_kmers")` (Path A: query too short),
    /// `Some("no_training_match")` (Path B: no compatible training seqs),
    /// `Some("below_threshold")` (Path C: Root confidence below threshold).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reject_reason: Option<String>,
    /// Leaf-phase similarity scalar. Average hits-per-bootstrap of the
    /// selected training-sequence group, normalized by the query's IDF-weighted
    /// k-mer sum. Zero on abstention paths.
    #[serde(default)]
    pub similarity: f64,
}

impl ClassificationResult {
    pub fn unclassified(reason: &str) -> Self {
        Self {
            taxon: vec!["Root".to_string()],
            confidence: vec![0.0],
            alternatives: Vec::new(),
            reject_reason: Some(reason.to_string()),
            similarity: 0.0,
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl ClassificationResult {
    fn __repr__(&self) -> String {
        let path = self.taxon.join(";");
        let alts_suffix = if self.alternatives.is_empty() {
            String::new()
        } else {
            format!(" alternatives={:?}", self.alternatives)
        };
        format!("ClassificationResult(taxon=\"{}\"{})", path, alts_suffix)
    }

    fn __len__(&self) -> usize {
        self.taxon.len()
    }
}

/// Output row for TSV file.
#[derive(Debug, Clone)]
pub struct TsvRow {
    pub read_id: String,
    pub taxonomic_path: String,
    pub confidence: f64,
}

/// Strategy for weighting child profiles during feature selection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DescendantWeighting {
    /// Weight by raw descendant count (original IDTAXA behavior).
    Count,
    /// Equal weight per immediate child (1/n_children each).
    Equal,
    /// Weight by log(1 + descendants).
    Log,
}

/// Configuration for training (LearnTaxa).
pub struct TrainConfig {
    pub k: Option<usize>,
    pub n: f64,
    pub min_fraction: f64,
    pub max_fraction: f64,
    pub max_iterations: usize,
    pub multiplier: f64,
    pub max_children: usize,
    /// Fraction of top cross-entropy k-mers retained at each decision node.
    /// Higher = more discriminating features but larger model. Default 0.10 (10%).
    pub record_kmers_fraction: f64,
    /// Spaced seed pattern (e.g., "11011011011"). None = contiguous k-mers.
    ///
    /// `weight = #1s` (number of bases sampled per window) and
    /// `span = pattern length` (number of bases the seed slides over). Validation
    /// rules — empty pattern, all-zeros pattern, non-binary characters, or
    /// `weight > 15` — are checked at the first call to `train()` /
    /// `prepare_data()`. Classify-time always reads the pattern from the saved
    /// model (no override knob), so train↔classify mismatch is impossible at
    /// the API level. Passing both `seed_pattern = Some(...)` and `k =
    /// Some(...)` is consistent only when `k == pattern.weight`; a mismatch
    /// is rejected at train time.
    pub seed_pattern: Option<String>,
    /// Bootstrap vote fraction required to descend during fraction learning.
    /// Default 0.8 matches R's hardcoded behavior. Set to match min_descend
    /// (e.g., 0.98) for consistent training/classification thresholds.
    pub training_threshold: f64,
    /// Strategy for weighting child profiles during feature selection.
    /// Modes (see `DescendantWeighting`):
    /// - `Count`: weight by raw descendant count (canonical IDTAXA, the default).
    /// - `Equal`: each immediate child gets weight `1/n_children`.
    /// - `Log`: weight by `ln(1 + descendants)` — compresses dominant subtrees
    ///   without erasing them.
    ///
    /// Affects training only — the chosen mode is baked into `decision_kmers`
    /// at tree-build time. Classify-time descent reads the resulting profiles.
    pub descendant_weighting: DescendantWeighting,
    /// Apply rank-appropriate IDF weights to per-child profile values during
    /// tree descent — at both training (fraction-learning) and classification.
    /// When true, descent scoring uses `profiles[j] * idf_row[depth]`. When
    /// false, descent scoring uses raw `profiles[j]`. Persisted on the trained
    /// model so classify-time descent matches the algorithm used at train
    /// time. Default false.
    pub use_idf_in_descent: bool,
    /// Exclude each sequence from its own subtree's profile during fraction
    /// learning (per-kmer leave-one-out). Subtracts the held-out sequence's
    /// contribution from the matching sibling's raw counts, then renormalizes.
    /// Singleton k-mers (held-out sequence is the only contributor) go to zero;
    /// conserved k-mers (every group member has them) stay unchanged.
    /// Reduces self-classification bias by removing circular evidence at the
    /// fraction-calibration step (does not affect classify-time scoring; the
    /// stored `dk.profiles` retain full discriminative information).
    /// Default false (legacy behavior).
    pub leave_one_out: bool,
    /// Use correlation-aware greedy feature selection instead of independent
    /// round-robin. Uses Bhattacharyya coefficient on L1-normalized sqrt
    /// profiles as the redundancy metric (mathematically justified for any
    /// split size, including `n_children = 2` where Pearson degenerates).
    /// Produces a more efficient feature set but slower to train: O(R · C)
    /// per node (R = `record_kmers`, C = candidate pool size), parallelized
    /// when `n_cand >= 2048`. No impact on classify speed. Default false.
    pub correlation_aware_features: bool,
    /// Number of threads for the rayon thread pool. Default 1.
    pub processors: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            k: None,
            n: 500.0,
            min_fraction: 0.01,
            max_fraction: 0.06,
            max_iterations: 10,
            multiplier: 100.0,
            max_children: 200,
            record_kmers_fraction: 0.10,
            seed_pattern: None,
            training_threshold: 0.8,
            descendant_weighting: DescendantWeighting::Count,
            use_idf_in_descent: false,
            leave_one_out: false,
            correlation_aware_features: false,
            processors: 1,
        }
    }
}

/// Configuration for classification (IdTaxa).
pub struct ClassifyConfig {
    pub threshold: f64,
    /// Number of bootstrap replicates (ceiling; short sequences may get fewer).
    /// Higher = more precise confidence, slower. Default 100.
    pub bootstraps: usize,
    pub min_descend: f64,
    pub processors: usize,
    /// Exponent for computing k-mers sampled per bootstrap: `S = L^sample_exponent`.
    /// Lower = fewer k-mers per replicate (faster, noisier). Default 0.47
    /// (canonical IDTAXA).
    ///
    /// Note: `S` is sized against `not_nas` (raw stream count after NA filter,
    /// before dedup), while the `too_few_kmers` abstention guard checks
    /// `my_kmers.len()` (post-dedup unique k-mer count). The asymmetry is
    /// intentional — `S` reflects apparent query complexity, the abstention
    /// guard reflects the unique-signal floor, so a low-complexity query (many
    /// repeated k-mers) is correctly caught.
    ///
    /// Implicit Pareto knob: classify-time bootstraps are capped at
    /// `b = min(5L/S, bootstraps)`, so lowering `sample_exponent` lets `b`
    /// climb toward the configured cap.
    pub sample_exponent: f64,
    /// Normalize per-reference leaf-phase scores by `sqrt(n_unique / avg)`,
    /// where `n_unique` is the unique-k-mer count of the training sequence
    /// (after seed/dedup) and `avg` is the keep-local mean unique-k-mer count
    /// of the candidate pool at the leaf-phase stop node. Symmetric: long
    /// references are demoted, short references are promoted. Default false.
    pub length_normalize: bool,
    /// Per-rank confidence thresholds applied during the Stage-3 threshold
    /// walk. When `Some(non-empty)`, `rank_thresholds[i]` is used for depth
    /// `i` (0=Root) and the global `threshold` is fully ignored — there is no
    /// per-element fallback. When `None`, the single `threshold` applies to
    /// every rank.
    ///
    /// Edge cases:
    /// - **Empty list** (`Some(vec![])`) is rejected at the Python binding;
    ///   in pure-Rust callers it would silently behave like `None`.
    /// - **Longer than path**: extra trailing entries are ignored.
    /// - **Shorter than path**: the last value is reused for every rank past
    ///   its end (matches the README "shorter list reuse last" contract).
    /// - **Element scale**: each element is on the same `[0, 100]` scale as
    ///   `threshold`. Out-of-range values are rejected at the Python binding.
    pub rank_thresholds: Option<Vec<f64>>,
    /// Number of candidate paths to maintain during tree descent.
    /// 1 = greedy descent (original behavior). Higher values explore
    /// alternative paths at ambiguous nodes. Default 1.
    ///
    /// **When to flip.** Beam diverges from greedy *only* when ≥2 children
    /// clear `min_descend` at the same split — typically requires either
    /// tied bootstraps (deterministic, common with byte-identical training
    /// references) or a relaxed `min_descend`. Under the default `min_descend
    /// = 0.98`, the divergence is rare. Sub-threshold runner-ups are not
    /// rescued by beam — see the explicit drop at `classify::classify_one_pass_beam`
    /// (gated runner-up retention).
    pub beam_width: usize,
    /// Relative margin below the max `tot_hits` within which sibling leaves are
    /// treated as tied winners for LCA-cap and `alternatives` reporting. At 0.0
    /// (default) only exact equalities fire, matching legacy behavior. At e.g.
    /// 0.05, any group scoring within 95% of the winner joins the tied set.
    pub tie_margin: f64,
    /// When true, each rank's confidence is discounted by the *per-rank*
    /// margin of the single descent step that selected it — non-cumulative.
    /// At each descent split, the raw margin `m = (top - runner_up) / b` is
    /// recorded (floored at 0.1 so a zero-runner-up doesn't zero out a rank).
    /// At leaf-phase, each rank's confidence is multiplied by the affine
    /// remap `MARGIN_FLOOR + (1 - MARGIN_FLOOR) * m` with `MARGIN_FLOOR = 0.8`,
    /// giving an effective per-rank multiplier in `[0.82, 1.0]`. Root is
    /// never discounted. Default false (legacy behavior).
    pub confidence_uses_descent_margin: bool,
    /// When true, widen `w_indices` to include siblings with
    /// `vote_counts[j] >= sibling_aware_min_vote_frac * b` at descent
    /// stopping points. Two sites apply this:
    ///
    /// 1. **Mech 2 — terminal sibling widening:** when greedy descent
    ///    succeeds at a leaf-parent (single child cleared `min_descend`),
    ///    keep that winner plus any sibling clearing the strict frac threshold.
    /// 2. **Mech 1 — halt-in-the-middle fallback:** when descent halts
    ///    because `|w| ≠ 1`, replace the loose `> 0 votes` filter with the
    ///    same strict frac threshold, preserving the empty-`w50` fallback to
    ///    ALL children for scattered cases.
    ///
    /// Both sites use the same `sibling_aware_min_vote_frac` knob. Default
    /// false (legacy: only the single winner contributes at Mech 2; Mech 1
    /// uses the loose `> 0` filter when `w50` non-empty).
    pub sibling_aware_leaf: bool,
    /// Strict vote-fraction threshold used by `sibling_aware_leaf` at both
    /// mid-tree halt (Mech 1) and terminal-sibling-widening (Mech 2) sites.
    /// A sibling whose `vote_counts[j] >= sibling_aware_min_vote_frac * b`
    /// joins the candidate set. Default 0.5 (matches the previously
    /// hardcoded constant).
    pub sibling_aware_min_vote_frac: f64,
    /// When true, dual-stage prefix suppression for ancestor-only training
    /// entries (e.g. "Oncorhynchus sp." after canonical NA-trim):
    ///
    /// 1. **Share-split stage** (per replicate, inside the bootstrap loop):
    ///    drop any group whose taxonomy path is a strict prefix of another
    ///    tied group's path BEFORE per-replicate `share = 1/n_tied` is
    ///    computed. Restores the descendant's full per-replicate credit
    ///    instead of halving it via parasitic prefix ties.
    /// 2. **Winner stage** (post-bootstrap, on `winners`): drop any winner
    ///    whose path is a strict prefix of another winner's path. Prevents
    ///    LCA-cap collapse when prefix-related groups land at equal
    ///    `tot_hits` from disjoint-replicate wins.
    ///
    /// The ancestor's outright-win replicates (where it strictly beats every
    /// descendant on its own evidence) are unaffected — the share-split-stage
    /// filter only fires when `n_tied > 1`. Ancestor-rank confidence is
    /// conserved relative to flag=off because the cross-rank accumulator
    /// climbs the descendant's full credit through `parents[]`. Default
    /// false (legacy behavior).
    ///
    /// Stage 1's tie definition tracks `tie_margin`: with `tie_margin = 0.0`
    /// (default), ties must be byte-exact equalities (`==`). With
    /// `tie_margin > 0`, the share-split stage fires on near-ties using the
    /// same relaxed cutoff as the post-bootstrap winner stage.
    pub suppress_ancestor_only_groups: bool,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self {
            threshold: 60.0,
            bootstraps: 100,
            min_descend: 0.98,
            processors: 1,
            sample_exponent: 0.47,
            length_normalize: false,
            rank_thresholds: None,
            beam_width: 1,
            tie_margin: 0.0,
            confidence_uses_descent_margin: false,
            sibling_aware_leaf: false,
            sibling_aware_min_vote_frac: 0.5,
            suppress_ancestor_only_groups: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrandMode {
    Top,
    Bottom,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputType {
    Extended,
    Collapsed,
}

impl TrainingSet {
    /// Save to bincode format.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let encoded =
            bincode::serialize(self).map_err(|e| format!("Serialization error: {}", e))?;
        std::fs::write(path, encoded).map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }

    /// Load from bincode format.
    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("Read error: {}", e))?;
        bincode::deserialize(&data).map_err(|e| format!("Deserialization error: {}", e))
    }
}
