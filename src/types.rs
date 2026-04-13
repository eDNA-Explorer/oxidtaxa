use serde::{Deserialize, Serialize};

/// Decision node in the taxonomic tree.
/// Maps to R's `decision_kmers[[k]]` = list(keep_indices, profile_matrix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode {
    /// K-mer indices used for classification decisions at this node.
    pub keep: Vec<i32>,
    /// Profile matrix: rows = child subtrees, cols = kept k-mers.
    pub profiles: Vec<Vec<f64>>,
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
    pub idf_weights: Vec<f64>,
    pub decision_kmers: Vec<Option<DecisionNode>>,
    pub problem_sequences: Vec<ProblemSequence>,
    pub problem_groups: Vec<String>,
    /// Spaced seed pattern used during training (e.g., "11011011011").
    /// None = contiguous k-mers (default).
    pub seed_pattern: Option<String>,
    /// Inverted k-mer index: for each k-mer id (0-indexed), sorted list of
    /// training sequence indices that contain it.
    pub inverted_index: Option<Vec<Vec<u32>>>,
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
}

impl ClassificationResult {
    pub fn unclassified() -> Self {
        Self {
            taxon: vec!["Root".to_string(), "unclassified_Root".to_string()],
            confidence: vec![0.0, 0.0],
            alternatives: Vec::new(),
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub seed_pattern: Option<String>,
    /// Bootstrap vote fraction required to descend during fraction learning.
    /// Default 0.8 matches R's hardcoded behavior. Set to match min_descend
    /// (e.g., 0.98) for consistent training/classification thresholds.
    pub training_threshold: f64,
    /// Strategy for weighting child profiles during feature selection.
    /// Default: Count (original behavior).
    pub descendant_weighting: DescendantWeighting,
    /// Use IDF weights (instead of profile weights) during the fraction-learning
    /// tree descent. Makes training scoring match classification scoring.
    /// Default false (original behavior uses profile weights).
    pub use_idf_in_training: bool,
    /// Exclude each sequence from its own node's profile during fraction
    /// learning (leave-one-out). Reduces self-classification bias for small
    /// groups. Default false (original behavior).
    pub leave_one_out: bool,
    /// Use correlation-aware greedy feature selection instead of independent
    /// round-robin. Selects k-mers that maximize conditional information gain.
    /// Produces a more efficient feature set but slower to train. Default false.
    pub correlation_aware_features: bool,
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
            use_idf_in_training: false,
            leave_one_out: false,
            correlation_aware_features: false,
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
    pub full_length: f64,
    pub processors: usize,
    /// Exponent for computing k-mers sampled per bootstrap: S = L^sample_exponent.
    /// Lower = fewer k-mers per replicate (faster, noisier). Default 0.47.
    pub sample_exponent: f64,
    /// Normalize scores by training sequence length. Corrects inflation from
    /// longer references having more k-mers. Default false.
    pub length_normalize: bool,
    /// Per-rank confidence thresholds. When Some, rank_thresholds[i] is used for
    /// depth i (0=Root). When None, uses single `threshold` for all ranks.
    pub rank_thresholds: Option<Vec<f64>>,
    /// Number of candidate paths to maintain during tree descent.
    /// 1 = greedy descent (original behavior). Higher values explore
    /// alternative paths at ambiguous nodes. Default 1.
    pub beam_width: usize,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self {
            threshold: 60.0,
            bootstraps: 100,
            min_descend: 0.98,
            full_length: 0.0,
            processors: 1,
            sample_exponent: 0.47,
            length_normalize: false,
            rank_thresholds: None,
            beam_width: 1,
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
