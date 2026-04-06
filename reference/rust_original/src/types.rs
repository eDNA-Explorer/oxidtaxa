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
}

/// Classification result for a single query sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub taxon: Vec<String>,
    pub confidence: Vec<f64>,
}

impl ClassificationResult {
    pub fn unclassified() -> Self {
        Self {
            taxon: vec!["Root".to_string(), "unclassified_Root".to_string()],
            confidence: vec![0.0, 0.0],
        }
    }
}

/// Output row for TSV file.
#[derive(Debug, Clone)]
pub struct TsvRow {
    pub read_id: String,
    pub taxonomic_path: String,
    pub confidence: f64,
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
