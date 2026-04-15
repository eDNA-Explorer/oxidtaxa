pub mod alphabet;
pub mod classify;
pub mod fasta;
pub mod kmer;
pub mod matching;
pub mod rng;
pub mod sequence;
pub mod training;
pub mod types;

#[cfg(feature = "python")]
mod python_bindings {
    use std::sync::Arc;

    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyclass(frozen, name = "PreparedData")]
    struct PyPreparedData {
        inner: Arc<crate::types::PreparedData>,
    }

    #[pymethods]
    impl PyPreparedData {
        fn save(&self, path: &str) -> PyResult<()> {
            self.inner.save(path).map_err(|e| PyValueError::new_err(e))
        }
        #[staticmethod]
        fn load(path: &str) -> PyResult<Self> {
            let inner = crate::types::PreparedData::load(path)
                .map_err(|e| PyValueError::new_err(e))?;
            Ok(Self { inner: Arc::new(inner) })
        }
        #[getter]
        fn n_sequences(&self) -> usize { self.inner.kmers.len() }
        #[getter]
        fn k(&self) -> usize { self.inner.k }
        #[getter]
        fn n_taxa(&self) -> usize { self.inner.taxonomy.len() }
    }

    #[pyclass(frozen, name = "BuiltTree")]
    struct PyBuiltTree {
        inner: Arc<crate::types::BuiltTree>,
    }

    #[pymethods]
    impl PyBuiltTree {
        fn save(&self, path: &str) -> PyResult<()> {
            self.inner.save(path).map_err(|e| PyValueError::new_err(e))
        }
        #[staticmethod]
        fn load(path: &str) -> PyResult<Self> {
            let inner = crate::types::BuiltTree::load(path)
                .map_err(|e| PyValueError::new_err(e))?;
            Ok(Self { inner: Arc::new(inner) })
        }
        #[getter]
        fn n_nodes(&self) -> usize { self.inner.decision_kmers.len() }
    }

    #[pyfunction]
    #[pyo3(signature = (
        fasta_path, taxonomy_path, output_path,
        seed = 42, k = None, record_kmers_fraction = 0.10, verbose = true,
        seed_pattern = None, training_threshold = 0.8,
        descendant_weighting = "count", use_idf_in_training = false,
        leave_one_out = false, correlation_aware_features = false,
        processors = 1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn train(
        py: Python<'_>,
        fasta_path: &str,
        taxonomy_path: &str,
        output_path: &str,
        seed: u32,
        k: Option<usize>,
        record_kmers_fraction: f64,
        verbose: bool,
        seed_pattern: Option<String>,
        training_threshold: f64,
        descendant_weighting: &str,
        use_idf_in_training: bool,
        leave_one_out: bool,
        correlation_aware_features: bool,
        processors: usize,
    ) -> PyResult<()> {
        let (names, seqs) =
            crate::fasta::read_fasta(fasta_path).map_err(|e| PyValueError::new_err(e))?;

        let taxonomy =
            crate::fasta::read_taxonomy(taxonomy_path, &names).map_err(|e| PyValueError::new_err(e))?;

        // Quality filtering (same as train_idtaxa.R)
        let (filtered_seqs, filtered_tax) = filter_for_training(&seqs, &taxonomy);

        let dw = parse_descendant_weighting(descendant_weighting)?;
        let config = crate::types::TrainConfig {
            k,
            record_kmers_fraction,
            seed_pattern,
            training_threshold,
            descendant_weighting: dw,
            use_idf_in_training,
            leave_one_out,
            correlation_aware_features,
            processors,
            ..Default::default()
        };

        // Release the GIL so parallel Python threads can run concurrently
        let model = py.allow_threads(|| {
            crate::training::learn_taxa(&filtered_seqs, &filtered_tax, &config, seed, verbose)
        }).map_err(|e| PyValueError::new_err(e))?;

        model.save(output_path).map_err(|e| PyValueError::new_err(e))?;
        Ok(())
    }

    /// Classify query sequences against a trained IDTAXA model.
    ///
    /// Returns a list of `ClassificationResult` objects — one per input
    /// sequence, in the same order as the query FASTA. Each result exposes:
    /// - `taxon`: list[str] — root-to-leaf lineage
    /// - `confidence`: list[float] — per-rank confidence percentages
    /// - `alternatives`: list[str] — tied species short-labels when the
    ///   classifier could not resolve between multiple equally-scored
    ///   references (empty for non-tied classifications)
    ///
    /// If `output_path` is provided, a TSV with columns
    /// `read_id, taxonomic_path, confidence, alternatives` is also written to
    /// that path. If omitted, no file is written and results are only
    /// returned in-memory.
    #[pyfunction]
    #[pyo3(signature = (
        query_path, model_path, output_path = None,
        threshold = 60.0, bootstraps = 100, strand = "both",
        min_descend = 0.98, full_length = 0.0, processors = 1,
        sample_exponent = 0.47, seed = 42, deterministic = false,
        length_normalize = false, rank_thresholds = None,
        beam_width = 1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn classify(
        py: Python<'_>,
        query_path: &str,
        model_path: &str,
        output_path: Option<String>,
        threshold: f64,
        bootstraps: usize,
        strand: &str,
        min_descend: f64,
        full_length: f64,
        processors: usize,
        sample_exponent: f64,
        seed: u32,
        deterministic: bool,
        length_normalize: bool,
        rank_thresholds: Option<Vec<f64>>,
        beam_width: usize,
    ) -> PyResult<Vec<crate::types::ClassificationResult>> {
        let model = crate::types::TrainingSet::load(model_path)
            .map_err(|e| PyValueError::new_err(e))?;

        let (names, seqs) =
            crate::fasta::read_fasta(query_path).map_err(|e| PyValueError::new_err(e))?;

        let clean_seqs = crate::sequence::remove_gaps(&seqs);

        let strand_mode = parse_strand(strand)?;
        let config = crate::types::ClassifyConfig {
            threshold,
            bootstraps,
            min_descend,
            full_length,
            processors,
            sample_exponent,
            length_normalize,
            rank_thresholds,
            beam_width,
        };

        // Release the GIL so parallel Python threads can run concurrently
        let results = py.allow_threads(|| {
            crate::classify::id_taxa(
                &clean_seqs,
                &names,
                &model,
                &config,
                strand_mode,
                crate::types::OutputType::Extended,
                seed,
                deterministic,
            )
        });

        if let Some(ref path) = output_path {
            crate::fasta::write_classification_tsv(path, &names, &results)
                .map_err(|e| PyValueError::new_err(e))?;
        }

        Ok(results)
    }

    #[pyfunction]
    #[pyo3(signature = (
        fasta_path, taxonomy_path, k = None, n = 500.0,
        seed_pattern = None, processors = 1
    ))]
    fn prepare_data_py(
        py: Python<'_>,
        fasta_path: &str,
        taxonomy_path: &str,
        k: Option<usize>,
        n: f64,
        seed_pattern: Option<String>,
        processors: usize,
    ) -> PyResult<PyPreparedData> {
        let (names, seqs) =
            crate::fasta::read_fasta(fasta_path).map_err(|e| PyValueError::new_err(e))?;
        let taxonomy =
            crate::fasta::read_taxonomy(taxonomy_path, &names).map_err(|e| PyValueError::new_err(e))?;
        let (filtered_seqs, filtered_tax) = filter_for_training(&seqs, &taxonomy);

        let inner = py.allow_threads(|| {
            crate::training::prepare_data(
                &filtered_seqs, &filtered_tax, k, n, seed_pattern, processors,
            )
        }).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyPreparedData { inner: Arc::new(inner) })
    }

    #[pyfunction]
    #[pyo3(signature = (
        prepared, record_kmers_fraction = 0.10, descendant_weighting = "count",
        correlation_aware_features = false, processors = 1
    ))]
    fn build_tree_py(
        py: Python<'_>,
        prepared: &PyPreparedData,
        record_kmers_fraction: f64,
        descendant_weighting: &str,
        correlation_aware_features: bool,
        processors: usize,
    ) -> PyResult<PyBuiltTree> {
        let dw = parse_descendant_weighting(descendant_weighting)?;
        let config = crate::types::BuildTreeConfig {
            record_kmers_fraction,
            descendant_weighting: dw,
            correlation_aware_features,
            max_children: 200,
            processors,
        };
        let data = Arc::clone(&prepared.inner);
        let built = py.allow_threads(move || {
            crate::training::build_tree(&data, &config)
        }).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyBuiltTree { inner: Arc::new(built) })
    }

    #[pyfunction]
    #[pyo3(signature = (
        prepared, built_tree, output_path, seed = 42, training_threshold = 0.8,
        use_idf_in_training = false, leave_one_out = false, processors = 1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn learn_fractions_py(
        py: Python<'_>,
        prepared: &PyPreparedData,
        built_tree: &PyBuiltTree,
        output_path: &str,
        seed: u32,
        training_threshold: f64,
        use_idf_in_training: bool,
        leave_one_out: bool,
        processors: usize,
    ) -> PyResult<()> {
        let config = crate::types::LearnFractionsConfig {
            training_threshold,
            use_idf_in_training,
            leave_one_out,
            min_fraction: 0.01,
            max_fraction: 0.06,
            max_iterations: 10,
            multiplier: 100.0,
            processors,
        };
        let prep = Arc::clone(&prepared.inner);
        let tree = Arc::clone(&built_tree.inner);
        let model = py.allow_threads(move || {
            crate::training::learn_fractions(&prep, &tree, &config, seed)
        }).map_err(|e| PyValueError::new_err(e))?;
        model.save(output_path).map_err(|e| PyValueError::new_err(e))?;
        Ok(())
    }

    fn parse_descendant_weighting(s: &str) -> PyResult<crate::types::DescendantWeighting> {
        match s {
            "count" => Ok(crate::types::DescendantWeighting::Count),
            "equal" => Ok(crate::types::DescendantWeighting::Equal),
            "log" => Ok(crate::types::DescendantWeighting::Log),
            _ => Err(PyValueError::new_err(format!(
                "Invalid descendant_weighting: '{}'. Expected 'count', 'equal', or 'log'", s
            ))),
        }
    }

    fn parse_strand(strand: &str) -> PyResult<crate::types::StrandMode> {
        match strand {
            "both" => Ok(crate::types::StrandMode::Both),
            "top" => Ok(crate::types::StrandMode::Top),
            "bottom" => Ok(crate::types::StrandMode::Bottom),
            _ => Err(PyValueError::new_err(format!("Invalid strand: {}", strand))),
        }
    }

    fn filter_for_training(seqs: &[String], taxonomy: &[String]) -> (Vec<String>, Vec<String>) {
        let mut filtered_seqs = Vec::new();
        let mut filtered_tax = Vec::new();

        for (i, seq) in seqs.iter().enumerate() {
            let tax = &taxonomy[i];
            // Prepend "Root; " and normalize
            let full_tax = format!("Root; {}", tax.replace(";", "; "));
            let rank_count = full_tax.split("; ").count();

            // Quality filters: >= 4 ranks, >= 30bp, <= 30% N
            if rank_count < 4 {
                continue;
            }
            let len = seq.len();
            if len < 30 {
                continue;
            }
            let n_count = seq.bytes().filter(|&b| b == b'N' || b == b'n').count();
            if (n_count as f64 / len as f64) > 0.3 {
                continue;
            }

            filtered_seqs.push(seq.clone());
            filtered_tax.push(full_tax);
        }

        (filtered_seqs, filtered_tax)
    }

    #[pymodule]
    fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(pyo3::wrap_pyfunction!(train, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(classify, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(prepare_data_py, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(build_tree_py, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(learn_fractions_py, m)?)?;
        m.add_class::<crate::types::ClassificationResult>()?;
        m.add_class::<PyPreparedData>()?;
        m.add_class::<PyBuiltTree>()?;
        Ok(())
    }
}
