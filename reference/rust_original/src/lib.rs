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
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyfunction]
    #[pyo3(signature = (
        fasta_path, taxonomy_path, output_path,
        seed = 42, k = None, record_kmers_fraction = 0.10, verbose = true
    ))]
    fn train(
        fasta_path: &str,
        taxonomy_path: &str,
        output_path: &str,
        seed: u32,
        k: Option<usize>,
        record_kmers_fraction: f64,
        verbose: bool,
    ) -> PyResult<()> {
        let (names, seqs) =
            crate::fasta::read_fasta(fasta_path).map_err(|e| PyValueError::new_err(e))?;

        let taxonomy =
            crate::fasta::read_taxonomy(taxonomy_path, &names).map_err(|e| PyValueError::new_err(e))?;

        // Quality filtering (same as train_idtaxa.R)
        let (filtered_seqs, filtered_tax) = filter_for_training(&seqs, &taxonomy);

        let mut rng = crate::rng::RRng::new(seed);
        let config = crate::types::TrainConfig {
            k,
            record_kmers_fraction,
            ..Default::default()
        };
        let model = crate::training::learn_taxa(&filtered_seqs, &filtered_tax, &config, &mut rng, verbose)
            .map_err(|e| PyValueError::new_err(e))?;

        model.save(output_path).map_err(|e| PyValueError::new_err(e))?;
        Ok(())
    }

    #[pyfunction]
    #[pyo3(signature = (
        query_path, model_path, output_path,
        threshold = 60.0, bootstraps = 100, strand = "both",
        min_descend = 0.98, full_length = 0.0, processors = 1,
        sample_exponent = 0.47, seed = 42, deterministic = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn classify(
        query_path: &str,
        model_path: &str,
        output_path: &str,
        threshold: f64,
        bootstraps: usize,
        strand: &str,
        min_descend: f64,
        full_length: f64,
        processors: usize,
        sample_exponent: f64,
        seed: u32,
        deterministic: bool,
    ) -> PyResult<()> {
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
        };
        let results = crate::classify::id_taxa(
            &clean_seqs,
            &names,
            &model,
            &config,
            strand_mode,
            crate::types::OutputType::Extended,
            seed,
            deterministic,
        );

        crate::fasta::write_classification_tsv(output_path, &names, &results)
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(())
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
        Ok(())
    }
}
