/// Read a FASTA file, returning (names, sequences).
/// All sequences are uppercased. Matches R's readFasta().
pub fn read_fasta(path: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Cannot read {}: {}", path, e))?;

    let mut names = Vec::new();
    let mut sequences = Vec::new();
    let mut current_seq = String::new();

    for line in content.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if !names.is_empty() {
                sequences.push(current_seq.to_uppercase());
                current_seq = String::new();
            }
            names.push(header.to_string());
        } else {
            current_seq.push_str(line);
        }
    }
    if !names.is_empty() {
        sequences.push(current_seq.to_uppercase());
    }

    Ok((names, sequences))
}

/// Read a taxonomy TSV file (accession\ttaxonomy), mapping to sequence names.
/// Returns taxonomy strings in the same order as the provided names.
pub fn read_taxonomy(
    path: &str,
    names: &[String],
) -> Result<Vec<String>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Cannot read {}: {}", path, e))?;

    let mut acc_to_tax = std::collections::HashMap::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.splitn(2, '\t').collect();
        if parts.len() == 2 {
            acc_to_tax.insert(parts[0].to_string(), parts[1].to_string());
        }
    }

    let mut taxonomy = Vec::with_capacity(names.len());
    for name in names {
        // Extract accession (first word before space)
        let accession = name.split_whitespace().next().unwrap_or(name);
        match acc_to_tax.get(accession) {
            Some(tax) => taxonomy.push(tax.clone()),
            None => return Err(format!("No taxonomy found for accession: {}", accession)),
        }
    }

    Ok(taxonomy)
}

/// Write classification results as TSV.
///
/// Schema (6 columns):
///   read_id, taxonomic_path, confidence, alternatives, reject_reason, similarity
///
/// Empty-path rows preserve `result.confidence[0]` (the Root-level
/// accumulated confidence) in the confidence column rather than hardcoding 0,
/// so Path-C abstentions (below-threshold) are distinguishable from Path-A/B
/// abstentions (no signal at all) by confidence > 0.
pub fn write_classification_tsv(
    path: &str,
    names: &[String],
    results: &[crate::types::ClassificationResult],
) -> Result<(), String> {
    let mut output = String::from(
        "read_id\ttaxonomic_path\tconfidence\talternatives\treject_reason\tsimilarity\n",
    );

    for (i, result) in results.iter().enumerate() {
        let read_id = names[i]
            .split_whitespace()
            .next()
            .unwrap_or(&names[i]);

        let alternatives_field = result.alternatives.join("|");
        let reject_reason_field = result.reject_reason.as_deref().unwrap_or("");
        let similarity_field = result.similarity;

        // Skip the leading "Root" element; remaining ranks form the path.
        if result.taxon.len() > 1 {
            let path_str = result.taxon[1..].join(";");
            let min_conf = result.confidence[1..]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            output.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\n",
                read_id,
                path_str,
                min_conf,
                alternatives_field,
                reject_reason_field,
                similarity_field,
            ));
        } else {
            let c0 = result.confidence.first().copied().unwrap_or(0.0);
            output.push_str(&format!(
                "{}\t\t{}\t{}\t{}\t{}\n",
                read_id,
                c0,
                alternatives_field,
                reject_reason_field,
                similarity_field,
            ));
        }
    }

    std::fs::write(path, output).map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}
