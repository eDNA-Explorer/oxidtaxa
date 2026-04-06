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
pub fn write_classification_tsv(
    path: &str,
    names: &[String],
    results: &[crate::types::ClassificationResult],
) -> Result<(), String> {
    let mut output = String::from("read_id\ttaxonomic_path\tconfidence\n");

    for (i, result) in results.iter().enumerate() {
        let read_id = names[i]
            .split_whitespace()
            .next()
            .unwrap_or(&names[i]);

        let mut taxa = result.taxon.clone();
        let mut conf = result.confidence.clone();

        // Skip Root, filter unclassified
        if taxa.len() > 1 {
            taxa.remove(0);
            conf.remove(0);
            let mut filtered_taxa = Vec::new();
            let mut filtered_conf = Vec::new();
            for (t, c) in taxa.iter().zip(conf.iter()) {
                if !t.starts_with("unclassified_") {
                    filtered_taxa.push(t.as_str());
                    filtered_conf.push(*c);
                }
            }
            if !filtered_taxa.is_empty() {
                let path_str = filtered_taxa.join(";");
                let min_conf = filtered_conf
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                output.push_str(&format!("{}\t{}\t{}\n", read_id, path_str, min_conf));
            } else {
                output.push_str(&format!("{}\t\t0\n", read_id));
            }
        } else {
            output.push_str(&format!("{}\t\t0\n", read_id));
        }
    }

    std::fs::write(path, output).map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}
