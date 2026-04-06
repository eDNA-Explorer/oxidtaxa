/// Compute the effective alphabet size from nucleotide distribution entropy.
/// Returns exp(-sum(p_i * ln(p_i))) for the 4 DNA bases.
/// Port of enumerate_sequence.c:alphabetSize (lines 389-430).
pub fn alphabet_size(sequences: &[String]) -> f64 {
    let mut dist = [0.0f64; 4];

    for seq in sequences {
        for &b in seq.as_bytes() {
            match b {
                b'A' | b'a' => dist[0] += 1.0,
                b'C' | b'c' => dist[1] += 1.0,
                b'G' | b'g' => dist[2] += 1.0,
                b'T' | b't' => dist[3] += 1.0,
                _ => {}
            }
        }
    }

    let total: f64 = dist.iter().sum();
    if total == 0.0 {
        return 1.0;
    }

    let mut entropy = 0.0f64;
    for &count in &dist {
        let p = count / total;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}
