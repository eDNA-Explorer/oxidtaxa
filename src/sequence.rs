use rayon::prelude::*;

/// Count occurrences of a single byte pattern in each sequence.
/// Port of R/seq_utils.R:vcountPattern.
pub fn vcount_pattern(pattern: u8, sequences: &[String]) -> Vec<usize> {
    sequences
        .iter()
        .map(|seq| seq.bytes().filter(|&b| b == pattern).count())
        .collect()
}

/// Reverse complement of a DNA sequence.
/// Handles IUPAC ambiguity codes.
/// Port of R/seq_utils.R:reverseComplement.
/// Works at byte level for performance (all DNA chars are ASCII).
pub fn reverse_complement(seq: &str) -> String {
    let bytes: Vec<u8> = seq.bytes()
        .rev()
        .map(|b| match b {
            b'A' | b'a' => b'T',
            b'T' | b't' => b'A',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'M' | b'm' => b'K',
            b'K' | b'k' => b'M',
            b'R' | b'r' => b'Y',
            b'Y' | b'y' => b'R',
            b'W' | b'w' => b'W',
            b'S' | b's' => b'S',
            b'V' | b'v' => b'B',
            b'B' | b'b' => b'V',
            b'H' | b'h' => b'D',
            b'D' | b'd' => b'H',
            b'N' | b'n' => b'N',
            other => other,
        })
        .collect();
    // SAFETY: input is ASCII DNA, complement mapping preserves ASCII
    unsafe { String::from_utf8_unchecked(bytes) }
}

/// Remove gap characters ('-', '.') from sequences.
/// Port of src/remove_gaps.c.
/// Uses rayon for parallelism. Works at byte level for performance.
pub fn remove_gaps(sequences: &[String]) -> Vec<String> {
    sequences
        .par_iter()
        .map(|seq| {
            let bytes: Vec<u8> = seq.bytes().filter(|&b| b != b'-' && b != b'.').collect();
            // SAFETY: input is ASCII DNA, filtering preserves valid UTF-8
            unsafe { String::from_utf8_unchecked(bytes) }
        })
        .collect()
}
