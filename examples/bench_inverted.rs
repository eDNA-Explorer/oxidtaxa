/// Quick benchmark: merge-join vs inverted index at different keep sizes.
/// Run: cargo run --example bench_inverted --release
use std::time::Instant;

use oxidaxa::fasta::{read_fasta, read_taxonomy};
use oxidaxa::kmer::{enumerate_sequences, NA_INTEGER};
use oxidaxa::matching::{parallel_match, parallel_match_inverted};
use oxidaxa::rng::RRng;
use oxidaxa::training::learn_taxa;
use oxidaxa::types::TrainConfig;

fn main() {
    let sizes = [1000, 5000, 10000];

    for &size in &sizes {
        let fasta = format!("benchmarks/data/bench_{}_ref.fasta", size);
        let tax = format!("benchmarks/data/bench_{}_ref_taxonomy.tsv", size);
        let query_fasta = format!("benchmarks/data/bench_{}_query.fasta", size);

        let (names, seqs) = match read_fasta(&fasta) {
            Ok(r) => r,
            Err(_) => {
                println!("Skipping {}: file not found", size);
                continue;
            }
        };
        let taxonomy = read_taxonomy(&tax, &names).unwrap();

        let mut rng = RRng::new(42);
        let config = TrainConfig::default();
        let ts = learn_taxa(&seqs, &taxonomy, &config, &mut rng, false).unwrap();

        let inv_idx = ts.inverted_index.as_ref().unwrap();

        // Enumerate one query sequence
        let (_, query_seqs) = read_fasta(&query_fasta).unwrap();
        let raw = enumerate_sequences(&query_seqs[..1], ts.k, false, false, &[], true, None);
        let mut query_kmers: Vec<i32> = raw[0]
            .iter()
            .filter(|&&x| x != NA_INTEGER)
            .map(|&x| x + 1)
            .collect();
        query_kmers.sort_unstable();
        query_kmers.dedup();

        let b = 100usize;
        let s = (query_kmers.len() as f64).powf(0.47).ceil() as usize;

        // Build sampling data
        let mut rng2 = RRng::new(42);
        let sampling: Vec<i32> = rng2.sample_replace(&query_kmers, s * b);
        let sb = s * b;
        let mut sort_idx: Vec<u32> = (0..sb as u32).collect();
        sort_idx.sort_unstable_by_key(|&i| sampling[i as usize]);

        let mut u_sampling: Vec<i32> = Vec::new();
        let mut positions: Vec<usize> = Vec::with_capacity(sb);
        let mut ranges: Vec<usize> = vec![0];
        let mut i = 0;
        while i < sb {
            let kmer = sampling[sort_idx[i] as usize];
            u_sampling.push(kmer);
            while i < sb && sampling[sort_idx[i] as usize] == kmer {
                positions.push(sort_idx[i] as usize % b);
                i += 1;
            }
            ranges.push(positions.len());
        }

        let u_weights: Vec<f64> = u_sampling
            .iter()
            .map(|&uk| {
                if uk > 0 && (uk as usize) <= ts.idf_weights.len() {
                    ts.idf_weights[(uk - 1) as usize]
                } else {
                    0.0
                }
            })
            .collect();

        println!(
            "=== {}K refs, {} unique query k-mers, {} unique sampled ===",
            size / 1000,
            query_kmers.len(),
            u_sampling.len()
        );

        // Test at different keep sizes
        let keep_sizes: Vec<usize> = vec![
            10,
            50,
            500,
            ts.kmers.len().min(2000),
            ts.kmers.len(),
        ];

        for &keep_n in &keep_sizes {
            let keep: Vec<usize> = (0..keep_n.min(ts.kmers.len())).collect();
            let actual_keep = keep.len();
            let n_iters = 20;

            let start = Instant::now();
            for _ in 0..n_iters {
                let _ = parallel_match(
                    &u_sampling, &ts.kmers, &keep, &u_weights, b, &positions, &ranges,
                );
            }
            let merge_time = start.elapsed().as_secs_f64() / n_iters as f64;

            let start = Instant::now();
            for _ in 0..n_iters {
                let _ = parallel_match_inverted(
                    &u_sampling, inv_idx, &keep, &u_weights, b, &positions, &ranges,
                );
            }
            let inv_time = start.elapsed().as_secs_f64() / n_iters as f64;

            let speedup = merge_time / inv_time;
            let winner = if speedup > 1.0 { "inverted wins" } else { "merge-join wins" };
            println!(
                "  keep={:>6}: merge={:.3}ms  inverted={:.3}ms  speedup={:.1}x  ({})",
                actual_keep,
                merge_time * 1000.0,
                inv_time * 1000.0,
                speedup,
                winner
            );
        }
        println!();
    }
}
