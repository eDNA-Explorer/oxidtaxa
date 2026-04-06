use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::path::PathBuf;

use idtaxa::kmer::enumerate_sequences;
use idtaxa::matching::{int_match, vector_sum, parallel_match};
use idtaxa::rng::RRng;
use idtaxa::sequence::{remove_gaps, reverse_complement};
use idtaxa::fasta::read_fasta;
use idtaxa::training::learn_taxa;
use idtaxa::classify::id_taxa;
use idtaxa::types::{TrainConfig, ClassifyConfig, StrandMode, OutputType};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn bench_data_path(name: &str) -> String {
    project_root().join("benchmarks").join("data").join(name)
        .to_string_lossy().to_string()
}

fn test_data_path(name: &str) -> String {
    project_root().join("tests").join("data").join(name)
        .to_string_lossy().to_string()
}

// ── enumerate_sequences ──────────────────────────────────────────────────────

fn bench_enumerate_sequences(c: &mut Criterion) {
    let (_, seqs) = read_fasta(&bench_data_path("bench_1000_ref.fasta")).unwrap();
    let k = 8;

    c.bench_function("enumerate_sequences/1K_k8", |b| {
        b.iter(|| {
            black_box(enumerate_sequences(black_box(&seqs), k, false, false, &[], true));
        });
    });
}

fn bench_enumerate_single(c: &mut Criterion) {
    // Use a single ~200bp sequence
    let (_, seqs) = read_fasta(&test_data_path("test_ref.fasta")).unwrap();
    let seq = vec![seqs[0].clone()];
    let k = 8;

    c.bench_function("enumerate_single/200bp_k8", |b| {
        b.iter(|| {
            black_box(enumerate_sequences(black_box(&seq), k, false, false, &[], true));
        });
    });
}

// ── int_match ──────────────────────────────────────��─────────────────────────

fn bench_int_match(c: &mut Criterion) {
    // Simulate realistic sorted k-mer vectors
    let x: Vec<i32> = (0..500).map(|i| i * 3 + 1).collect();
    let y: Vec<i32> = (0..200).map(|i| i * 5 + 2).collect();

    c.bench_function("int_match/500v200", |b| {
        b.iter(|| {
            black_box(int_match(black_box(&x), black_box(&y)));
        });
    });
}

// ── vector_sum ───────────────────────────────────────────────────────────────

fn bench_vector_sum(c: &mut Criterion) {
    let n = 100;
    let b_count = 50;
    let s = 20;
    let matches: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let weights: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.01).collect();
    let mut rng = RRng::new(42);
    let sampling = rng.sample_int_replace(n, s * b_count);

    c.bench_function("vector_sum/100k_50b", |b| {
        b.iter(|| {
            black_box(vector_sum(
                black_box(&matches),
                black_box(&weights),
                black_box(&sampling),
                black_box(b_count),
            ));
        });
    });
}

// ���─ parallel_match ─────────��─────────────────────────────────────────────────

fn bench_parallel_match(c: &mut Criterion) {
    // Build realistic data: train a model, then use its k-mers
    let (_, seqs) = read_fasta(&test_data_path("test_ref.fasta")).unwrap();
    let k = 8;
    let raw = enumerate_sequences(&seqs, k, false, false, &[], true);
    let train_kmers: Vec<Vec<i32>> = raw.into_iter().map(|v| {
        let mut sorted: Vec<i32> = v.into_iter()
            .filter(|&x| x != i32::MIN).map(|x| x + 1).collect();
        sorted.sort_unstable();
        sorted.dedup();
        sorted
    }).collect();

    // Query k-mers (first 5 sequences as queries)
    let query_km = &train_kmers[0];
    let b_count = 50;
    let s = 15;
    let mut rng = RRng::new(42);
    let sampling: Vec<i32> = rng.sample_replace(query_km, s * b_count);
    let mut u_sampling: Vec<i32> = sampling.clone();
    u_sampling.sort_unstable();
    u_sampling.dedup();

    let mut grouped: Vec<Vec<usize>> = vec![Vec::new(); u_sampling.len()];
    for (idx, &sk) in sampling.iter().enumerate() {
        if let Ok(pos) = u_sampling.binary_search(&sk) {
            grouped[pos].push(idx % b_count);
        }
    }
    let mut positions: Vec<usize> = Vec::new();
    let mut ranges: Vec<usize> = vec![0];
    for group in &grouped {
        positions.extend(group);
        ranges.push(positions.len());
    }

    // Use IDF weights of 1.0 for simplicity
    let u_weights: Vec<f64> = vec![1.0; u_sampling.len()];
    let indices: Vec<usize> = (0..train_kmers.len()).collect();

    c.bench_function("parallel_match/80seqs_50b", |b| {
        b.iter(|| {
            black_box(parallel_match(
                black_box(&u_sampling),
                black_box(&train_kmers),
                black_box(&indices),
                black_box(&u_weights),
                black_box(b_count),
                black_box(&positions),
                black_box(&ranges),
            ));
        });
    });
}

// ── sample_int_replace ──────────────���────────────────────────────────────────

fn bench_sample_int_replace(c: &mut Criterion) {
    c.bench_function("sample_int_replace/10K_from_1000", |b| {
        b.iter(|| {
            let mut rng = RRng::new(42);
            black_box(rng.sample_int_replace(black_box(1000), black_box(10000)));
        });
    });
}

// ── reverse_complement ───────────────────────────────────────────────────────

fn bench_reverse_complement(c: &mut Criterion) {
    let (_, seqs) = read_fasta(&test_data_path("test_ref.fasta")).unwrap();

    c.bench_function("reverse_complement/80seqs", |b| {
        b.iter(|| {
            for seq in &seqs {
                black_box(reverse_complement(black_box(seq)));
            }
        });
    });
}

// ── remove_gaps ─────────��────────────────────────��───────────────────────────

fn bench_remove_gaps(c: &mut Criterion) {
    let (_, seqs) = read_fasta(&test_data_path("test_ref.fasta")).unwrap();
    // Add some gaps for realism
    let gapped: Vec<String> = seqs.iter().map(|s| {
        let mut g = String::with_capacity(s.len() + 20);
        for (i, c) in s.chars().enumerate() {
            if i % 10 == 0 { g.push('-'); }
            g.push(c);
        }
        g
    }).collect();

    c.bench_function("remove_gaps/80seqs", |b| {
        b.iter(|| {
            black_box(remove_gaps(black_box(&gapped)));
        });
    });
}

// ── read_fasta ─────��─────────────────────────────────────────────────────────

fn bench_read_fasta(c: &mut Criterion) {
    let path = bench_data_path("bench_1000_ref.fasta");

    c.bench_function("read_fasta/1K", |b| {
        b.iter(|| {
            black_box(read_fasta(black_box(&path)).unwrap());
        });
    });
}

// ── learn_taxa (full training) ───────────────────────────────────────────────

fn bench_learn_taxa(c: &mut Criterion) {
    let (names, seqs) = read_fasta(&test_data_path("test_ref.fasta")).unwrap();
    let taxonomy = idtaxa::fasta::read_taxonomy(
        &test_data_path("test_ref_taxonomy.tsv"), &names
    ).unwrap();

    // Apply same filtering as lib.rs
    let mut filtered_seqs = Vec::new();
    let mut filtered_tax = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        let tax = &taxonomy[i];
        let full_tax = format!("Root; {}", tax.replace(";", "; "));
        let rank_count = full_tax.split("; ").count();
        if rank_count < 4 { continue; }
        if seq.len() < 30 { continue; }
        let n_count = seq.bytes().filter(|&b| b == b'N' || b == b'n').count();
        if (n_count as f64 / seq.len() as f64) > 0.3 { continue; }
        filtered_seqs.push(seq.clone());
        filtered_tax.push(full_tax);
    }

    let config = TrainConfig::default();

    c.bench_function("learn_taxa/80seqs", |b| {
        b.iter(|| {
            let mut rng = RRng::new(42);
            black_box(learn_taxa(
                black_box(&filtered_seqs),
                black_box(&filtered_tax),
                black_box(&config),
                &mut rng,
                false,
            ).unwrap());
        });
    });
}

// ── id_taxa (classification) ───────────────��─────────────────────────────────

fn bench_id_taxa(c: &mut Criterion) {
    // Train a model first
    let (names, seqs) = read_fasta(&test_data_path("test_ref.fasta")).unwrap();
    let taxonomy = idtaxa::fasta::read_taxonomy(
        &test_data_path("test_ref_taxonomy.tsv"), &names
    ).unwrap();

    let mut filtered_seqs = Vec::new();
    let mut filtered_tax = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        let tax = &taxonomy[i];
        let full_tax = format!("Root; {}", tax.replace(";", "; "));
        let rank_count = full_tax.split("; ").count();
        if rank_count < 4 { continue; }
        if seq.len() < 30 { continue; }
        let n_count = seq.bytes().filter(|&b| b == b'N' || b == b'n').count();
        if (n_count as f64 / seq.len() as f64) > 0.3 { continue; }
        filtered_seqs.push(seq.clone());
        filtered_tax.push(full_tax);
    }

    let config = TrainConfig::default();
    let mut rng = RRng::new(42);
    let model = learn_taxa(&filtered_seqs, &filtered_tax, &config, &mut rng, false).unwrap();

    // Load query sequences
    let (query_names, query_seqs) = read_fasta(&test_data_path("test_query.fasta")).unwrap();
    let clean_seqs = idtaxa::sequence::remove_gaps(&query_seqs);

    let classify_config = ClassifyConfig {
        threshold: 40.0,
        min_descend: 0.98,
        full_length: 0.0,
        processors: 1,
        ..Default::default()
    };

    c.bench_function("id_taxa_sequential/15q_80ref", |b| {
        b.iter(|| {
            black_box(id_taxa(
                black_box(&clean_seqs),
                black_box(&query_names),
                black_box(&model),
                black_box(&classify_config),
                StrandMode::Both,
                OutputType::Extended,
                42,
                true, // deterministic for benchmarks
            ));
        });
    });
}

// ── Scaling benchmarks with different dataset sizes ──────────────────────────

fn bench_enumerate_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("enumerate_scaling");

    for size in &["bench_1000_ref.fasta", "bench_5000_ref.fasta", "bench_10000_ref.fasta"] {
        let path = bench_data_path(size);
        if std::path::Path::new(&path).exists() {
            let (_, seqs) = read_fasta(&path).unwrap();
            let label = size.replace("_ref.fasta", "");
            group.bench_with_input(
                BenchmarkId::new("enumerate", &label),
                &seqs,
                |b, seqs| {
                    b.iter(|| {
                        black_box(enumerate_sequences(black_box(seqs), 8, false, false, &[], true));
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_enumerate_sequences,
    bench_enumerate_single,
    bench_int_match,
    bench_vector_sum,
    bench_parallel_match,
    bench_sample_int_replace,
    bench_reverse_complement,
    bench_remove_gaps,
    bench_read_fasta,
    bench_learn_taxa,
    bench_id_taxa,
    bench_enumerate_scaling,
);
criterion_main!(benches);
