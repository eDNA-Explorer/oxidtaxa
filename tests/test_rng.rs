mod common;

use common::load_json;
use oxidtaxa::rng::RRng;

#[test]
fn test_prng_matches_r_unif_rand() {
    let golden: Vec<f64> = load_json("prng_seed42_100draws");
    let mut rng = RRng::new(42);
    for (i, expected) in golden.iter().enumerate() {
        let got = rng.unif_rand();
        assert!(
            (got - expected).abs() < 1e-15,
            "PRNG mismatch at draw {}: got {} expected {} (diff: {})",
            i,
            got,
            expected,
            (got - expected).abs()
        );
    }
}

#[test]
fn test_prng_matches_r_sample_10from50() {
    let golden: Vec<i32> = load_json("prng_sample_10from50");
    let mut rng = RRng::new(42);
    let got = rng.sample_int_replace(50, 10);
    for (i, &expected) in golden.iter().enumerate() {
        // R returns 1-indexed, our function returns 0-indexed
        assert_eq!(
            got[i],
            (expected - 1) as usize,
            "sample mismatch at index {}: got {} expected {}",
            i,
            got[i],
            expected - 1
        );
    }
}

#[test]
fn test_prng_matches_r_sample_100from1000() {
    let golden: Vec<i32> = load_json("prng_sample_100from1000");
    let mut rng = RRng::new(42);
    let got = rng.sample_int_replace(1000, 100);
    for (i, &expected) in golden.iter().enumerate() {
        assert_eq!(
            got[i],
            (expected - 1) as usize,
            "sample mismatch at index {}: got {} expected {}",
            i,
            got[i],
            expected - 1
        );
    }
}

#[test]
fn test_prng_sample_edge_cases() {
    // sample(1, 1, replace=TRUE) should always return 0 (0-indexed)
    // Golden may be auto-unboxed to bare int or wrapped in array
    let path = common::golden_json_dir().join("prng_sample_1from1.json");
    let content = std::fs::read_to_string(&path).unwrap();
    let golden_val: i32 = match serde_json::from_str::<Vec<i32>>(&content) {
        Ok(v) => v[0],
        Err(_) => serde_json::from_str(&content).unwrap(),
    };
    let mut rng = RRng::new(42);
    let got = rng.sample_int_replace(1, 1);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0], (golden_val - 1) as usize);

    // sample(5, 0, replace=TRUE) should return empty
    let mut rng2 = RRng::new(42);
    let got_empty = rng2.sample_int_replace(5, 0);
    assert_eq!(got_empty.len(), 0);
}
