use std::path::PathBuf;

/// Get the path to the golden_json directory.
pub fn golden_json_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.parent().unwrap().join("tests").join("golden_json")
}

/// Load a JSON golden file and deserialize.
pub fn load_json<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = golden_json_dir().join(format!("{}.json", name));
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Cannot parse {}: {}", path.display(), e))
}

/// Assert two f64 values are within epsilon.
#[allow(dead_code)]
pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64, msg: &str) {
    assert!(
        (a - b).abs() < epsilon,
        "{}: {} vs {} (diff: {})",
        msg,
        a,
        b,
        (a - b).abs()
    );
}

/// Assert two f64 slices are element-wise within epsilon.
#[allow(dead_code)]
pub fn assert_vec_approx_eq(a: &[f64], b: &[f64], epsilon: f64, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", msg, a.len(), b.len());
    for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (ai - bi).abs() < epsilon,
            "{}[{}]: {} vs {} (diff: {})",
            msg,
            i,
            ai,
            bi,
            (ai - bi).abs()
        );
    }
}
