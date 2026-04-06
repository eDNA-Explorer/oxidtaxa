## Rust Development

The Rust code at `rust/` is a Cargo workspace with one crate: `fasta-gen` (deduplicated FASTA generation with Parquet output and GCS integration). CI runs via `.github/workflows/rust.yml`.

### CI Workflow

The following checks run on all PRs touching `rust/**`:
- **Format**: `cargo fmt --check`
- **Lint**: `cargo clippy -D warnings` (all warnings are errors)
- **Test**: `cargo test`
- **Security**: `cargo deny check`
- **MSRV**: Verified against Rust 1.83
- **Coverage**: Generated with cargo-tarpaulin (artifact only)

### Local Development Commands

```bash
# Format code
cargo fmt --manifest-path rust/Cargo.toml

# Run clippy (with warnings as errors)
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings

# Run tests
cargo test --manifest-path rust/Cargo.toml

# Security audit (requires cargo-deny: cargo install cargo-deny)
cargo deny --manifest-path rust/Cargo.toml check
```

### Key Conventions

- All fallible functions return `anyhow::Result<T>` — use `.context()` to add details when propagating errors
- Zero `unsafe` code
- `FxHashMap` (from `rustc_hash`) for performance-critical paths; standard `HashMap` elsewhere
- `rayon` for parallelism, `parking_lot::Mutex` over `std::sync::Mutex`
- Unit tests inline (`#[cfg(test)] mod tests`), integration tests in `tests/integration_tests.rs`

### Configuration Files

- `rust/rustfmt.toml` — Formatting rules (`max_width = 100`, Unix line endings)
- `rust/clippy.toml` — Linter thresholds
- `rust/deny.toml` — Security and license policy
