#!/bin/bash
cd /Users/ryanmartin/idtaxa-optim
cargo test --manifest-path rust/Cargo.toml --release 2>&1
