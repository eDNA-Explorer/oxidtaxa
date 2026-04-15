"""Oxidtaxa: High-performance taxonomic classification for eDNA metabarcoding.

Rust-accelerated IDTAXA implementation with per-query independent PRNG,
inverted k-mer index, spaced seed support, and configurable scoring.
"""

from oxidtaxa._core import (
    BuiltTree,
    ClassificationResult,
    PreparedData,
    build_tree_py as build_tree,
    classify,
    learn_fractions_py as learn_fractions,
    prepare_data_py as prepare_data,
    train,
)

__all__ = [
    "train",
    "classify",
    "ClassificationResult",
    "prepare_data",
    "build_tree",
    "learn_fractions",
    "PreparedData",
    "BuiltTree",
]
