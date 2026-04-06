"""Oxidaxa: High-performance taxonomic classification for eDNA metabarcoding.

Rust-accelerated IDTAXA implementation with per-query independent PRNG,
inverted k-mer index, spaced seed support, and configurable scoring.
"""

from oxidaxa._core import classify, train

__all__ = ["train", "classify"]
