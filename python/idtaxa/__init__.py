"""IDTAXA: Taxonomic classification of DNA sequences.

All computation is performed in Rust. This Python package provides
a thin CLI interface.
"""

from idtaxa._core import classify, train

__all__ = ["train", "classify"]
