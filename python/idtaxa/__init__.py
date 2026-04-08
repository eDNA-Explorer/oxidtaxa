"""IDTAXA: Taxonomic classification of DNA sequences.

All computation is performed in Rust. This Python package provides
a thin CLI interface.
"""

from oxidaxa._core import ClassificationResult, classify, train

__all__ = ["train", "classify", "ClassificationResult"]
