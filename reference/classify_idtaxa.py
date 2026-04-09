#!/usr/bin/env python3
"""Classify sequences using a trained Oxidtaxa model.

Usage:
    python classify_idtaxa.py <query.fasta> <model> <output.tsv> \
        <threshold> <strand> <min_descend> <full_length> <processors>
"""
import argparse

from oxidtaxa import classify


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify with Oxidtaxa")
    parser.add_argument("query_fasta")
    parser.add_argument("model_path")
    parser.add_argument("output_tsv")
    parser.add_argument("threshold", type=float)
    parser.add_argument("strand", choices=["top", "bottom", "both"])
    parser.add_argument("min_descend", type=float)
    parser.add_argument("full_length", type=float)
    parser.add_argument("processors", type=int)
    parser.add_argument("--bootstraps", type=int, default=100,
                        help="Bootstrap replicates (default: 100, use 50 for sweeps)")
    parser.add_argument("--sample-exponent", type=float, default=0.47,
                        help="Exponent for k-mers per bootstrap: S = L^x (default: 0.47)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Sequential mode for reproducibility",
    )
    args = parser.parse_args()

    classify(
        query_path=args.query_fasta,
        model_path=args.model_path,
        output_path=args.output_tsv,
        threshold=args.threshold,
        bootstraps=args.bootstraps,
        strand=args.strand,
        min_descend=args.min_descend,
        full_length=args.full_length,
        processors=args.processors,
        sample_exponent=args.sample_exponent,
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
