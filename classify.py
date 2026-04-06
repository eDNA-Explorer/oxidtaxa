#!/usr/bin/env python3
"""Classify sequences using a trained Oxidaxa model.

Usage:
    python classify.py query.fasta model.bin results.tsv 40 both 0.98 0.0 8
"""
import argparse

from oxidaxa import classify


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify with Oxidaxa")
    parser.add_argument("query_fasta")
    parser.add_argument("model_path")
    parser.add_argument("output_tsv")
    parser.add_argument("threshold", type=float)
    parser.add_argument("strand", choices=["top", "bottom", "both"])
    parser.add_argument("min_descend", type=float)
    parser.add_argument("full_length", type=float)
    parser.add_argument("processors", type=int)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--sample-exponent", type=float, default=0.47)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--length-normalize", action="store_true",
                        help="Normalize scores by training sequence length")
    parser.add_argument("--rank-thresholds", type=str, default=None,
                        help="Comma-separated per-rank thresholds (e.g., 90,80,70,60,50,40,40)")
    args = parser.parse_args()

    rank_thresholds = None
    if args.rank_thresholds:
        rank_thresholds = [float(x) for x in args.rank_thresholds.split(",")]

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
        length_normalize=args.length_normalize,
        rank_thresholds=rank_thresholds,
    )


if __name__ == "__main__":
    main()
