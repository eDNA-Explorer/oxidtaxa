#!/usr/bin/env python3
"""Train an Oxidtaxa classifier from reference sequences.

Usage:
    python train.py reference.fasta taxonomy.tsv model.bin
"""
import argparse

from oxidtaxa import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Oxidtaxa classifier")
    parser.add_argument("reference_fasta", help="Reference FASTA")
    parser.add_argument("taxonomy_tsv", help="Tab-separated: accession<TAB>taxonomy")
    parser.add_argument("output_model", help="Output model file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=None, help="K-mer size (default: auto)")
    parser.add_argument("--record-kmers-fraction", type=float, default=0.10,
                        help="Fraction of top k-mers per decision node (default: 0.10)")
    parser.add_argument("--seed-pattern", type=str, default=None,
                        help="Spaced seed pattern (e.g., '11011011011'). Default: contiguous")
    args = parser.parse_args()

    train(
        fasta_path=args.reference_fasta,
        taxonomy_path=args.taxonomy_tsv,
        output_path=args.output_model,
        seed=args.seed,
        k=args.k,
        record_kmers_fraction=args.record_kmers_fraction,
        seed_pattern=args.seed_pattern,
    )


if __name__ == "__main__":
    main()
