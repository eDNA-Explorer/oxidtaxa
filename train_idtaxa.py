#!/usr/bin/env python3
"""Train an Oxidaxa classifier from CruxV2 reference files.

Usage:
    python train_idtaxa.py <reference.fasta> <taxonomy.tsv> <output.model>
"""
import argparse

from idtaxa import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Oxidaxa classifier")
    parser.add_argument("reference_fasta", help="CruxV2 reference FASTA")
    parser.add_argument("taxonomy_tsv", help="Tab-separated: accession<TAB>taxonomy")
    parser.add_argument("output_model", help="Output model file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--k", type=int, default=None, help="K-mer size (default: auto)")
    parser.add_argument("--record-kmers-fraction", type=float, default=0.10,
                        help="Fraction of top k-mers retained per decision node (default: 0.10)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    train(
        fasta_path=args.reference_fasta,
        taxonomy_path=args.taxonomy_tsv,
        output_path=args.output_model,
        seed=args.seed,
        k=args.k,
        record_kmers_fraction=args.record_kmers_fraction,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
