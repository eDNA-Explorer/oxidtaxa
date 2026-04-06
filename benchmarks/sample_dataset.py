#!/usr/bin/env python3
"""Sample sequences from the vert12S LCA dataset for benchmarking."""
import random
import sys
from pathlib import Path

SRC_FASTA = Path.home() / "rcrux-py/databases/vert12s/unfiltered/vert12S_lca.fasta"
SRC_TAX = Path.home() / "rcrux-py/databases/vert12s/unfiltered/vert12S_lca_taxonomy.txt"
OUT_DIR = Path("benchmarks/data")

def read_fasta(path):
    """Read FASTA, return list of (header, sequence)."""
    records = []
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records

def read_taxonomy(path):
    """Read taxonomy TSV, return dict accession -> taxonomy."""
    tax = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                tax[parts[0]] = parts[1]
    return tax

def extract_accession(header):
    """Extract the accession from a FASTA header to match taxonomy keys."""
    # Header format: "gi|...|gb|AF014587.1|AF014587" or similar
    # Taxonomy key is the full first field before tab
    return header.split()[0]

def main():
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading FASTA...")
    records = read_fasta(SRC_FASTA)
    print(f"  {len(records)} total sequences")

    print("Reading taxonomy...")
    tax_map = read_taxonomy(SRC_TAX)
    print(f"  {len(tax_map)} taxonomy entries")

    # Match FASTA records to taxonomy
    matched = []
    for header, seq in records:
        acc = extract_accession(header)
        if acc in tax_map:
            t = tax_map[acc]
            # Filter: need >= 4 ranks, no NA-only paths, reasonable length
            ranks = [r for r in t.split(";") if r.strip() and r.strip() != "NA"]
            if len(ranks) >= 4 and len(seq) >= 30:
                n_count = seq.upper().count("N")
                if n_count / len(seq) <= 0.3:
                    matched.append((acc, header, seq, t))

    print(f"  {len(matched)} matched with >= 4 ranks, >= 30bp, <= 30% N")

    # Shuffle and sample
    random.shuffle(matched)

    for n_ref in [1000, 5000, 10000]:
        if n_ref > len(matched):
            print(f"  Skipping {n_ref}: not enough matched sequences")
            continue

        ref_set = matched[:n_ref]
        # Use 500 query sequences from outside the reference set (or tail)
        n_query = min(500, len(matched) - n_ref)
        query_set = matched[n_ref:n_ref + n_query]

        prefix = f"bench_{n_ref}"
        fasta_path = OUT_DIR / f"{prefix}_ref.fasta"
        tax_path = OUT_DIR / f"{prefix}_ref_taxonomy.tsv"
        query_path = OUT_DIR / f"{prefix}_query.fasta"

        with open(fasta_path, "w") as f:
            for acc, header, seq, _ in ref_set:
                f.write(f">{acc}\n{seq}\n")

        with open(tax_path, "w") as f:
            for acc, _, _, t in ref_set:
                f.write(f"{acc}\t{t}\n")

        with open(query_path, "w") as f:
            for acc, header, seq, _ in query_set:
                f.write(f">{acc}\n{seq}\n")

        print(f"  {prefix}: {n_ref} ref, {n_query} query -> {fasta_path}")

    print("Done.")

if __name__ == "__main__":
    main()
