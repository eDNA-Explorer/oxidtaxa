"""Golden tests: compare Python/Rust output against R/C baselines."""
import json
import os
from pathlib import Path

import pytest

from conftest import load_golden

DATA_DIR = Path(__file__).parent.parent / "tests" / "data"
GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden_json"


class TestImport:
    def test_import_train(self):
        from oxidtaxa import train
        assert callable(train)

    def test_import_classify(self):
        from oxidtaxa import classify
        assert callable(classify)


class TestFullPipeline:
    """End-to-end pipeline tests matching Section 10 of run_golden.R."""

    def test_train_and_classify(self, tmp_path):
        from oxidtaxa import classify, train

        model_path = str(tmp_path / "model.bin")
        output_path = str(tmp_path / "output.tsv")

        train(
            fasta_path=str(DATA_DIR / "test_ref.fasta"),
            taxonomy_path=str(DATA_DIR / "test_ref_taxonomy.tsv"),
            output_path=model_path,
            seed=42,
        )

        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0

        classify(
            query_path=str(DATA_DIR / "test_query.fasta"),
            model_path=model_path,
            output_path=output_path,
            threshold=60.0,
            bootstraps=100,
            strand="both",
            min_descend=0.98,
            full_length=0.0,
            processors=1,
            seed=42,
            deterministic=True,
        )

        assert os.path.exists(output_path)

        with open(output_path) as f:
            lines = f.readlines()

        header = lines[0].strip().split("\t")
        assert header == ["read_id", "taxonomic_path", "confidence"]

        golden = load_golden("s10a_e2e_tsv")

        for i, line in enumerate(lines[1:]):
            parts = line.strip().split("\t")
            assert parts[0] == golden[i]["read_id"], f"read_id mismatch at row {i}"
            assert parts[1] == golden[i]["taxonomic_path"], f"taxonomic_path mismatch at row {i}"
            assert abs(float(parts[2]) - golden[i]["confidence"]) < 10.0, (
                f"confidence mismatch at row {i}: {parts[2]} vs {golden[i]['confidence']}"
            )

    def test_train_only(self, tmp_path):
        from oxidtaxa import train

        model_path = str(tmp_path / "model.bin")
        train(
            fasta_path=str(DATA_DIR / "test_ref.fasta"),
            taxonomy_path=str(DATA_DIR / "test_ref_taxonomy.tsv"),
            output_path=model_path,
            seed=42,
        )
        assert os.path.getsize(model_path) > 0

    def test_classify_strand_top(self, tmp_path):
        from oxidtaxa import classify, train

        model_path = str(tmp_path / "model.bin")
        output_path = str(tmp_path / "output.tsv")

        train(
            fasta_path=str(DATA_DIR / "test_ref.fasta"),
            taxonomy_path=str(DATA_DIR / "test_ref_taxonomy.tsv"),
            output_path=model_path,
            seed=42,
        )

        classify(
            query_path=str(DATA_DIR / "test_query.fasta"),
            model_path=model_path,
            output_path=output_path,
            threshold=60.0,
            bootstraps=100,
            strand="top",
            min_descend=0.98,
            full_length=0.0,
            processors=1,
            seed=42,
        )

        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) > 1  # header + at least 1 result
