import json
from pathlib import Path

import pytest

GOLDEN_JSON_DIR = Path(__file__).parent.parent / "tests" / "golden_json"


@pytest.fixture
def golden_dir():
    return GOLDEN_JSON_DIR


def load_golden(name: str):
    path = GOLDEN_JSON_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)
