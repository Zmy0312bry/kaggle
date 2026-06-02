from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_species_list(data_dir: str | Path, sample_submission: str | Path | None = None) -> list[str]:
    data_dir = Path(data_dir)
    if sample_submission is not None and Path(sample_submission).exists():
        sample = pd.read_csv(sample_submission, nrows=1)
        return [c for c in sample.columns if c != "row_id"]
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path, nrows=1)
        return [c for c in sample.columns if c != "row_id"]
    taxonomy_path = data_dir / "taxonomy.csv"
    taxonomy = pd.read_csv(taxonomy_path)
    if "primary_label" in taxonomy.columns:
        return taxonomy["primary_label"].astype(str).tolist()
    first_col = taxonomy.columns[0]
    return taxonomy[first_col].astype(str).tolist()


def parse_label_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v) for v in parsed if str(v)]
    except (SyntaxError, ValueError):
        pass
    return [token for token in text.replace(",", " ").replace(";", " ").split() if token]


def multi_hot(labels: Iterable[str], species_to_idx: dict[str, int], value: float = 1.0) -> np.ndarray:
    target = np.zeros(len(species_to_idx), dtype=np.float32)
    for label in labels:
        idx = species_to_idx.get(str(label))
        if idx is not None:
            target[idx] = max(target[idx], value)
    return target


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

