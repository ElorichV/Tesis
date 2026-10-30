"""Bootstrap del clasificador estático con landmarks públicos de MediaPipe."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from features.landmarks import STATIC_FEATURE_DIM, normalize_hand_landmarks


def load_hf_asl(cache_dir: Path) -> dict[str, list[np.ndarray]]:
    """Carga el dataset público ASL de forma diferida para evitar scipy al importar."""
    from avatar.train import load_hf_asl as load_dataset

    return load_dataset(cache_dir)


def build_bootstrap_rows(
    samples: dict[str, list[np.ndarray]],
    static_letters: list[str],
    canonical_hand: str = "Right",
) -> tuple[np.ndarray, list[str]]:
    """Normaliza las letras estáticas permitidas y añade su versión espejada en X."""
    allowed = {str(letter).strip().upper() for letter in static_letters}
    rows: list[np.ndarray] = []
    labels: list[str] = []

    for raw_label, landmarks_list in samples.items():
        label = str(raw_label).strip().upper()
        if label not in allowed:
            continue
        for landmarks in landmarks_list:
            normalized = normalize_hand_landmarks(
                landmarks,
                handedness="Right",
                canonical_hand=canonical_hand,
            )
            mirrored = normalized.reshape(-1, 3).copy()
            mirrored[:, 0] *= -1
            rows.extend((normalized, mirrored.reshape(-1)))
            labels.extend((label, label))

    return np.asarray(rows, dtype=np.float32).reshape(-1, STATIC_FEATURE_DIM), labels


def ensure_bootstrap_static_csv(
    cfg: dict,
    spec: dict,
    out_path: str | Path = "data/bootstrap_static.csv",
    cache_dir: str | Path = "data/external/asl_static_landmarks_v1",
    loader: Callable[[Path], dict[str, list[np.ndarray]]] | None = None,
) -> Path:
    """Crea el CSV público estático una vez y conserva las filas ya existentes."""
    output = Path(out_path)
    if output.exists():
        return output

    samples = (loader or load_hf_asl)(Path(cache_dir))
    rows, labels = build_bootstrap_rows(
        samples,
        static_letters=spec["static"],
        canonical_hand=cfg.get("features", {}).get("canonical_hand", "Right"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label", *(f"f_{idx}" for idx in range(STATIC_FEATURE_DIM))])
        for label, row in zip(labels, rows):
            writer.writerow([label, *(f"{float(value):.6f}" for value in row)])
    return output
