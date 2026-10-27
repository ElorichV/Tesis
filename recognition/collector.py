"""Herramienta de almacenamiento en CSV para captura de landmarks estáticos y dinámicos."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from features.landmarks import (
    DYNAMIC_FEATURE_DIM,
    STATIC_FEATURE_DIM,
    extract_dynamic_features,
    normalize_hand_landmarks,
)


def get_collection_guidance(current_idx: int, total_target: int, is_dynamic: bool = False) -> str:
    """
    Devuelve avisos en pantalla según el progreso de captura para guiar al usuario
    a variar ángulo, distancia e iluminación.
    """
    if is_dynamic:
        return "Realiza el trazo completo de la letra con movimiento fluido frente a la cámara."

    ratio = current_idx / max(1, total_target)
    if ratio < 0.34:
        return "Fase 1/3: Mantén la mano de frente a distancia media (ángulo base)."
    if ratio < 0.67:
        return "Fase 2/3: Inclina/rota ligeramente la mano (±15°) para variar el ángulo."
    return "Fase 3/3: Acerca/aleja la mano (~15 cm) y varía ligeramente la posición/luz."


class SampleCollector:
    """Gestiona el guardado incremental de vectores de landmarks en archivos CSV."""

    def __init__(
        self,
        static_csv_path: str | Path = "data/landmarks_static.csv",
        dynamic_csv_path: str | Path = "data/landmarks_dynamic.csv",
        canonical_hand: str = "Right",
    ) -> None:
        self.static_csv_path = Path(static_csv_path)
        self.dynamic_csv_path = Path(dynamic_csv_path)
        self.canonical_hand = canonical_hand

        self.static_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.dynamic_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def save_static_sample(
        self,
        label: str,
        landmarks: Sequence[Sequence[float]] | np.ndarray,
        handedness: str = "Right",
    ) -> np.ndarray:
        """Normaliza los 21 landmarks de una mano y añade una fila al CSV estático."""
        feat = normalize_hand_landmarks(
            landmarks,
            handedness=handedness,
            canonical_hand=self.canonical_hand,
        )
        self._append_row(self.static_csv_path, label.strip().upper(), feat, STATIC_FEATURE_DIM, "f")
        return feat

    def save_dynamic_sequence(
        self,
        label: str,
        frames_landmarks: Sequence[Sequence[Sequence[float]] | np.ndarray],
        handedness: str = "Right",
    ) -> np.ndarray:
        """Extrae características de una ventana temporal y añade una fila al CSV dinámico."""
        feat = extract_dynamic_features(
            frames_landmarks,
            handedness=handedness,
            canonical_hand=self.canonical_hand,
        )
        self._append_row(
            self.dynamic_csv_path, label.strip().upper(), feat, DYNAMIC_FEATURE_DIM, "d"
        )
        return feat

    @staticmethod
    def _append_row(
        csv_path: Path,
        label: str,
        features: np.ndarray,
        expected_dim: int,
        prefix: str,
    ) -> None:
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if not file_exists:
                header = ["label"] + [f"{prefix}_{i}" for i in range(expected_dim)]
                writer.writerow(header)
            row = [label] + [f"{float(v):.6f}" for v in features]
            writer.writerow(row)

    def count_samples_by_label(self, is_dynamic: bool = False) -> dict[str, int]:
        """Devuelve el conteo de muestras almacenadas por cada letra."""
        path = self.dynamic_csv_path if is_dynamic else self.static_csv_path
        if not path.exists() or path.stat().st_size == 0:
            return {}
        try:
            df = pd.read_csv(path, usecols=["label"])
            counts = df["label"].astype(str).value_counts().to_dict()
            return {str(k): int(v) for k, v in sorted(counts.items())}
        except Exception:
            return {}

