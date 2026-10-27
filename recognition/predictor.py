"""Inferencia en tiempo real para letras estáticas (Fase 1) y dinámicas (Fase 2)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from features.landmarks import TrajectoryBuffer, normalize_hand_landmarks

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    """Resultado de clasificación para un frame."""

    letter: str | None
    confidence: float
    mode: str  # "static" | "dynamic" | "untrained" | "no_hand"


class SignPredictor:
    """
    Carga los clasificadores `joblib` de Fase 1 (estático) y Fase 2 (dinámico)
    y predice la letra junto con su probabilidad en cada frame.
    """

    def __init__(
        self,
        static_model_path: str | Path = "models/sign_classifier.joblib",
        dynamic_model_path: str | Path = "models/sign_dynamic_classifier.joblib",
        canonical_hand: str = "Right",
        dynamic_window_frames: int = 25,
        dynamic_min_motion: float = 0.08,
    ) -> None:
        self.static_model_path = Path(static_model_path)
        self.dynamic_model_path = Path(dynamic_model_path)
        self.canonical_hand = canonical_hand
        self.dynamic_min_motion = dynamic_min_motion

        self.trajectory_buffer = TrajectoryBuffer(
            maxlen=dynamic_window_frames,
            canonical_hand=canonical_hand,
        )

        self._static_model = None
        self._static_classes: list[str] = []
        self._dynamic_model = None
        self._dynamic_classes: list[str] = []

        self.reload_models()

    def reload_models(self) -> None:
        """Recarga los modelos desde disco si existen."""
        if self.static_model_path.exists():
            try:
                payload = joblib.load(self.static_model_path)
                self._static_model = payload["model"]
                self._static_classes = list(payload.get("classes", []))
                logger.info(
                    "Modelo estático cargado (%d clases): %s",
                    len(self._static_classes),
                    ", ".join(self._static_classes),
                )
            except Exception as exc:
                logger.warning("No se pudo cargar modelo estático %s: %s", self.static_model_path, exc)
                self._static_model = None

        if self.dynamic_model_path.exists():
            try:
                payload = joblib.load(self.dynamic_model_path)
                self._dynamic_model = payload["model"]
                self._dynamic_classes = list(payload.get("classes", []))
                logger.info(
                    "Modelo dinámico cargado (%d clases): %s",
                    len(self._dynamic_classes),
                    ", ".join(self._dynamic_classes),
                )
            except Exception as exc:
                logger.warning(
                    "No se pudo cargar modelo dinámico %s: %s", self.dynamic_model_path, exc
                )
                self._dynamic_model = None

    @property
    def is_ready(self) -> bool:
        return self._static_model is not None or self._dynamic_model is not None

    @property
    def static_classes(self) -> list[str]:
        return list(self._static_classes)

    @property
    def dynamic_classes(self) -> list[str]:
        return list(self._dynamic_classes)

    def clear_temporal_state(self) -> None:
        self.trajectory_buffer.clear()

    def predict(
        self,
        landmarks: Sequence[Sequence[float]] | np.ndarray | None,
        handedness: str = "Right",
    ) -> PredictionOutput:
        """
        Realiza inferencia sobre los landmarks de la mano detectada.
        Si hay movimiento significativo y el modelo de Fase 2 está disponible, evalúa Fase 2;
        en caso contrario, evalúa el clasificador estático de Fase 1.
        """
        if landmarks is None:
            self.trajectory_buffer.clear()
            return PredictionOutput(letter=None, confidence=0.0, mode="no_hand")

        if not self.is_ready:
            return PredictionOutput(letter=None, confidence=0.0, mode="untrained")

        self.trajectory_buffer.push(landmarks, handedness=handedness)

        # Fase 2: evaluar trayectoria dinámica si la ventana está llena y hay desplazamiento
        if (
            self._dynamic_model is not None
            and self.trajectory_buffer.is_full()
            and self.trajectory_buffer.motion_score() >= self.dynamic_min_motion
        ):
            dyn_feat = self.trajectory_buffer.get_features()
            if dyn_feat is not None:
                dyn_letter, dyn_conf = self._predict_with_pipeline(
                    self._dynamic_model, dyn_feat.reshape(1, -1)
                )
                if dyn_letter is not None and dyn_conf >= 0.70:
                    return PredictionOutput(
                        letter=dyn_letter, confidence=dyn_conf, mode="dynamic"
                    )

        # Fase 1: evaluar postura estática de la mano
        if self._static_model is not None:
            static_feat = normalize_hand_landmarks(
                landmarks,
                handedness=handedness,
                canonical_hand=self.canonical_hand,
            )
            letter, conf = self._predict_with_pipeline(
                self._static_model, static_feat.reshape(1, -1)
            )
            return PredictionOutput(letter=letter, confidence=conf, mode="static")

        return PredictionOutput(letter=None, confidence=0.0, mode="untrained")

    @staticmethod
    def _predict_with_pipeline(pipeline, X: np.ndarray) -> tuple[str | None, float]:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)[0]
            best_idx = int(np.argmax(probs))
            letter = str(pipeline.classes_[best_idx])
            confidence = float(probs[best_idx])
            return letter, confidence

        pred = str(pipeline.predict(X)[0])
        return pred, 1.0

