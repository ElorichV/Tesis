"""Pruebas unitarias para colección CSV, entrenamiento, inferencia y adaptador de voz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from features.landmarks import HAND_LANDMARK_COUNT
from recognition.collector import SampleCollector
from recognition.predictor import SignPredictor
from recognition.trainer import train_models
from speech.voice_adapter import SpeechOutput


def _synthetic_hand_for_class(class_idx: int, rng: np.random.Generator) -> np.ndarray:
    pts = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float64)
    for i in range(1, HAND_LANDMARK_COUNT):
        angle = (i * 0.25) + class_idx * 1.2
        pts[i, 0] = np.cos(angle) * (0.1 + 0.02 * i) + rng.normal(0, 0.005)
        pts[i, 1] = np.sin(angle) * (0.1 + 0.02 * i) + rng.normal(0, 0.005)
        pts[i, 2] = (class_idx * 0.05) + rng.normal(0, 0.002)
    return pts


def test_collector_trainer_and_predictor_pipeline(tmp_path: Path) -> None:
    static_csv = tmp_path / "static.csv"
    dynamic_csv = tmp_path / "dynamic.csv"
    static_model = tmp_path / "static_model.joblib"
    dynamic_model = tmp_path / "dynamic_model.joblib"

    collector = SampleCollector(static_csv_path=static_csv, dynamic_csv_path=dynamic_csv)
    rng = np.random.default_rng(42)

    classes = ["A", "B", "C"]
    for idx, label in enumerate(classes):
        for _ in range(15):
            lm = _synthetic_hand_for_class(idx, rng)
            collector.save_static_sample(label, lm, handedness="Right")

    counts = collector.count_samples_by_label(is_dynamic=False)
    assert counts == {"A": 15, "B": 15, "C": 15}

    reports = train_models(
        static_csv=static_csv,
        dynamic_csv=dynamic_csv,
        static_model_path=static_model,
        dynamic_model_path=dynamic_model,
        classifier_type="knn",
        alphabet="ASL",
    )
    assert len(reports) == 1
    assert reports[0].accuracy >= 0.90
    assert static_model.exists()

    predictor = SignPredictor(
        static_model_path=static_model,
        dynamic_model_path=dynamic_model,
    )
    assert predictor.is_ready
    sample_a = _synthetic_hand_for_class(0, rng)
    pred = predictor.predict(sample_a, handedness="Right")
    assert pred.letter == "A"
    assert pred.confidence > 0.70
    assert pred.mode == "static"


def test_speech_output_loads_voice_translate_or_stub() -> None:
    adapter = SpeechOutput(voice_translate_dir="./voice-translate", async_playback=False)
    # Con texto vacío no debe invocar síntesis
    assert adapter.speak("   ") is False

