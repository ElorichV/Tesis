from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

import app.cli as cli_module
import recognition.bootstrap as bootstrap_module
from recognition.bootstrap import build_bootstrap_rows
from recognition.trainer import _train_single_csv


def _hand(offset: float = 0.0) -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(offset, offset + 1, 21)
    landmarks[:, 1] = np.linspace(0, 0.5, 21)
    landmarks[:, 2] = np.linspace(0, 0.2, 21)
    return landmarks


def test_build_bootstrap_rows_filters_and_mirrors() -> None:
    rows, labels = build_bootstrap_rows(
        {"a": [_hand()], "B": [_hand(1)], "J": [_hand(2)]},
        static_letters=["A", "b"],
    )

    assert rows.shape == (4, 63)
    assert labels == ["A", "A", "B", "B"]
    normalized = rows[0].reshape(21, 3)
    mirrored = rows[1].reshape(21, 3)
    np.testing.assert_allclose(mirrored[:, 0], -normalized[:, 0])
    np.testing.assert_allclose(mirrored[:, 1:], normalized[:, 1:])


def _write_landmark_csv(path: Path, entries: list[tuple[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label", *(f"f_{idx}" for idx in range(63))])
        for label, value in entries:
            writer.writerow([label, value, *([0.0] * 62)])


def test_trainer_merges_base_rows_and_drops_replaced_classes(tmp_path: Path) -> None:
    user_csv = tmp_path / "user.csv"
    base_csv = tmp_path / "public.csv"
    _write_landmark_csv(
        base_csv,
        [("A", 0.0)] * 4 + [("B", 1.0)] * 4,
    )
    _write_landmark_csv(
        user_csv,
        [("A", 10.0)] * 4 + [("C", 2.0)] * 4,
    )

    report = _train_single_csv(
        csv_path=user_csv,
        model_path=tmp_path / "model.joblib",
        classifier_type="knn",
        alphabet="ASL",
        phase_name="static",
        base_csv=base_csv,
    )

    assert report.num_samples == 12
    assert report.classes == ["A", "B", "C"]


def _mock_model_downloads(monkeypatch, calls: list[tuple[Path, str]]) -> None:
    module = ModuleType("perception.mediapipe_tracker")

    def ensure_model_downloaded(models_dir: Path, filename: str) -> Path:
        calls.append((Path(models_dir), filename))
        return Path(models_dir) / filename

    module.ensure_model_downloaded = ensure_model_downloaded
    monkeypatch.setitem(sys.modules, "perception.mediapipe_tracker", module)


def _recognition_config(tmp_path: Path) -> dict:
    return {
        "features": {"canonical_hand": "Right"},
        "perception": {"models_dir": str(tmp_path / "mediapipe")},
        "recognition": {
            "model_path": str(tmp_path / "sign_classifier.joblib"),
            "bootstrap_public_dataset": True,
        },
    }


def test_ensure_recognition_ready_skips_existing_model(tmp_path: Path, monkeypatch) -> None:
    downloads = []
    _mock_model_downloads(monkeypatch, downloads)
    config = _recognition_config(tmp_path)
    model_path = Path(config["recognition"]["model_path"])
    model_path.touch()
    monkeypatch.setattr(
        bootstrap_module,
        "load_hf_asl",
        lambda _: (_ for _ in ()).throw(AssertionError("no debe cargar el dataset")),
    )
    monkeypatch.setattr(
        cli_module,
        "train_models",
        lambda **_: (_ for _ in ()).throw(AssertionError("no debe entrenar")),
    )

    assert cli_module._ensure_recognition_ready(config, {"code": "ASL", "static": ["A"]})
    assert [name for _, name in downloads] == [
        "hand_landmarker.task",
        "face_landmarker.task",
        "pose_landmarker_lite.task",
    ]


def test_ensure_recognition_ready_bootstraps_missing_model(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    downloads = []
    _mock_model_downloads(monkeypatch, downloads)
    monkeypatch.chdir(tmp_path)
    config = _recognition_config(tmp_path)
    loaded_cache = []
    monkeypatch.setattr(
        bootstrap_module,
        "load_hf_asl",
        lambda cache: loaded_cache.append(cache) or {"A": [_hand()], "B": [_hand(1)]},
    )

    def train(**kwargs):
        assert Path(kwargs["bootstrap_static_csv"]).is_file()
        model_path = Path(kwargs["static_model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()
        return [SimpleNamespace(accuracy=0.95, classes=["A", "B"])]

    monkeypatch.setattr(cli_module, "train_models", train)

    ready = cli_module._ensure_recognition_ready(
        config,
        {"code": "ASL", "static": ["A", "B"]},
    )

    assert ready
    assert loaded_cache == [Path("data/external/asl_static_landmarks_v1")]
    assert len(downloads) == 3
    with Path("data/bootstrap_static.csv").open(encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        first_row = next(reader)
    assert header == ["label", *(f"f_{idx}" for idx in range(63))]
    assert len(first_row) == 64
    assert all("." in value and len(value.rsplit(".", 1)[1]) == 6 for value in first_row[1:])
    output = capsys.readouterr().out
    assert "dataset público ASL (CC-BY-4.0)" in output
    assert "95.00%" in output
    assert "clases: A, B" in output


def test_ensure_recognition_ready_returns_false_when_loader_fails(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _mock_model_downloads(monkeypatch, [])
    config = _recognition_config(tmp_path)
    monkeypatch.setattr(
        bootstrap_module,
        "load_hf_asl",
        lambda _: (_ for _ in ()).throw(OSError("sin conexión")),
    )

    assert not cli_module._ensure_recognition_ready(
        config,
        {"code": "ASL", "static": ["A", "B"]},
    )
    assert "sin conexión" in capsys.readouterr().err
