"""Pruebas unitarias para la normalización de landmarks y trayectorias dinámicas."""

from __future__ import annotations

import numpy as np
import pytest

from features.landmarks import (
    DYNAMIC_FEATURE_DIM,
    HAND_LANDMARK_COUNT,
    STATIC_FEATURE_DIM,
    TrajectoryBuffer,
    extract_dynamic_features,
    normalize_hand_landmarks,
)


def _sample_right_hand() -> np.ndarray:
    rng = np.random.default_rng(12345)
    pts = rng.uniform(0.2, 0.8, size=(HAND_LANDMARK_COUNT, 3)).astype(np.float64)
    pts[0] = [0.5, 0.5, 0.0]  # Muñeca en el centro
    return pts


def test_normalize_output_shape_and_wrist_origin() -> None:
    pts = _sample_right_hand()
    norm = normalize_hand_landmarks(pts, handedness="Right", canonical_hand="Right")

    assert norm.shape == (STATIC_FEATURE_DIM,)
    assert norm.dtype == np.float32
    # La muñeca (primeros 3 valores) debe quedar en (0, 0, 0)
    np.testing.assert_allclose(norm[:3], [0.0, 0.0, 0.0], atol=1e-6)
    # La distancia máxima desde la muñeca debe ser 1.0
    reshaped = norm.reshape(HAND_LANDMARK_COUNT, 3)
    max_dist = float(np.max(np.linalg.norm(reshaped, axis=1)))
    assert pytest.approx(max_dist, rel=1e-5) == 1.0


def test_translation_invariance() -> None:
    pts = _sample_right_hand()
    shifted = pts + np.array([12.5, -7.3, 3.1])

    norm_orig = normalize_hand_landmarks(pts, handedness="Right")
    norm_shifted = normalize_hand_landmarks(shifted, handedness="Right")

    np.testing.assert_allclose(norm_orig, norm_shifted, atol=1e-6)


def test_scale_invariance() -> None:
    pts = _sample_right_hand()
    wrist = pts[0:1]
    scaled = wrist + (pts - wrist) * 3.75

    norm_orig = normalize_hand_landmarks(pts, handedness="Right")
    norm_scaled = normalize_hand_landmarks(scaled, handedness="Right")

    np.testing.assert_allclose(norm_orig, norm_scaled, atol=1e-6)


def test_left_right_hand_mirror_invariance() -> None:
    right_pts = _sample_right_hand()
    # Espejar horizontalmente respecto a x = 0.5 (como una mano izquierda simétrica)
    left_pts = right_pts.copy()
    left_pts[:, 0] = 1.0 - right_pts[:, 0]

    norm_right = normalize_hand_landmarks(right_pts, handedness="Right", canonical_hand="Right")
    norm_left = normalize_hand_landmarks(left_pts, handedness="Left", canonical_hand="Right")

    np.testing.assert_allclose(norm_right, norm_left, atol=1e-6)


def test_zero_collapsed_hand_returns_zeros() -> None:
    collapsed = np.ones((HAND_LANDMARK_COUNT, 3), dtype=np.float32) * 0.5
    norm = normalize_hand_landmarks(collapsed)
    assert norm.shape == (STATIC_FEATURE_DIM,)
    assert np.all(norm == 0.0)


def test_invalid_landmark_count_raises() -> None:
    bad = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        normalize_hand_landmarks(bad)


def test_dynamic_trajectory_features_and_buffer() -> None:
    base = _sample_right_hand()
    buf = TrajectoryBuffer(maxlen=20, canonical_hand="Right")

    for step in range(20):
        frame_pts = base.copy()
        frame_pts[:, 0] += step * 0.01
        frame_pts[:, 1] += np.sin(step * 0.2) * 0.02
        buf.push(frame_pts, handedness="Right")

    assert buf.is_full()
    assert buf.motion_score() > 0.05

    feat = buf.get_features()
    assert feat is not None
    assert feat.shape == (DYNAMIC_FEATURE_DIM,)

    # Verificar invarianza a traslación global en la secuencia dinámica
    frames_shifted = [f + np.array([2.0, -1.0, 0.5]) for f in buf._frames]
    feat_shifted = extract_dynamic_features(frames_shifted, handedness="Right")
    np.testing.assert_allclose(feat, feat_shifted, atol=1e-5)

