"""Normalización de landmarks de mano y trayectorias temporales para señas."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np

HAND_LANDMARK_COUNT = 21
STATIC_FEATURE_DIM = HAND_LANDMARK_COUNT * 3  # 63 coordenadas (x, y, z) normalizadas
DYNAMIC_KEYFRAMES = 10
DYNAMIC_FEATURE_DIM = STATIC_FEATURE_DIM + (DYNAMIC_KEYFRAMES * 6) + 2  # 125 dims

WRIST_IDX = 0
INDEX_TIP_IDX = 8
PINKY_TIP_IDX = 20


def normalize_hand_landmarks(
    landmarks: Sequence[Sequence[float]] | np.ndarray,
    handedness: str = "Right",
    canonical_hand: str = "Right",
) -> np.ndarray:
    """
    Normaliza 21 puntos 3D de una mano para lograr invarianza a:
    1. Traslación (posición en la imagen): origen en la muñeca (índice 0).
    2. Lateralidad (mano izquierda/derecha): espeja el eje X si no coincide con canonical_hand.
    3. Escala (distancia a la cámara / tamaño de mano): divide por la distancia máxima a la muñeca.

    Args:
        landmarks: Secuencia o arreglo de forma (21, 3) o (21, 2) con coordenadas (x, y[, z]).
        handedness: "Left" o "Right" según el detector.
        canonical_hand: Marco canónico ("Right" por defecto).

    Returns:
        np.ndarray unidimensional de tamaño 63 (float32) con valores en [-1.0, 1.0].
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] != HAND_LANDMARK_COUNT or pts.shape[1] not in (2, 3):
        raise ValueError(
            f"Se esperaban {HAND_LANDMARK_COUNT} landmarks con 2 o 3 coordenadas; "
            f"se recibió forma {pts.shape}."
        )

    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((HAND_LANDMARK_COUNT, 1), dtype=np.float64)])

    # 1. Traslación relativa a la muñeca (landmark 0)
    wrist = pts[WRIST_IDX].copy()
    rel = pts - wrist

    # 2. Espejado horizontal si la mano detectada es opuesta a la canónica
    if handedness.strip().lower() != canonical_hand.strip().lower():
        rel[:, 0] = -rel[:, 0]

    # 3. Escalado por tamaño de la mano (distancia euclidiana máxima desde la muñeca)
    distances = np.linalg.norm(rel, axis=1)
    max_dist = float(np.max(distances))
    if max_dist < 1e-6:
        return np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)

    normalized = (rel / max_dist).astype(np.float32)
    return normalized.reshape(-1)


def extract_dynamic_features(
    frames_landmarks: Sequence[Sequence[Sequence[float]] | np.ndarray],
    handedness: str = "Right",
    canonical_hand: str = "Right",
) -> np.ndarray:
    """
    Extrae un vector de características de longitud fija (125 dims) a partir de una ventana
    de frames (p. ej. 20-30 frames) para clasificar letras con movimiento (J, Z, Ñ, etc.).

    Combina:
    - Forma promedio normalizada de la mano en la ventana (63 dims).
    - Trayectoria relativa de muñeca, punta del índice (8) y punta del meñique (20)
      muestreada en DYNAMIC_KEYFRAMES instantes y escalada por el tamaño medio de la mano (60 dims).
    - Longitud de recorrido normalizada del índice y del meñique (2 dims).
    """
    if len(frames_landmarks) < 2:
        raise ValueError("Se requieren al menos 2 frames para extraer características dinámicas.")

    raw_seq = []
    norm_shapes = []
    hand_scales = []

    mirror_x = handedness.strip().lower() != canonical_hand.strip().lower()

    for lm in frames_landmarks:
        pts = np.asarray(lm, dtype=np.float64)
        if pts.shape == (HAND_LANDMARK_COUNT, 2):
            pts = np.hstack([pts, np.zeros((HAND_LANDMARK_COUNT, 1), dtype=np.float64)])
        elif pts.shape != (HAND_LANDMARK_COUNT, 3):
            raise ValueError(f"Frame con forma inválida: {pts.shape}")

        norm_shapes.append(
            normalize_hand_landmarks(pts, handedness=handedness, canonical_hand=canonical_hand)
        )

        # Escala aparente de la mano en el espacio de imagen (2D)
        rel_2d = pts[:, :2] - pts[WRIST_IDX, :2]
        scale = float(np.max(np.linalg.norm(rel_2d, axis=1)))
        hand_scales.append(max(scale, 1e-4))

        # Puntos clave de trayectoria (x, y): muñeca, índice, meñique
        key_pts = pts[[WRIST_IDX, INDEX_TIP_IDX, PINKY_TIP_IDX], :2].copy()
        if mirror_x:
            key_pts[:, 0] = -key_pts[:, 0]
        raw_seq.append(key_pts)

    mean_shape = np.mean(np.stack(norm_shapes, axis=0), axis=0).astype(np.float32)
    ref_scale = float(np.median(hand_scales))

    seq_arr = np.stack(raw_seq, axis=0)  # (T, 3, 2)
    origin = seq_arr[0:1, :, :]          # posición inicial de cada punto en t=0
    rel_traj = (seq_arr - origin) / ref_scale  # (T, 3, 2)

    # Interpolación/submuestreo a DYNAMIC_KEYFRAMES instantes uniformes
    t_len = rel_traj.shape[0]
    indices = np.linspace(0, t_len - 1, DYNAMIC_KEYFRAMES)
    t_orig = np.arange(t_len, dtype=np.float64)

    sampled = np.zeros((DYNAMIC_KEYFRAMES, 3, 2), dtype=np.float32)
    for pt_i in range(3):
        for coord_j in range(2):
            sampled[:, pt_i, coord_j] = np.interp(
                indices, t_orig, rel_traj[:, pt_i, coord_j]
            ).astype(np.float32)

    # Longitud total recorrida por punta del índice y punta del meñique
    diffs = np.diff(rel_traj, axis=0)  # (T-1, 3, 2)
    step_lengths = np.linalg.norm(diffs, axis=2)  # (T-1, 3)
    index_path_len = float(np.sum(step_lengths[:, 1]))
    pinky_path_len = float(np.sum(step_lengths[:, 2]))
    path_stats = np.array([index_path_len, pinky_path_len], dtype=np.float32)

    return np.concatenate([mean_shape, sampled.reshape(-1), path_stats]).astype(np.float32)


def compute_hand_motion(
    frames_landmarks: Sequence[Sequence[Sequence[float]] | np.ndarray],
) -> float:
    """
    Calcula el desplazamiento relativo máximo de muñeca/índice/meñique en la ventana
    (normalizado por el tamaño de la mano) para decidir si hay movimiento significativo.
    """
    if len(frames_landmarks) < 2:
        return 0.0

    pts_list = [np.asarray(lm, dtype=np.float64)[:, :2] for lm in frames_landmarks]
    scales = [
        max(float(np.max(np.linalg.norm(p - p[WRIST_IDX], axis=1))), 1e-4)
        for p in pts_list
    ]
    ref_scale = float(np.median(scales))

    tracked = np.stack(
        [p[[WRIST_IDX, INDEX_TIP_IDX, PINKY_TIP_IDX]] for p in pts_list],
        axis=0,
    )  # (T, 3, 2)
    disp = np.linalg.norm(tracked - tracked[0:1], axis=2) / ref_scale  # (T, 3)
    return float(np.max(disp))


class TrajectoryBuffer:
    """Ventana deslizante de frames de mano para reconocimiento de señas dinámicas."""

    def __init__(self, maxlen: int = 25, canonical_hand: str = "Right") -> None:
        if maxlen < 2:
            raise ValueError("maxlen debe ser al menos 2.")
        self.maxlen = maxlen
        self.canonical_hand = canonical_hand
        self._frames: deque[np.ndarray] = deque(maxlen=maxlen)
        self._handedness: str = "Right"

    def push(self, landmarks: Iterable[Sequence[float]] | np.ndarray, handedness: str = "Right") -> None:
        arr = np.asarray(landmarks, dtype=np.float32)
        if self._frames and handedness != self._handedness:
            self._frames.clear()
        self._handedness = handedness
        self._frames.append(arr)

    def clear(self) -> None:
        self._frames.clear()

    def is_full(self) -> bool:
        return len(self._frames) >= self.maxlen

    def __len__(self) -> int:
        return len(self._frames)

    def motion_score(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return compute_hand_motion(list(self._frames))

    def get_features(self) -> np.ndarray | None:
        if len(self._frames) < 2:
            return None
        return extract_dynamic_features(
            list(self._frames),
            handedness=self._handedness,
            canonical_hand=self.canonical_hand,
        )

