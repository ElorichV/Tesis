"""
Percepción unificada con MediaPipe Tasks:
- HandLandmarker: 21 puntos 3D por mano y lateralidad (Left/Right).
- FaceLandmarker (con output_face_blendshapes=True): contorno facial y gestos faciales
  (sonrisa, cejas arriba, cejas abajo, boca abierta, parpadeo).
- PoseLandmarker (lite): hombros, codos y muñecas para seguimiento de brazos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

# Índices de pose para hombros, codos y muñecas (MediaPipe Pose)
POSE_ARM_INDICES = (11, 12, 13, 14, 15, 16)
POSE_ARM_CONNECTIONS = (
    (11, 12),  # línea entre hombros
    (11, 13),  # hombro izq -> codo izq
    (13, 15),  # codo izq -> muñeca izq
    (12, 14),  # hombro der -> codo der
    (14, 16),  # codo der -> muñeca der
)

# Conexiones de los 21 landmarks de la mano
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anular
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique y palma
)

# Índices clave del contorno facial, cejas, ojos y labios (MediaPipe FaceMesh 478 puntos)
FACE_OVAL_INDICES = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
)
LEFT_EYEBROW_INDICES = (70, 63, 105, 66, 107)
RIGHT_EYEBROW_INDICES = (336, 296, 334, 293, 300)
LIPS_OUTER_INDICES = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61)


@dataclass
class DetectedHand:
    """Representa una mano detectada con sus 21 puntos normalizados a la imagen [0,1]."""

    landmarks: np.ndarray  # shape (21, 3) en coordenadas de imagen [x, y, z]
    handedness: str        # "Left" o "Right"
    score: float


@dataclass
class FaceState:
    """Estado facial: puntos de la malla y valores de blendshapes clave."""

    landmarks: np.ndarray | None = None  # shape (N, 3)
    blendshapes: dict[str, float] = field(default_factory=dict)
    active_gestures: set[str] = field(default_factory=set)


@dataclass
class PerceptionResult:
    """Resultado completo de percepción para un frame."""

    hands: list[DetectedHand] = field(default_factory=list)
    face: FaceState = field(default_factory=FaceState)
    pose_landmarks: np.ndarray | None = None  # shape (33, 4) -> x, y, z, visibility
    active_gestures: set[str] = field(default_factory=set)

    @property
    def primary_hand(self) -> DetectedHand | None:
        if not self.hands:
            return None
        # Prioriza la mano con mayor confianza de detección
        return max(self.hands, key=lambda h: h.score)


def ensure_model_downloaded(models_dir: Path, filename: str) -> Path:
    """Descarga el archivo `.task` oficial de MediaPipe si no existe localmente."""
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / filename
    if target.exists() and target.stat().st_size > 10_000:
        return target

    url = MODEL_URLS[filename]
    logger.info("Descargando modelo MediaPipe '%s' desde %s ...", filename, url)
    urllib.request.urlretrieve(url, str(target))
    return target


class PerceptionTracker:
    """
    Ejecuta HandLandmarker, FaceLandmarker (con blendshapes) y PoseLandmarker en CPU.
    Optimiza el rendimiento ejecutando manos en cada frame y cara/pose con stride configurable.
    """

    def __init__(
        self,
        models_dir: str | Path = "models/mediapipe",
        max_num_hands: int = 2,
        min_hand_detection_confidence: float = 0.55,
        min_hand_presence_confidence: float = 0.55,
        min_tracking_confidence: float = 0.50,
        enable_face: bool = True,
        enable_pose: bool = True,
        blendshape_thresholds: dict[str, float] | None = None,
        face_pose_stride: int = 2,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.enable_face = enable_face
        self.enable_pose = enable_pose
        self.face_pose_stride = max(1, face_pose_stride)
        self.thresholds = {
            "smile": 0.45,
            "brow_up": 0.45,
            "brow_down": 0.40,
            "mouth_open": 0.45,
            "blink": 0.50,
        }
        if blendshape_thresholds:
            self.thresholds.update(blendshape_thresholds)

        self._frame_idx = 0
        self._last_timestamp_ms = 0
        self._cached_face = FaceState()
        self._cached_pose: np.ndarray | None = None

        BaseOptions = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        RunningMode = vision.RunningMode

        hand_model_path = ensure_model_downloaded(self.models_dir, "hand_landmarker.task")
        hand_opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

        self._face_landmarker = None
        if self.enable_face:
            try:
                face_model_path = ensure_model_downloaded(self.models_dir, "face_landmarker.task")
                face_opts = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(face_model_path)),
                    running_mode=RunningMode.VIDEO,
                    num_faces=1,
                    output_face_blendshapes=True,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
            except Exception as exc:
                logger.warning("FaceLandmarker deshabilitado por error al cargar: %s", exc)

        self._pose_landmarker = None
        if self.enable_pose:
            try:
                pose_model_path = ensure_model_downloaded(
                    self.models_dir, "pose_landmarker_lite.task"
                )
                pose_opts = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(pose_model_path)),
                    running_mode=RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
            except Exception as exc:
                logger.warning("PoseLandmarker deshabilitado por error al cargar: %s", exc)

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> PerceptionResult:
        """Procesa un frame BGR y devuelve manos, cara (blendshapes) y pose de brazos."""
        self._frame_idx += 1
        if timestamp_ms is None:
            timestamp_ms = self._last_timestamp_ms + 33
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 1. Detección de manos (en cada frame)
        hands: list[DetectedHand] = []
        hand_res = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        if hand_res and hand_res.hand_landmarks:
            for lm_list, handed_list in zip(hand_res.hand_landmarks, hand_res.handedness):
                coords = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
                category = handed_list[0]
                label = category.category_name or "Right"
                score = float(category.score or 0.0)
                hands.append(DetectedHand(landmarks=coords, handedness=label, score=score))

        # 2. Detección de cara + blendshapes y pose (cada `face_pose_stride` frames)
        run_heavy = (self._frame_idx % self.face_pose_stride == 1) or (self.face_pose_stride == 1)

        if run_heavy and self._face_landmarker is not None:
            face_res = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
            self._cached_face = self._extract_face_state(face_res)

        if run_heavy and self._pose_landmarker is not None:
            pose_res = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_res and pose_res.pose_landmarks:
                plm = pose_res.pose_landmarks[0]
                self._cached_pose = np.array(
                    [[p.x, p.y, p.z, getattr(p, "visibility", 1.0)] for p in plm],
                    dtype=np.float32,
                )
            else:
                self._cached_pose = None

        active_gestures = set(self._cached_face.active_gestures)
        for h in hands:
            if self._is_fist(h.landmarks) and h.handedness.lower() == "left":
                active_gestures.add("fist_left")

        return PerceptionResult(
            hands=hands,
            face=self._cached_face,
            pose_landmarks=self._cached_pose,
            active_gestures=active_gestures,
        )

    def _extract_face_state(self, face_res) -> FaceState:
        if not face_res or not face_res.face_landmarks:
            return FaceState()

        flm = face_res.face_landmarks[0]
        coords = np.array([[p.x, p.y, p.z] for p in flm], dtype=np.float32)

        raw_bs: dict[str, float] = {}
        if face_res.face_blendshapes:
            for cat in face_res.face_blendshapes[0]:
                raw_bs[cat.category_name] = float(cat.score)

        smile = 0.5 * (raw_bs.get("mouthSmileLeft", 0.0) + raw_bs.get("mouthSmileRight", 0.0))
        brow_up = max(
            raw_bs.get("browInnerUp", 0.0),
            0.5 * (raw_bs.get("browOuterUpLeft", 0.0) + raw_bs.get("browOuterUpRight", 0.0)),
        )
        brow_down = 0.5 * (raw_bs.get("browDownLeft", 0.0) + raw_bs.get("browDownRight", 0.0))
        mouth_open = raw_bs.get("jawOpen", 0.0)
        blink = 0.5 * (raw_bs.get("eyeBlinkLeft", 0.0) + raw_bs.get("eyeBlinkRight", 0.0))

        summary_bs = {
            "smile": smile,
            "brow_up": brow_up,
            "brow_down": brow_down,
            "mouth_open": mouth_open,
            "blink": blink,
        }

        active: set[str] = set()
        for name, val in summary_bs.items():
            if val >= self.thresholds.get(name, 0.45):
                active.add(name)

        return FaceState(landmarks=coords, blendshapes=summary_bs, active_gestures=active)

    @staticmethod
    def _is_fist(hand_landmarks: np.ndarray) -> bool:
        """Heurística geométrica simple para detectar puño cerrado."""
        wrist = hand_landmarks[0, :2]
        fingertips = hand_landmarks[[8, 12, 16, 20], :2]
        pip_joints = hand_landmarks[[6, 10, 14, 18], :2]
        tip_dists = np.linalg.norm(fingertips - wrist, axis=1)
        pip_dists = np.linalg.norm(pip_joints - wrist, axis=1)
        return bool(np.all(tip_dists < pip_dists))

    def close(self) -> None:
        for attr in ("_hand_landmarker", "_face_landmarker", "_pose_landmarker"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)

