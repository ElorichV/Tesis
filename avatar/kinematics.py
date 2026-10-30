"""
Cinemática directa de la mano del avatar, compartida por el entrenamiento (Python) y el
visor (JavaScript, `web/js/handRig.js`).

Los ejes articulares se derivan geométricamente de la pose de reposo exportada por Blender
(`data/rig_skeleton.json`), de modo que los mismos parámetros producen la misma forma en
ambas manos (la izquierda queda espejada automáticamente) y en ambos lenguajes.

Convención de parámetros (grados):
- Dedos (index, middle, ring, pinky): `<dedo>_mcp` (flexión), `<dedo>_abd` (positivo hacia
  el pulgar), `<dedo>_pip`, `<dedo>_dip`.
- Pulgar: `thumb_cmc_flex` (cruza la palma), `thumb_cmc_abd` (se separa de la palma),
  `thumb_cmc_twist` (oposición), `thumb_mcp`, `thumb_ip`.
- `palm_cup`: ahuecado de la palma (metacarpos de anular y meñique).
Valores negativos de flexión = hiperextensión (manos hiperelásticas).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
SKELETON_PATH = DATA_DIR / "rig_skeleton.json"

FINGERS = ("index", "middle", "ring", "pinky")

# (mínimo, máximo) en grados: rango natural y rango hiperelástico.
JOINT_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "natural": {
        "thumb_cmc_flex": (-15, 55), "thumb_cmc_abd": (-10, 70), "thumb_cmc_twist": (-30, 45),
        "thumb_mcp": (-10, 60), "thumb_ip": (-15, 80),
        **{f"{f}_mcp": (-20, 90) for f in FINGERS},
        **{f"{f}_abd": (-20, 20) for f in FINGERS},
        **{f"{f}_pip": (0, 105) for f in FINGERS},
        **{f"{f}_dip": (-5, 80) for f in FINGERS},
        "palm_cup": (0, 20),
    },
    "hyper": {
        "thumb_cmc_flex": (-45, 85), "thumb_cmc_abd": (-35, 100), "thumb_cmc_twist": (-75, 75),
        "thumb_mcp": (-50, 95), "thumb_ip": (-60, 105),
        **{f"{f}_mcp": (-75, 115) for f in FINGERS},
        **{f"{f}_abd": (-40, 40) for f in FINGERS},
        **{f"{f}_pip": (-35, 130) for f in FINGERS},
        **{f"{f}_dip": (-45, 105) for f in FINGERS},
        "palm_cup": (-15, 45),
    },
}

PARAM_NAMES: tuple[str, ...] = tuple(JOINT_LIMITS["hyper"].keys())

# Correspondencia landmark MediaPipe -> (hueso, "head"|"tail")
LANDMARK_BONES: list[tuple[str, str]] = [("hand", "head"),
                                         ("thumb_01", "head"), ("thumb_02", "head"),
                                         ("thumb_03", "head"), ("thumb_03", "tail")]
for _f in FINGERS:
    LANDMARK_BONES += [(f"{_f}_01", "head"), (f"{_f}_02", "head"),
                       (f"{_f}_03", "head"), (f"{_f}_03", "tail")]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def axis_angle_matrix(axis: np.ndarray, deg: float) -> np.ndarray:
    a = _unit(np.asarray(axis, dtype=np.float64))
    t = np.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


def rotvec_matrix(rv: np.ndarray) -> np.ndarray:
    ang = float(np.linalg.norm(rv))
    if ang < 1e-12:
        return np.eye(3)
    return axis_angle_matrix(rv / ang, np.degrees(ang))


@dataclass
class BoneDef:
    name: str
    parent: str | None
    head: np.ndarray
    tail: np.ndarray


class HandModel:
    """Modelo cinemático de una mano ("R" o "L") del avatar."""

    def __init__(self, side: str = "R", skeleton_path: str | Path = SKELETON_PATH) -> None:
        self.side = side
        data = json.loads(Path(skeleton_path).read_text(encoding="utf-8"))
        all_bones = {b["name"]: b for b in data["bones"]}
        suffix = f"_{side}"
        names = [n for n in all_bones if n.endswith(suffix) and n.split("_")[0] in
                 ("hand", "thumb", "meta", *FINGERS)]
        self.bones: dict[str, BoneDef] = {}
        for n in names:
            b = all_bones[n]
            base = n[: -len(suffix)]
            parent = b["parent"][: -len(suffix)] if b["parent"] and b["parent"] in names else None
            self.bones[base] = BoneDef(base, parent, np.array(b["head"], float), np.array(b["tail"], float))
        self.order = self._topo_order()
        self.axes = self.compute_axes()

    def _topo_order(self) -> list[str]:
        order, seen = [], set()

        def visit(n):
            if n in seen:
                return
            p = self.bones[n].parent
            if p:
                visit(p)
            seen.add(n)
            order.append(n)

        for n in self.bones:
            visit(n)
        return order

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectores de reposo: muñeca, normal palmar `n` y lateral hacia el pulgar `t`."""
        b = self.bones
        wrist = b["hand"].head
        idx, pky = b["index_01"].head, b["pinky_01"].head
        n = _unit(np.cross(idx - wrist, pky - wrist))
        if np.dot(n, b["thumb_01"].head - wrist) < 0:
            n = -n
        t = _unit(idx - pky)
        return wrist, n, t

    def compute_axes(self) -> dict[str, dict[str, np.ndarray]]:
        _, n, t = self.frame()
        axes: dict[str, dict[str, np.ndarray]] = {}
        for name, bd in self.bones.items():
            y = _unit(bd.tail - bd.head)
            if name.startswith("thumb"):
                flex_dir = _unit(-t + 0.6 * n)
                ax = {"flex": _unit(np.cross(y, flex_dir)), "twist": y}
                if name == "thumb_01":
                    ax["flex"] = _unit(np.cross(y, -t))
                    ax["abd"] = _unit(np.cross(y, n))
            else:
                ax = {"flex": _unit(np.cross(y, n)), "abd": _unit(np.cross(y, t)), "twist": y}
            axes[name] = ax
        return axes

    def local_rotations(self, params: dict[str, float]) -> dict[str, np.ndarray]:
        """Rotaciones de cada hueso expresadas en el marco de reposo (mundo)."""
        p = {k: float(params.get(k, 0.0)) for k in PARAM_NAMES}
        A = self.axes
        R = {name: np.eye(3) for name in self.bones}

        def compose(name, abd=0.0, flex=0.0, twist=0.0):
            ax = A[name]
            m = np.eye(3)
            if abd:
                m = m @ axis_angle_matrix(ax["abd"], abd)
            if flex:
                m = m @ axis_angle_matrix(ax["flex"], flex)
            if twist:
                m = m @ axis_angle_matrix(ax["twist"], twist)
            R[name] = m

        compose("thumb_01", abd=p["thumb_cmc_abd"], flex=p["thumb_cmc_flex"], twist=p["thumb_cmc_twist"])
        compose("thumb_02", flex=p["thumb_mcp"])
        compose("thumb_03", flex=p["thumb_ip"])
        for f in FINGERS:
            compose(f"{f}_01", abd=p[f"{f}_abd"], flex=p[f"{f}_mcp"])
            compose(f"{f}_02", flex=p[f"{f}_pip"])
            compose(f"{f}_03", flex=p[f"{f}_dip"])
        compose("meta_ring", flex=0.5 * p["palm_cup"])
        compose("meta_pinky", flex=p["palm_cup"])
        return R

    def forward(self, params: dict[str, float], global_rot: np.ndarray | None = None) -> np.ndarray:
        """Devuelve los 21 landmarks (21, 3) relativos a la muñeca."""
        Rl = self.local_rotations(params)
        G: dict[str, np.ndarray] = {}
        H: dict[str, np.ndarray] = {}
        root = np.eye(3) if global_rot is None else global_rot
        wrist = self.bones["hand"].head
        for name in self.order:
            bd = self.bones[name]
            if bd.parent is None:
                G[name] = root @ Rl[name]
                H[name] = np.zeros(3)
            else:
                par = self.bones[bd.parent]
                H[name] = H[bd.parent] + G[bd.parent] @ (bd.head - par.head)
                G[name] = G[bd.parent] @ Rl[name]
        del wrist
        out = np.zeros((21, 3))
        for i, (bn, end) in enumerate(LANDMARK_BONES):
            bd = self.bones[bn]
            out[i] = H[bn] if end == "head" else H[bn] + G[bn] @ (bd.tail - bd.head)
        return out


def normalize_points(pts: np.ndarray) -> np.ndarray:
    """Misma normalización que `features.landmarks.normalize_hand_landmarks` (sin espejado)."""
    rel = pts - pts[0]
    m = float(np.max(np.linalg.norm(rel, axis=1)))
    return rel / m if m > 1e-9 else rel


def mediapipe_to_avatar(points_xyz: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Convierte coordenadas de imagen MediaPipe (x der., y abajo, z hacia adentro) al marco
    del visor (x der. del espectador, y arriba, z hacia el espectador)."""
    p = np.asarray(points_xyz, dtype=np.float64).reshape(21, 3).copy()
    p[:, 1] *= -1.0
    p[:, 2] *= -1.0
    if mirror:
        p[:, 0] *= -1.0
    return p
