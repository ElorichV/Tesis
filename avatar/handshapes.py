"""
Biblioteca de configuraciones manuales (handshapes) del avatar.

Combina dos fuentes:
1. `data/handshapes_trained.json`: configuraciones ajustadas a landmarks reales (`python -m avatar.train`).
2. `AUTHORED`: configuraciones descritas anatómicamente para las formas que no tienen datos
   (clasificadores LSM y letras cuya forma difiere entre ASL y LSM).

`build_library()` genera `web/assets/handshapes.json`, que contiene, por configuración, la rotación
de cada hueso de la mano (cuaternión x, y, z, w en el marco de reposo del avatar) para ambas manos.
El visor sólo tiene que interpolar esos cuaterniones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avatar.kinematics import DATA_DIR, FINGERS, JOINT_LIMITS, PARAM_NAMES, HandModel

TRAINED_PATH = DATA_DIR / "handshapes_trained.json"
WEB_LIBRARY_PATH = Path(__file__).resolve().parent / "web" / "assets" / "handshapes.json"

# Letras LSM cuya configuración coincide en la mano con la letra homónima de ASL: se toman
# de los datos entrenados. La orientación propia de LSM se define en el léxico/alfabeto.
LSM_FROM_ASL = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H", "I": "I",
    "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P", "R": "R", "S": "S", "T": "T",
    "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Q": "Q",
}


def _fingers(mcp=0.0, pip=0.0, dip=0.0, abd=0.0, which=FINGERS) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in which:
        out[f"{f}_mcp"], out[f"{f}_pip"], out[f"{f}_dip"] = mcp, pip, dip
        if abd:
            sign = {"index": 1, "middle": 0.3, "ring": -0.4, "pinky": -1}[f]
            out[f"{f}_abd"] = abd * sign
    return out


FIST = _fingers(85, 100, 55)
THUMB_ACROSS = {"thumb_cmc_flex": 45, "thumb_cmc_abd": 10, "thumb_mcp": 40, "thumb_ip": 30}
THUMB_SIDE = {"thumb_cmc_flex": 10, "thumb_cmc_abd": 5, "thumb_mcp": 10, "thumb_ip": 10}
THUMB_OPEN = {"thumb_cmc_flex": -10, "thumb_cmc_abd": 45, "thumb_mcp": 0, "thumb_ip": 0}

AUTHORED: dict[str, dict[str, float]] = {
    # Clasificadores / configuraciones básicas de LSM
    "plana": {**THUMB_SIDE, "thumb_cmc_flex": 0},
    "5": {**THUMB_OPEN, **_fingers(0, 0, 0, abd=18)},
    "5_garra": {**THUMB_OPEN, "thumb_ip": 35, **_fingers(25, 60, 45, abd=15)},
    "puño": {**FIST, **THUMB_ACROSS},
    "1": {**FIST, **THUMB_ACROSS, **_fingers(0, 0, 0, which=("index",))},
    "2": {**FIST, **THUMB_ACROSS, **_fingers(0, 0, 0, abd=12, which=("index", "middle"))},
    "3": {**FIST, **THUMB_OPEN, **_fingers(0, 0, 0, abd=12, which=("index", "middle"))},
    "4": {**_fingers(0, 0, 0, abd=10), **THUMB_ACROSS, "thumb_cmc_flex": 60, "thumb_mcp": 50},
    "pulgar": {**FIST, **THUMB_OPEN, "thumb_cmc_abd": 20, "thumb_cmc_flex": -20},
    "pinza": {**FIST, "index_mcp": 40, "index_pip": 55, "index_dip": 30,
              "thumb_cmc_flex": 35, "thumb_cmc_abd": 40, "thumb_cmc_twist": 30, "thumb_mcp": 15},
    "bolsa": {**_fingers(45, 45, 25), "thumb_cmc_flex": 40, "thumb_cmc_abd": 40,
              "thumb_cmc_twist": 35, "thumb_mcp": 10},
    "curva": {**_fingers(45, 40, 20), **THUMB_OPEN, "thumb_cmc_abd": 35},
    "doblada": {**_fingers(90, 0, 0), **THUMB_SIDE},
    "cuernos": {**FIST, **THUMB_ACROSS, **_fingers(0, 0, 0, which=("index", "pinky"))},
    "llamar": {**FIST, **THUMB_OPEN, **_fingers(0, 0, 0, which=("pinky",))},
    # Hiperextensión (rangos "hyper"): dedos arqueados hacia atrás, usada en señas enfáticas
    "plana_hiper": {**_fingers(-40, -20, -15), "thumb_cmc_flex": -30, "thumb_cmc_abd": 60,
                    "thumb_mcp": -30, "thumb_ip": -40},
    # Respaldo por si no hay datos entrenados para estas letras
    "fallback_A": {**FIST, **THUMB_SIDE},
    "fallback_I": {**FIST, **THUMB_ACROSS, **_fingers(0, 0, 0, which=("pinky",))},
    "fallback_L": {**FIST, **THUMB_OPEN, **_fingers(0, 0, 0, which=("index",))},
    "fallback_Y": {**FIST, **THUMB_OPEN, **_fingers(0, 0, 0, which=("pinky",))},
    "fallback_O": {**_fingers(45, 55, 35), "thumb_cmc_flex": 40, "thumb_cmc_abd": 40,
                   "thumb_cmc_twist": 30, "thumb_mcp": 20, "thumb_ip": 20},
}


def clamp_params(params: dict[str, float], limits: str = "hyper") -> dict[str, float]:
    lim = JOINT_LIMITS[limits]
    return {k: float(np.clip(params.get(k, 0.0), *lim[k])) for k in PARAM_NAMES}


def _quat(m: np.ndarray) -> list[float]:
    from avatar.train import _matrix_to_quat

    return [round(float(c), 5) for c in _matrix_to_quat(m)]


def load_trained(path: Path = TRAINED_PATH) -> dict:
    if not path.exists():
        return {"alphabets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_params(trained: dict | None = None) -> dict[str, dict]:
    """Devuelve {nombre: {"params": {...}, "orientation": quat | None, "source": str}}."""
    trained = trained if trained is not None else load_trained()
    shapes: dict[str, dict] = {}
    for name, params in AUTHORED.items():
        shapes[name] = {"params": clamp_params(params), "orientation": None, "source": "authored"}
    asl = trained.get("alphabets", {}).get("ASL", {})
    lsm = trained.get("alphabets", {}).get("LSM", {})
    for letter, entry in asl.items():
        shapes[f"ASL_{letter}"] = {"params": clamp_params(entry["params"]),
                                   "orientation": entry.get("orientation"), "source": entry["source"],
                                   "rms_error": entry.get("rms_error")}
    for letter, asl_letter in LSM_FROM_ASL.items():
        if letter in lsm:
            e = lsm[letter]
        elif asl_letter in asl:
            e = asl[asl_letter]
        elif f"fallback_{letter}" in AUTHORED:
            e = {"params": AUTHORED[f"fallback_{letter}"], "orientation": None, "source": "authored"}
        else:
            continue
        shapes[f"LSM_{letter}"] = {"params": clamp_params(e["params"]), "orientation": e.get("orientation"),
                                   "source": e.get("source", "authored"), "rms_error": e.get("rms_error")}
    return shapes


def build_library(out_path: Path = WEB_LIBRARY_PATH) -> dict:
    models = {"R": HandModel("R"), "L": HandModel("L")}
    shapes = resolve_params()
    library = {"param_names": list(PARAM_NAMES), "bones": sorted(models["R"].bones), "handshapes": {}}
    for name, sh in sorted(shapes.items()):
        item = {"source": sh["source"], "params": {k: round(v, 1) for k, v in sh["params"].items()},
                "orientation": sh["orientation"], "rotations": {}}
        if sh.get("rms_error") is not None:
            item["rms_error"] = sh["rms_error"]
        for side, model in models.items():
            rots = model.local_rotations(sh["params"])
            item["rotations"][side] = {b: _quat(m) for b, m in rots.items() if not np.allclose(m, np.eye(3))}
        library["handshapes"][name] = item
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(library, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return library


if __name__ == "__main__":
    lib = build_library()
    print(f"{len(lib['handshapes'])} configuraciones -> {WEB_LIBRARY_PATH}")
