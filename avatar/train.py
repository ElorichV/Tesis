"""
Entrenamiento de configuraciones manuales del avatar a partir de landmarks reales.

Para cada letra/configuración, ajusta por mínimos cuadrados (scipy) los ángulos articulares
de la mano del avatar y su orientación global para que los 21 puntos de su cinemática
directa coincidan con los landmarks MediaPipe observados. Agrega las soluciones por letra
(mediana de ángulos + promedio de cuaterniones) y guarda `data/handshapes_trained.json`.

Fuentes de datos soportadas:
- CSV propio del repositorio (`python -m app collect`): `data/landmarks_static.csv`.
- Dataset público "ASL Static Landmarks v1" (CC-BY-4.0, Hugging Face: Siruyy/asl-static-landmarks-v1).

Uso:
    python -m avatar.train --hf-asl --csv data/landmarks_static.csv --alphabet LSM
"""

from __future__ import annotations

import argparse
import json
import string
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from avatar.kinematics import (
    DATA_DIR,
    JOINT_LIMITS,
    PARAM_NAMES,
    HandModel,
    mediapipe_to_avatar,
    normalize_points,
    rotvec_matrix,
)

HF_ASL_REPO = "Siruyy/asl-static-landmarks-v1"
HF_ASL_URL = "https://huggingface.co/datasets/" + HF_ASL_REPO + "/resolve/main/{name}"
OUTPUT_PATH = DATA_DIR / "handshapes_trained.json"
REG_WEIGHT = 0.004


def _rotvec_from_matrix(m: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(m) - 1) / 2, -1, 1)
    ang = np.arccos(cos)
    if ang < 1e-8:
        return np.zeros(3)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    if np.linalg.norm(v) < 1e-8:
        w, vecs = np.linalg.eigh((m + np.eye(3)) / 2)
        axis = vecs[:, np.argmax(w)]
        return axis * ang
    return v / np.linalg.norm(v) * ang


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Cuaternión (x, y, z, w) de una matriz de rotación."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def average_quaternions(qs: np.ndarray) -> np.ndarray:
    """Promedio de cuaterniones por el método de Markley (autovector principal)."""
    acc = np.zeros((4, 4))
    for q in qs:
        acc += np.outer(q, q)
    w, v = np.linalg.eigh(acc)
    q = v[:, np.argmax(w)]
    return q if q[3] >= 0 else -q


class HandshapeFitter:
    def __init__(self, model: HandModel | None = None, limits: str = "hyper") -> None:
        self.model = model or HandModel("R")
        lim = JOINT_LIMITS[limits]
        # Parámetros: ángulos articulares + rotación global (3) + relación de aspecto x/y de la cámara (1)
        self.lo = np.array([lim[k][0] for k in PARAM_NAMES] + [-np.pi * 2] * 3 + [0.5])
        self.hi = np.array([lim[k][1] for k in PARAM_NAMES] + [np.pi * 2] * 3 + [1.5])
        self.n_params = len(PARAM_NAMES)
        rng = np.random.default_rng(7)
        self.orientation_seeds = [self._seed_palm_forward()] + [rng.normal(size=3) for _ in range(3)]
        fist = {k: 0.0 for k in PARAM_NAMES}
        for k in PARAM_NAMES:
            if k.endswith("_mcp") and not k.startswith("thumb"):
                fist[k] = 80.0
            elif k.endswith("_pip"):
                fist[k] = 90.0
            elif k.endswith("_dip"):
                fist[k] = 45.0
        self.pose_seeds = [np.zeros(self.n_params), np.array([fist[k] for k in PARAM_NAMES])]

    def _seed_palm_forward(self) -> np.ndarray:
        # Reposo (dedos hacia -X, palma hacia -Y) -> dedos arriba, palma hacia el espectador.
        target = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float).T
        return _rotvec_from_matrix(target)

    def unpack(self, x: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        params = {k: float(v) for k, v in zip(PARAM_NAMES, x[: self.n_params])}
        return params, x[self.n_params:self.n_params + 3]

    def predict(self, x: np.ndarray) -> np.ndarray:
        params, rv = self.unpack(x)
        pts = self.model.forward(params, rotvec_matrix(rv))
        pts[:, 0] *= x[-1]
        return normalize_points(pts)

    def residuals(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        pts = self.predict(x)
        reg = REG_WEIGHT * x[: self.n_params]
        return np.concatenate([(pts - target).ravel(), reg])

    def fit(self, target: np.ndarray, inits: list[np.ndarray] | None = None,
            full_search: bool = True) -> tuple[np.ndarray, float]:
        target = normalize_points(target)
        starts = list(inits or [])
        if full_search or not starts:
            for pose in self.pose_seeds:
                for rv in self.orientation_seeds:
                    starts.append(np.concatenate([pose, rv, [1.0]]))
        best, best_cost = None, np.inf
        for x0 in starts:
            x0 = np.clip(x0, self.lo + 1e-6, self.hi - 1e-6)
            res = least_squares(self.residuals, x0, bounds=(self.lo, self.hi), args=(target,),
                                x_scale=np.concatenate([np.full(self.n_params, 30.0), np.ones(3), [0.2]]),
                                max_nfev=300)
            if res.cost < best_cost:
                best, best_cost = res.x, res.cost
        pts = self.predict(best)
        rms = float(np.sqrt(np.mean(np.sum((pts - target) ** 2, axis=1))))
        return best, rms


def load_hf_asl(cache_dir: Path) -> dict[str, list[np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in ("X_train.npy", "y_train.npy", "mean.npy", "std.npy"):
        path = cache_dir / name
        if not path.exists():
            print(f"[train] Descargando {HF_ASL_REPO}/{name} ...")
            urllib.request.urlretrieve(HF_ASL_URL.format(name=name), path)
        arrays[name] = np.load(path, allow_pickle=True)
    X = arrays["X_train.npy"] * arrays["std.npy"] + arrays["mean.npy"]
    y = arrays["y_train.npy"].astype(int)
    letters = string.ascii_uppercase
    out: dict[str, list[np.ndarray]] = {}
    for row, label in zip(X, y):
        out.setdefault(letters[label], []).append(row[:63].reshape(21, 3))
    return out


def load_repo_csv(path: Path) -> dict[str, list[np.ndarray]]:
    import pandas as pd

    df = pd.read_csv(path)
    feats = [c for c in df.columns if c != "label"]
    out: dict[str, list[np.ndarray]] = {}
    for label, grp in df.groupby("label"):
        out[str(label).upper()] = [r.reshape(21, 3) for r in grp[feats].to_numpy(dtype=float)]
    return out


def choose_chirality(fitter: HandshapeFitter, samples: dict[str, list[np.ndarray]]) -> bool:
    """Prueba si los datos corresponden a una mano derecha espejada (cámara tipo selfie)."""
    errs = {False: [], True: []}
    for label in sorted(samples)[:: max(1, len(samples) // 6)]:
        pts = samples[label][0]
        for mirror in (False, True):
            errs[mirror].append(fitter.fit(mediapipe_to_avatar(pts, mirror))[1])
    mirror = float(np.mean(errs[True])) < float(np.mean(errs[False]))
    print(f"[train] RMS sin espejo={np.mean(errs[False]):.4f}  con espejo={np.mean(errs[True]):.4f}"
          f" -> espejo={'sí' if mirror else 'no'}")
    return mirror


def train_source(fitter: HandshapeFitter, samples: dict[str, list[np.ndarray]], source: str,
                 per_class: int, mirror: bool | None = None) -> dict[str, dict]:
    if mirror is None:
        mirror = choose_chirality(fitter, samples)
    rng = np.random.default_rng(0)
    results = {}
    for label in sorted(samples):
        rows = samples[label]
        pick = rng.choice(len(rows), size=min(per_class, len(rows)), replace=False)
        sols, errs = [], []
        prev = None
        t0 = time.time()
        for n_done, i in enumerate(pick):
            target = mediapipe_to_avatar(rows[i], mirror)
            x, rms = fitter.fit(target, inits=[prev] if prev is not None else None,
                                full_search=n_done < 3)
            sols.append(x)
            errs.append(rms)
            prev = x
        sols = np.array(sols)
        keep = np.array(errs) <= np.percentile(errs, 75)
        angles = np.median(sols[keep, : fitter.n_params], axis=0)
        quats = np.array([_matrix_to_quat(rotvec_matrix(rv))
                          for rv in sols[keep, fitter.n_params:fitter.n_params + 3]])
        q = average_quaternions(quats)
        results[label] = {
            "params": {k: round(float(v), 2) for k, v in zip(PARAM_NAMES, angles)},
            "orientation": [round(float(c), 5) for c in q],
            "source": source,
            "samples_fitted": int(len(pick)),
            "rms_error": round(float(np.median(errs)), 4),
        }
        print(f"[train] {source:>14} {label}: {len(pick)} muestras, RMS mediano={np.median(errs):.4f}"
              f" ({time.time() - t0:.1f}s)")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ajusta configuraciones manuales del avatar a landmarks reales")
    ap.add_argument("--csv", default=None, help="CSV del repositorio (python -m app collect)")
    ap.add_argument("--csv-alphabet", default="LSM", help="Alfabeto de las letras del CSV propio")
    ap.add_argument("--hf-asl", action="store_true", help="Usar el dataset público ASL Static Landmarks v1")
    ap.add_argument("--per-class", type=int, default=24, help="Muestras ajustadas por letra")
    ap.add_argument("--cache-dir", default="data/external/asl_static_landmarks_v1")
    ap.add_argument("--out", default=str(OUTPUT_PATH))
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    result = {"description": existing.get("description",
                                          "Configuraciones manuales ajustadas a landmarks MediaPipe reales"),
              "alphabets": existing.get("alphabets", {})}
    fitter = HandshapeFitter()

    if not args.csv and not args.hf_asl:
        ap.error("Indica al menos una fuente: --csv y/o --hf-asl")
    if args.hf_asl:
        samples = load_hf_asl(Path(args.cache_dir))
        result["alphabets"]["ASL"] = train_source(fitter, samples, "hf:" + HF_ASL_REPO, args.per_class)
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[train] No existe {csv_path}; captura muestras con `python -m app collect`.")
        else:
            samples = load_repo_csv(csv_path)
            result["alphabets"][args.csv_alphabet.upper()] = train_source(
                fitter, samples, f"csv:{csv_path.name}", args.per_class)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[train] Guardado {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
