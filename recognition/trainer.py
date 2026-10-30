"""Entrenamiento de clasificadores estático (Fase 1) y dinámico (Fase 2) con scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainingReport:
    """Resumen de resultados tras entrenar un clasificador de señas."""

    phase: str
    model_path: Path
    classes: list[str]
    num_samples: int
    accuracy: float
    confusion_matrix: np.ndarray
    classification_summary: str


def _build_classifier(classifier_type: str = "mlp", num_samples: int = 100) -> Pipeline:
    kind = classifier_type.strip().lower()
    if kind == "knn":
        k = max(1, min(5, num_samples // 3))
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=600,
            random_state=42,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _train_single_csv(
    csv_path: Path,
    model_path: Path,
    classifier_type: str,
    alphabet: str,
    phase_name: str,
    base_csv: Path | None = None,
) -> TrainingReport:
    user_exists = csv_path.exists() and csv_path.stat().st_size > 0
    base_exists = base_csv is not None and base_csv.exists() and base_csv.stat().st_size > 0
    if not user_exists and not base_exists:
        raise FileNotFoundError(
            f"No se encontró el archivo de datos '{csv_path}'. "
            "Ejecuta primero: python -m app collect"
        )

    user_df = pd.read_csv(csv_path) if user_exists else None
    base_df = pd.read_csv(base_csv) if base_exists else None
    if user_df is not None and base_df is not None:
        if "label" in user_df.columns and "label" in base_df.columns:
            user_labels = user_df["label"].astype(str).str.upper().unique()
            base_df = base_df[~base_df["label"].astype(str).str.upper().isin(user_labels)]
        df = pd.concat([base_df, user_df], ignore_index=True)
    else:
        df = user_df if user_df is not None else base_df

    if "label" not in df.columns or len(df) < 4:
        raise ValueError(
            f"El dataset '{csv_path}' tiene muy pocas muestras ({len(df)}). "
            "Captura muestras con: python -m app collect"
        )

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).str.upper().to_numpy()

    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Se necesitan al menos 2 letras distintas en '{csv_path}' para entrenar "
            f"(actualmente solo existe: {list(unique_classes)})."
        )

    # División estratificada train/test si cada clase tiene al menos 2 muestras
    if np.all(counts >= 5):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    elif np.all(counts >= 2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    pipeline = _build_classifier(classifier_type, len(X_train))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    labels_sorted = sorted(list(unique_classes))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    summary = classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0)

    # Re-entrenar con el 100% de las muestras antes de persistir el modelo final
    final_pipeline = _build_classifier(classifier_type, len(X))
    final_pipeline.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_pipeline,
        "classes": labels_sorted,
        "alphabet": alphabet,
        "phase": phase_name,
        "feature_dim": X.shape[1],
        "accuracy": acc,
        "num_samples": int(len(X)),
    }
    joblib.dump(payload, model_path)

    return TrainingReport(
        phase=phase_name,
        model_path=model_path,
        classes=labels_sorted,
        num_samples=int(len(X)),
        accuracy=acc,
        confusion_matrix=cm,
        classification_summary=summary,
    )


def format_confusion_matrix(classes: list[str], cm: np.ndarray) -> str:
    """Formatea la matriz de confusión como tabla legible por consola."""
    col_w = max(4, max((len(c) for c in classes), default=3) + 1)
    header = " " * (col_w + 3) + "".join(f"{c:>{col_w}}" for c in classes)
    sep = " " * (col_w + 3) + "-" * (col_w * len(classes))
    rows = [header, sep]
    for idx, cls_name in enumerate(classes):
        row_vals = "".join(f"{int(val):>{col_w}}" for val in cm[idx])
        rows.append(f"{cls_name:>{col_w}} | {row_vals}")
    return "\n".join(rows)


def train_models(
    static_csv: str | Path = "data/landmarks_static.csv",
    dynamic_csv: str | Path = "data/landmarks_dynamic.csv",
    static_model_path: str | Path = "models/sign_classifier.joblib",
    dynamic_model_path: str | Path = "models/sign_dynamic_classifier.joblib",
    classifier_type: str = "mlp",
    alphabet: str = "ASL",
    bootstrap_static_csv: str | Path | None = None,
) -> list[TrainingReport]:
    """
    Entrena el modelo estático (Fase 1) y, si existen suficientes datos dinámicos,
    también entrena el modelo dinámico (Fase 2).
    """
    reports: list[TrainingReport] = []

    static_report = _train_single_csv(
        csv_path=Path(static_csv),
        model_path=Path(static_model_path),
        classifier_type=classifier_type,
        alphabet=alphabet,
        phase_name="Fase 1 (Letras Estáticas)",
        base_csv=Path(bootstrap_static_csv) if bootstrap_static_csv is not None else None,
    )
    reports.append(static_report)

    dyn_path = Path(dynamic_csv)
    if dyn_path.exists() and dyn_path.stat().st_size > 0:
        try:
            dyn_report = _train_single_csv(
                csv_path=dyn_path,
                model_path=Path(dynamic_model_path),
                classifier_type=classifier_type,
                alphabet=alphabet,
                phase_name="Fase 2 (Letras Dinámicas)",
            )
            reports.append(dyn_report)
        except ValueError as exc:
            print(f"[Aviso Fase 2] Se omite entrenamiento dinámico: {exc}")

    return reports
