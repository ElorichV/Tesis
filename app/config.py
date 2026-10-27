"""Carga y validación de `config.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def get_alphabet_spec(cfg: dict[str, Any], alphabet_override: str | None = None) -> dict[str, Any]:
    """Devuelve las listas de letras estáticas y dinámicas del alfabeto activo."""
    alphabet_key = (alphabet_override or cfg.get("alphabet", "ASL")).strip().upper()
    alphabets = cfg.get("alphabets", {})
    if alphabet_key not in alphabets:
        raise ValueError(
            f"Alfabeto '{alphabet_key}' no definido en config.yaml. "
            f"Disponibles: {list(alphabets.keys())}"
        )
    spec = alphabets[alphabet_key]
    return {
        "code": alphabet_key,
        "name": spec.get("name", alphabet_key),
        "static": [str(x).upper() for x in spec.get("static", [])],
        "dynamic": [str(x).upper() for x in spec.get("dynamic", [])],
    }

