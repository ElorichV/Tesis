"""Gestión de cámara con soporte para DirectShow (Windows) y contador de FPS."""

from __future__ import annotations

from collections import deque
import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Error cuando la cámara web no está disponible o no entrega frames."""


class FPSCounter:
    """Calcula FPS suavizado sobre una ventana deslizante de instantes de tiempo."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=max(2, window_size))

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 1e-9:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class CameraStream:
    """
    Envuelve `cv2.VideoCapture` probando `cv2.CAP_DSHOW` en Windows (configurable)
    con fallback automático al backend por defecto si falla.
    Nunca almacena video en disco.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        use_dshow_on_windows: bool = True,
        mirror_display: bool = True,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.use_dshow_on_windows = use_dshow_on_windows
        self.mirror_display = mirror_display
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Abre la cámara web verificando que entregue al menos un frame válido."""
        backends: list[tuple[str, int]] = []
        if os.name == "nt" and self.use_dshow_on_windows:
            backends.append(("CAP_DSHOW", cv2.CAP_DSHOW))
        backends.append(("CAP_ANY", cv2.CAP_ANY))

        for backend_name, backend_flag in backends:
            cap = cv2.VideoCapture(self.device_index, backend_flag)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                logger.info(
                    "Cámara %d abierta correctamente con backend %s (%dx%d).",
                    self.device_index,
                    backend_name,
                    frame.shape[1],
                    frame.shape[0],
                )
                self._cap = cap
                return

            cap.release()

        raise CameraError(
            f"No se pudo abrir la cámara web (índice {self.device_index}). "
            "Verifica que la webcam esté conectada, que ninguna otra aplicación la esté usando "
            "y que los permisos de cámara de Windows estén habilitados (Configuración -> "
            "Privacidad y seguridad -> Cámara)."
        )

    def read(self) -> np.ndarray:
        """Lee un frame BGR de la cámara (espejado si `mirror_display=True`)."""
        if self._cap is None or not self._cap.isOpened():
            raise CameraError("La cámara no está abierta.")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("No se pudo leer el frame de la cámara web.")

        if self.mirror_display:
            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __enter__(self) -> CameraStream:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

