"""Renderizado de overlays en vivo con OpenCV (manos, cara+blendshapes, brazos y HUD)."""

from __future__ import annotations

import cv2
import numpy as np

from perception.mediapipe_tracker import (
    FACE_OVAL_INDICES,
    HAND_CONNECTIONS,
    LEFT_EYEBROW_INDICES,
    LIPS_OUTER_INDICES,
    POSE_ARM_CONNECTIONS,
    POSE_ARM_INDICES,
    RIGHT_EYEBROW_INDICES,
    PerceptionResult,
)
from recognition.predictor import PredictionOutput
from text.buffer import BufferUpdateResult


def _draw_polyline(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    indices: tuple[int, ...],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    h, w = canvas.shape[:2]
    pts = []
    for idx in indices:
        if idx < len(landmarks):
            x = int(landmarks[idx, 0] * w)
            y = int(landmarks[idx, 1] * h)
            pts.append([x, y])
    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            canvas, [arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA
        )


def draw_perception_overlay(canvas: np.ndarray, perception: PerceptionResult) -> np.ndarray:
    """
    Dibuja sobre el frame:
    - Pose de hombros y brazos.
    - Contorno facial, cejas y labios + panel de blendshapes faciales.
    - Los 21 puntos y conexiones de cada mano detectada.
    """
    h, w = canvas.shape[:2]

    # 1. Pose: hombros, codos y muñecas
    if perception.pose_landmarks is not None:
        plm = perception.pose_landmarks
        for i_start, i_end in POSE_ARM_CONNECTIONS:
            if i_start < len(plm) and i_end < len(plm):
                if plm[i_start, 3] > 0.35 and plm[i_end, 3] > 0.35:
                    pt1 = (int(plm[i_start, 0] * w), int(plm[i_start, 1] * h))
                    pt2 = (int(plm[i_end, 0] * w), int(plm[i_end, 1] * h))
                    cv2.line(canvas, pt1, pt2, (255, 170, 30), 3, cv2.LINE_AA)
        for idx in POSE_ARM_INDICES:
            if idx < len(plm) and plm[idx, 3] > 0.35:
                pt = (int(plm[idx, 0] * w), int(plm[idx, 1] * h))
                cv2.circle(canvas, pt, 5, (0, 215, 255), -1, cv2.LINE_AA)

    # 2. Cara: contorno, cejas y boca
    if perception.face.landmarks is not None:
        flm = perception.face.landmarks
        _draw_polyline(canvas, flm, FACE_OVAL_INDICES, (180, 220, 180), 1)
        _draw_polyline(canvas, flm, LEFT_EYEBROW_INDICES, (80, 230, 255), 2)
        _draw_polyline(canvas, flm, RIGHT_EYEBROW_INDICES, (80, 230, 255), 2)
        _draw_polyline(canvas, flm, LIPS_OUTER_INDICES, (150, 150, 255), 1)

    # 3. Manos: 21 puntos y conexiones articulares
    for hand in perception.hands:
        hlm = hand.landmarks
        pts_px = [(int(p[0] * w), int(p[1] * h)) for p in hlm]
        line_color = (60, 230, 120) if hand.handedness.lower() == "right" else (255, 190, 70)

        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts_px[a], pts_px[b], line_color, 2, cv2.LINE_AA)

        for idx, pt in enumerate(pts_px):
            radius = 5 if idx in (0, 4, 8, 12, 16, 20) else 3
            dot_color = (0, 80, 255) if idx in (4, 8, 12, 16, 20) else (255, 255, 255)
            cv2.circle(canvas, pt, radius, dot_color, -1, cv2.LINE_AA)

        wrist_x, wrist_y = pts_px[0]
        cv2.putText(
            canvas,
            f"{hand.handedness} ({hand.score:.0%})",
            (max(5, wrist_x - 30), min(h - 10, wrist_y + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            line_color,
            1,
            cv2.LINE_AA,
        )

    # 4. Panel compacto de Blendshapes faciales (esquina superior derecha)
    if perception.face.blendshapes:
        panel_x = max(10, w - 205)
        panel_y = 10
        cv2.rectangle(canvas, (panel_x, panel_y), (w - 10, panel_y + 125), (25, 25, 25), -1)
        cv2.putText(
            canvas,
            "Gestos Faciales",
            (panel_x + 8, panel_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        labels = [
            ("smile", "Sonrisa"),
            ("brow_up", "Cejas ^"),
            ("brow_down", "Cejas v"),
            ("mouth_open", "Boca O"),
            ("blink", "Parpadeo"),
        ]
        for idx, (key, title) in enumerate(labels):
            val = float(perception.face.blendshapes.get(key, 0.0))
            y_row = panel_y + 36 + idx * 18
            active = key in perception.active_gestures
            bar_color = (50, 220, 100) if active else (110, 150, 190)
            cv2.putText(
                canvas,
                f"{title}:",
                (panel_x + 8, y_row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )
            bar_x = panel_x + 82
            bar_w = int(max(0.0, min(1.0, val)) * 95)
            cv2.rectangle(canvas, (bar_x, y_row - 9), (bar_x + 95, y_row - 1), (60, 60, 60), -1)
            cv2.rectangle(canvas, (bar_x, y_row - 9), (bar_x + bar_w, y_row - 1), bar_color, -1)

    return canvas


def draw_run_hud(
    canvas: np.ndarray,
    fps: float,
    prediction: PredictionOutput,
    buffer_state: BufferUpdateResult,
    alphabet_code: str,
    last_spoken: str = "",
) -> np.ndarray:
    """Dibuja el HUD de traducción en vivo (FPS, letra/confianza, barras de estabilidad y texto)."""
    h, w = canvas.shape[:2]

    # Caja superior izquierda: FPS, alfabeto y predicción
    cv2.rectangle(canvas, (10, 10), (355, 118), (25, 25, 25), -1)
    fps_color = (50, 230, 100) if fps >= 15.0 else (0, 165, 255)
    cv2.putText(
        canvas,
        f"FPS: {fps:4.1f} | Alfabeto: {alphabet_code}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    if prediction.mode == "untrained":
        status_line = "Modelo NO entrenado (usa collect/train)"
        status_color = (60, 130, 255)
    elif prediction.mode == "no_hand":
        status_line = "Mano no detectada"
        status_color = (180, 180, 180)
    else:
        mode_tag = "DIN" if prediction.mode == "dynamic" else "EST"
        status_line = f"Letra: {prediction.letter or '-'} ({prediction.confidence:.0%}) [{mode_tag}]"
        status_color = (80, 245, 120) if prediction.confidence >= 0.70 else (0, 210, 255)

    cv2.putText(
        canvas,
        status_line,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Barra de estabilidad de la letra candidata
    prog_w = int(buffer_state.candidate_progress * 230)
    cv2.putText(
        canvas,
        "Estab.:",
        (20, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (85, 75), (315, 88), (65, 65, 65), -1)
    cv2.rectangle(canvas, (85, 75), (85 + prog_w, 88), (50, 220, 120), -1)

    if buffer_state.active_control_gesture:
        g_w = int(buffer_state.gesture_progress * 230)
        cv2.putText(
            canvas,
            f"Gesto ({buffer_state.active_control_gesture}):",
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (100, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (150, 98), (315, 109), (65, 65, 65), -1)
        cv2.rectangle(
            canvas,
            (150, 98),
            (150 + int(buffer_state.gesture_progress * 165), 109),
            (80, 200, 255),
            -1,
        )

    # Barra inferior con texto acumulado y controles
    cv2.rectangle(canvas, (0, h - 82), (w, h), (20, 20, 20), -1)
    display_text = buffer_state.current_text[-42:] if buffer_state.current_text else "( deletrea frente a la camara )"
    cv2.putText(
        canvas,
        f"Texto: {display_text}",
        (16, h - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    help_line = "[Espacio]=Espacio  [Backspace]=Borrar  [Enter]=Hablar  [C]=Limpiar  [Q]=Salir"
    if last_spoken:
        help_line = f"Voz: '{last_spoken[:32]}' | [Enter]=Hablar [Backspace]=Borrar [C]=Limpiar [Q]=Salir"
    cv2.putText(
        canvas,
        help_line,
        (16, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 200, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_collect_hud(
    canvas: np.ndarray,
    fps: float,
    alphabet_code: str,
    target_letter: str,
    is_dynamic: bool,
    recording: bool,
    captured_count: int,
    target_count: int,
    guidance_text: str,
    hand_detected: bool,
) -> np.ndarray:
    """Dibuja el HUD interactivo para captura de dataset (modo collect)."""
    h, w = canvas.shape[:2]

    cv2.rectangle(canvas, (10, 10), (420, 115), (25, 25, 25), -1)
    kind_str = "DINAMICA (ventana 25 frames)" if is_dynamic else "ESTATICA (1 frame/muestra)"
    cv2.putText(
        canvas,
        f"FPS: {fps:4.1f} | {alphabet_code} | Letra: [{target_letter}] ({kind_str})",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 240, 140),
        1,
        cv2.LINE_AA,
    )

    state_txt = "CAPTURANDO..." if recording else "EN ESPERA (Pulsa ESPACIO para iniciar)"
    state_col = (50, 100, 255) if recording else (220, 220, 100)
    cv2.putText(
        canvas,
        f"{state_txt}  [{captured_count}/{target_count}]",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        state_col,
        2,
        cv2.LINE_AA,
    )

    hand_msg = "Mano detectada: SI" if hand_detected else "Mano detectada: NO (muestra tu mano)"
    hand_col = (80, 235, 80) if hand_detected else (80, 80, 255)
    cv2.putText(
        canvas,
        hand_msg,
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        hand_col,
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(canvas, (0, h - 75), (w, h), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        guidance_text,
        (16, h - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (100, 230, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "[A-Z]=Elegir letra  [Tab]=Sig. letra  [Espacio]=Capturar lote  [Q/ESC]=Salir",
        (16, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return canvas

