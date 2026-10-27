"""Pruebas unitarias para el buffer de estabilidad de texto y post-procesamiento."""

from __future__ import annotations

from text.buffer import StableTextBuffer
from text.gemini_postprocessor import GeminiTextPostProcessor


def test_letter_accepted_only_after_stability_frames() -> None:
    buf = StableTextBuffer(stability_frames=5, confidence_threshold=0.75, release_frames=3)

    for i in range(4):
        res = buf.update("A", confidence=0.90)
        assert res.committed_letter is None
        assert buf.text == ""
        assert res.candidate_letter == "A"
        assert res.candidate_progress == (i + 1) / 5

    res5 = buf.update("A", confidence=0.90)
    assert res5.committed_letter == "A"
    assert buf.text == "A"


def test_low_confidence_does_not_accumulate() -> None:
    buf = StableTextBuffer(stability_frames=4, confidence_threshold=0.80, release_frames=2)

    for _ in range(3):
        buf.update("B", confidence=0.85)
    # Un frame con baja confianza reinicia la racha
    res_low = buf.update("B", confidence=0.50)
    assert res_low.committed_letter is None
    assert res_low.candidate_letter is None

    # No debe aceptar en el siguiente frame aislado
    res_next = buf.update("B", confidence=0.90)
    assert res_next.committed_letter is None
    assert buf.text == ""


def test_anti_repetition_requires_release_or_change() -> None:
    buf = StableTextBuffer(stability_frames=3, confidence_threshold=0.70, release_frames=3)

    # Acepta "L" en 3 frames
    for _ in range(3):
        buf.update("L", confidence=0.95)
    assert buf.text == "L"

    # Mantener "L" durante 20 frames adicionales NO debe repetir la letra
    for _ in range(20):
        res = buf.update("L", confidence=0.95)
        assert res.committed_letter is None
    assert buf.text == "L"

    # Retirar la mano durante `release_frames` (3 frames) desbloquea la repetición
    for _ in range(3):
        buf.update(None, confidence=0.0)
    assert buf.locked_letter is None

    # Ahora sí permite una segunda "L" ("LL")
    for _ in range(3):
        buf.update("L", confidence=0.95)
    assert buf.text == "LL"


def test_transition_to_different_letter_immediately_accumulates() -> None:
    buf = StableTextBuffer(stability_frames=3, confidence_threshold=0.70, release_frames=2)

    for ch in ("H", "O", "L", "A"):
        for _ in range(3):
            buf.update(ch, confidence=0.88)

    assert buf.text == "HOLA"


def test_control_gestures_space_backspace_and_speak() -> None:
    buf = StableTextBuffer(
        stability_frames=2,
        confidence_threshold=0.70,
        release_frames=2,
        control_gesture_hold_frames=3,
        gestures_config={
            "space": "mouth_open",
            "backspace": "brow_down",
            "speak": "brow_up",
        },
    )

    for ch in ("H", "I"):
        buf.update(ch, 0.9)
        buf.update(ch, 0.9)
    assert buf.text == "HI"

    # Gesto de espacio sostenido 3 frames
    for _ in range(3):
        res = buf.update(None, 0.0, active_gestures={"mouth_open"})
    assert res.triggered_action == "space"
    assert buf.text == "HI "

    # Gesto de borrar sostenido 3 frames
    for _ in range(3):
        res = buf.update(None, 0.0, active_gestures={"brow_down"})
    assert res.triggered_action == "backspace"
    assert buf.text == "HI"

    # Gesto de hablar sostenido 3 frames
    for _ in range(3):
        res = buf.update(None, 0.0, active_gestures={"brow_up"})
    assert res.triggered_action == "speak"


def test_gemini_postprocessor_fallback_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    proc = GeminiTextPostProcessor(enabled=True)
    assert not proc.is_available
    assert proc.clean_text("H O L A") == "hola"
    assert proc.clean_text("HOLA MUNDO") == "hola mundo"

