"""Pruebas deterministas del servidor web unificado y su sesión en vivo."""

from __future__ import annotations

from io import BytesIO
import json
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from app.web import (
    RecognitionSession,
    create_server,
    encode_websocket_frame,
    read_websocket_frame,
    websocket_accept,
)
from text.buffer import StableTextBuffer


def masked_client_frame(payload: bytes, opcode: int = 2) -> bytes:
    mask = b"\x01\x02\x03\x04"
    if len(payload) < 126:
        header = bytes((0x80 | opcode, 0x80 | len(payload)))
    elif len(payload) <= 0xFFFF:
        header = bytes((0x80 | opcode, 0x80 | 126)) + len(payload).to_bytes(2, "big")
    else:
        header = bytes((0x80 | opcode, 0x80 | 127)) + len(payload).to_bytes(8, "big")
    return header + mask + bytes(value ^ mask[index % 4] for index, value in enumerate(payload))


def test_websocket_frames_and_rfc_handshake() -> None:
    payload = b"mensaje del navegador"
    fin, opcode, decoded = read_websocket_frame(BytesIO(masked_client_frame(payload, opcode=2)))
    assert (fin, opcode, decoded) == (True, 2, payload)

    large = b"x" * 300
    encoded = encode_websocket_frame(large, opcode=1)
    assert read_websocket_frame(BytesIO(encoded))[2] == large
    assert websocket_accept("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_unified_http_routes() -> None:
    server = create_server(port=0)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            index = response.read().decode("utf-8")
            assert response.status == 200
            assert "AULA_CONFIG" in index
            assert "DEMO_MODE:false" in index

        with urlopen(base + "/avatar/js/viewer.js") as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/javascript")

        with urlopen(base + "/salud") as response:
            assert json.loads(response.read()) == {"ok": True}

        request = Request(
            base + "/traducir-texto",
            data=json.dumps({"texto": "hola"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            translated = json.loads(response.read())
            assert translated["secuencia"]
            assert translated["animacion_url"] is None

        empty = Request(
            base + "/traducir-texto",
            data=b'{"texto":"   "}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(empty)
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("El texto vacío debía responder 400.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class FakeTracker:
    def __init__(self, perceptions):
        self.perceptions = iter(perceptions)

    def process_frame(self, frame, timestamp_ms):
        return next(self.perceptions)

    def close(self):
        pass


class FakePredictor:
    is_ready = True

    def __init__(self, letters):
        self.letters = iter(letters)

    def predict(self, landmarks, handedness="Right"):
        letter = next(self.letters)
        return SimpleNamespace(letter=letter, confidence=1.0)


class FakeGemini:
    def clean_text(self, text, lang):
        return text.lower()


class FakeSpeech:
    def __init__(self):
        self.calls = []

    def speak(self, text, lang):
        self.calls.append((text, lang))


def perception(with_hand: bool, face: bool = True):
    hand = SimpleNamespace(
        landmarks=np.zeros((21, 3), dtype=np.float32),
        handedness="Right",
        score=1.0,
    )
    return SimpleNamespace(
        hands=[hand] if with_hand else [],
        primary_hand=hand if with_hand else None,
        active_gestures=set(),
        face=SimpleNamespace(landmarks=np.ones((1, 3)) if face else None),
        pose_landmarks=None,
    )


def jpeg_frame(value: int = 100) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((32, 32, 3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_recognition_session_finalizes_after_hands_down() -> None:
    now = [0.0]
    events = []
    speech = FakeSpeech()
    session = RecognitionSession(
        {"camera": {"mirror_display": True}, "web": {"hands_down_seconds": 1.5}},
        lang="es",
        emit=events.append,
        tracker=FakeTracker([perception(True), perception(True), perception(False), perception(False)]),
        predictor=FakePredictor(["H", "O", None, None]),
        text_buffer=StableTextBuffer(stability_frames=1, release_frames=1, confidence_threshold=0),
        gemini_proc=FakeGemini(),
        speech_out=speech,
        clock=lambda: now[0],
    )
    session.process_jpeg(jpeg_frame())
    now[0] = 0.1
    session.process_jpeg(jpeg_frame())
    now[0] = 1.6
    session.process_jpeg(jpeg_frame())
    now[0] = 3.2
    session.process_jpeg(jpeg_frame())

    assert {"tipo": "manos", "detectadas": True} in events
    assert {"tipo": "parcial", "texto": "H"} in events
    assert {"tipo": "parcial", "texto": "HO"} in events
    assert any(event["tipo"] == "calidad" for event in events)
    assert {"tipo": "final", "texto": "ho"} in events
    assert speech.calls == [("ho", "es")]


def test_recognition_session_reports_unrecognized_after_empty_hands_down() -> None:
    now = [0.0]
    events = []
    session = RecognitionSession(
        {"camera": {"mirror_display": False}, "web": {"hands_down_seconds": 1.5}},
        lang="es",
        emit=events.append,
        tracker=FakeTracker(
            [perception(True), perception(False), perception(False), perception(False)]
        ),
        predictor=FakePredictor([None, None, None, None]),
        text_buffer=StableTextBuffer(
            stability_frames=1,
            release_frames=1,
            confidence_threshold=0,
        ),
        gemini_proc=FakeGemini(),
        speech_out=FakeSpeech(),
        clock=lambda: now[0],
    )

    for timestamp in (0.0, 0.1, 1.6, 3.2):
        now[0] = timestamp
        session.process_jpeg(jpeg_frame())

    assert events.count({"tipo": "no_reconocido"}) == 1


def test_recognition_session_speak_with_empty_buffer() -> None:
    now = [0.0]
    events = []
    buffer = StableTextBuffer(
        stability_frames=1,
        release_frames=1,
        confidence_threshold=0,
        control_gesture_hold_frames=1,
        gestures_config={"speak": "brow_up", "space": "none", "backspace": "none"},
    )
    hand_perception = perception(True)
    hand_perception.active_gestures = {"brow_up"}
    session = RecognitionSession(
        {"camera": {"mirror_display": False}},
        lang="es",
        emit=events.append,
        tracker=FakeTracker([hand_perception, hand_perception]),
        predictor=FakePredictor([None, None]),
        text_buffer=buffer,
        gemini_proc=FakeGemini(),
        speech_out=FakeSpeech(),
        clock=lambda: now[0],
    )
    session.process_jpeg(jpeg_frame())
    session.process_jpeg(jpeg_frame())
    assert {"tipo": "no_reconocido"} in events


def test_recognition_session_ignores_speak_without_hands_seen() -> None:
    now = [0.0]
    events = []
    brow_up = perception(False)
    brow_up.active_gestures = {"brow_up"}
    session = RecognitionSession(
        {"camera": {"mirror_display": False}},
        lang="es",
        emit=events.append,
        tracker=FakeTracker([brow_up, brow_up]),
        predictor=FakePredictor([None, None]),
        text_buffer=StableTextBuffer(
            stability_frames=1,
            release_frames=1,
            confidence_threshold=0,
            control_gesture_hold_frames=1,
            gestures_config={"speak": "brow_up", "space": "none", "backspace": "none"},
        ),
        gemini_proc=FakeGemini(),
        speech_out=FakeSpeech(),
        clock=lambda: now[0],
    )

    session.process_jpeg(jpeg_frame())
    now[0] = 0.1
    session.process_jpeg(jpeg_frame())

    assert not any(event["tipo"] in {"no_reconocido", "final"} for event in events)
