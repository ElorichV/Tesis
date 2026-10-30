"""Pruebas deterministas del servidor web unificado y su sesión en vivo."""

from __future__ import annotations

from io import BytesIO
import json
import socket
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

import app.cli as cli_module
import app.web as web_module
from app.cli import _text_buffer_frame_counts
from app.web import (
    RecognitionSession,
    create_server,
    encode_websocket_frame,
    jpeg_dimensions,
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


def raw_http_response(port: int, request: bytes) -> tuple[str, dict[str, str], bytes]:
    with socket.create_connection(("127.0.0.1", port), timeout=3) as connection:
        connection.settimeout(3)
        connection.sendall(request)
        reader = connection.makefile("rb")
        status = reader.readline().decode("iso-8859-1").rstrip()
        headers = {}
        while True:
            line = reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, value = line.decode("iso-8859-1").split(":", 1)
            headers[name.lower()] = value.strip()
        body = reader.read(int(headers.get("content-length", "0")))
        reader.close()
    return status, headers, body


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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            index = response.read().decode("utf-8")
            assert response.status == 200
            assert "AULA_CONFIG" in index
            assert "DEMO_MODE:false" in index
            assert "FRAME_INTERVAL_MS:100" in index
            assert "CAPTURE_WIDTH:640" in index

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


def test_translate_text_truncates_after_stripping_and_caps_request_body(monkeypatch) -> None:
    translated_texts = []
    monkeypatch.setattr(
        web_module,
        "translate_text",
        lambda lexicon, text: (
            translated_texts.append(text) or {"glosa": "HOLA", "secuencia": []}
        ),
    )
    server = create_server(port=0, no_voice=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        raw_text = f" {('hola' * 130)} "
        expected = raw_text.strip()[:500]
        request = Request(
            base + "/traducir-texto",
            data=json.dumps({"texto": raw_text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            translated = json.loads(response.read())
        assert translated["texto_sencillo"] == expected
        assert translated_texts == [expected]

        request_bytes = (
            f"POST /traducir-texto HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.server_port}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 65537\r\n\r\n"
        ).encode("ascii")
        status, _, body = raw_http_response(server.server_port, request_bytes)
        assert status.startswith("HTTP/1.1 413")
        assert json.loads(body)["error"]

        for content_length in ("invalid", "-1"):
            request_bytes = (
                f"POST /traducir-texto HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{server.server_port}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {content_length}\r\n\r\n"
            ).encode("ascii")
            status, _, body = raw_http_response(server.server_port, request_bytes)
            assert status.startswith("HTTP/1.1 400")
            assert json.loads(body)["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_websocket_origin_check_and_shared_speech(monkeypatch) -> None:
    shared_speech = FakeSpeech()
    speech_creations = []
    session_speech = []
    monkeypatch.setattr(
        web_module,
        "_create_speech_output",
        lambda cfg, target_lang, async_playback=True: (
            speech_creations.append((target_lang, async_playback)) or shared_speech
        ),
    )

    class CapturingSession:
        def __init__(self, *args, speech_out=None, **kwargs):
            session_speech.append(speech_out)

        def process_jpeg(self, jpeg):
            pass

        def close(self):
            pass

    monkeypatch.setattr(web_module, "RecognitionSession", CapturingSession)
    server = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port

    def connect(origin: str):
        connection = socket.create_connection(("127.0.0.1", port), timeout=3)
        connection.settimeout(3)
        connection.sendall(
            (
                "GET /reconocer HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                f"Origin: {origin}\r\n\r\n"
            ).encode("ascii")
        )
        reader = connection.makefile("rb")
        status = reader.readline().decode("ascii").strip()
        while reader.readline() not in (b"\r\n", b"\n", b""):
            pass
        return connection, reader, status

    try:
        connection, reader, status = connect(f"http://other.example:{port}")
        assert status.startswith("HTTP/1.1 403")
        reader.close()
        connection.close()

        for _ in range(2):
            connection, reader, status = connect(f"http://127.0.0.1:{port}")
            assert status.startswith("HTTP/1.1 101")
            connection.sendall(masked_client_frame(b"", opcode=0x8))
            assert reader.read(2) == b"\x88\x00"
            reader.close()
            connection.close()

        assert speech_creations == [("es", True)]
        assert session_speech == [shared_speech, shared_speech]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_web_frame_thresholds_scale_from_camera_fps() -> None:
    cfg = {
        "camera": {"fps": 30},
        "text": {
            "stability_frames": 12,
            "release_frames": 4,
            "control_gesture_hold_frames": 20,
        },
    }
    assert _text_buffer_frame_counts(cfg, frame_rate_hz=10) == {
        "stability_frames": 4,
        "release_frames": 1,
        "control_gesture_hold_frames": 7,
    }
    assert _text_buffer_frame_counts(cfg, frame_rate_hz=30) == {
        "stability_frames": 12,
        "release_frames": 4,
        "control_gesture_hold_frames": 20,
    }


def test_dynamic_window_scales_for_web_but_not_run(monkeypatch) -> None:
    dynamic_windows = []

    class Predictor:
        is_ready = True

        def __init__(self, **kwargs):
            dynamic_windows.append(kwargs["dynamic_window_frames"])

    monkeypatch.setattr(cli_module, "SignPredictor", Predictor)
    monkeypatch.setattr(cli_module, "StableTextBuffer", lambda **kwargs: object())
    monkeypatch.setattr(cli_module, "GeminiTextPostProcessor", lambda **kwargs: object())
    cfg = {"camera": {"fps": 30}, "features": {"dynamic_window_frames": 25}}

    cli_module._create_run_components(
        cfg,
        "es",
        frame_rate_hz=10,
        create_speech=False,
    )
    cli_module._create_run_components(cfg, "es", create_speech=False)

    assert dynamic_windows == [8, 25]


class FakeTracker:
    def __init__(self, perceptions):
        self.perceptions = iter(perceptions)
        self.calls = 0

    def process_frame(self, frame, timestamp_ms):
        self.calls += 1
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


class UntrainedFakePredictor(FakePredictor):
    is_ready = False


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


def test_jpeg_dimensions_and_oversized_frame_skip(monkeypatch) -> None:
    encoded = jpeg_frame()
    assert jpeg_dimensions(encoded) == (32, 32)
    assert jpeg_dimensions(b"not a jpeg") is None

    oversized = (
        b"\xff\xd8\xff\xc0\x00\x08\x08"
        + (1081).to_bytes(2, "big")
        + (1920).to_bytes(2, "big")
        + b"\x01"
    )
    assert jpeg_dimensions(oversized) == (1081, 1920)

    tracker = FakeTracker([])
    session = RecognitionSession(
        {"camera": {"mirror_display": False}},
        lang="es",
        tracker=tracker,
        predictor=FakePredictor([]),
        text_buffer=StableTextBuffer(),
        gemini_proc=FakeGemini(),
        speech_out=FakeSpeech(),
    )
    decode_calls = []
    monkeypatch.setattr(
        web_module.cv2,
        "imdecode",
        lambda *args, **kwargs: decode_calls.append(args) or None,
    )
    session.process_jpeg(oversized)
    assert tracker.calls == 0
    assert decode_calls == []


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

    assert events.count({"tipo": "no_reconocido", "motivo": "sin_letras"}) == 1


def test_recognition_session_reports_missing_model_after_empty_hands_down() -> None:
    now = [0.0]
    events = []
    session = RecognitionSession(
        {"camera": {"mirror_display": False}, "web": {"hands_down_seconds": 1.5}},
        lang="es",
        emit=events.append,
        tracker=FakeTracker(
            [perception(True), perception(False), perception(False), perception(False)]
        ),
        predictor=UntrainedFakePredictor([None, None, None, None]),
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

    assert events.count({"tipo": "no_reconocido", "motivo": "sin_modelo"}) == 1


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
    assert {"tipo": "no_reconocido", "motivo": "sin_letras"} in events


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
