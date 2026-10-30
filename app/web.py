"""Servidor HTTP local para la aplicación web unificada."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from app.cli import _create_run_components, _create_speech_output, _create_tracker
from app.config import get_alphabet_spec, load_config
from avatar.lexicon import Lexicon
from avatar.server import ensure_handshape_library, list_lexicon, translate_text

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
AVATAR_WEB_DIR = Path(__file__).resolve().parent.parent / "avatar" / "web"
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_WEBSOCKET_MESSAGE = 2 * 1024 * 1024
MAX_JPEG_PIXELS = 1920 * 1080
LOGGER = logging.getLogger(__name__)
_warning_lock = threading.Lock()
_untrained_warning_emitted = False


def websocket_accept(key: str) -> str:
    """Calcula el valor Sec-WebSocket-Accept definido por RFC 6455."""
    digest = hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_websocket_frame(payload: bytes, opcode: int = 2, fin: bool = True) -> bytes:
    """Codifica un frame servidor no enmascarado."""
    first = (0x80 if fin else 0) | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        header = bytes((first, length))
    elif length <= 0xFFFF:
        header = bytes((first, 126)) + length.to_bytes(2, "big")
    else:
        header = bytes((first, 127)) + length.to_bytes(8, "big")
    return header + payload


def _read_exact(stream, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = stream.read(length - len(data))
        if not chunk:
            raise EOFError("Conexión WebSocket cerrada.")
        data.extend(chunk)
    return bytes(data)


def read_websocket_frame(stream) -> tuple[bool, int, bytes]:
    """Lee un frame WebSocket, decodificando máscaras y longitudes extendidas."""
    first, second = _read_exact(stream, 2)
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = int.from_bytes(_read_exact(stream, 2), "big")
    elif length == 127:
        length = int.from_bytes(_read_exact(stream, 8), "big")
        if length & (1 << 63):
            raise ValueError("Longitud WebSocket inválida.")
    if length > MAX_WEBSOCKET_MESSAGE:
        raise ValueError("Frame WebSocket demasiado grande.")
    mask = _read_exact(stream, 4) if masked else None
    payload = _read_exact(stream, length)
    if mask:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return fin, opcode, payload


def read_websocket_message(stream, send_frame: Callable[[bytes, int], None]) -> tuple[int, bytes] | None:
    """Lee mensajes completos, atendiendo ping/close durante fragmentaciones."""
    fragments: list[bytes] = []
    fragmented_opcode: int | None = None
    while True:
        fin, opcode, payload = read_websocket_frame(stream)
        if opcode == 0x8:
            send_frame(payload, 0x8)
            return None
        if opcode == 0x9:
            send_frame(payload, 0xA)
            continue
        if opcode == 0xA:
            continue
        if opcode in (0x1, 0x2):
            if fragmented_opcode is not None:
                raise ValueError("Se recibió un mensaje nuevo antes de completar la fragmentación.")
            if fin:
                return opcode, payload
            fragmented_opcode = opcode
            fragments = [payload]
        elif opcode == 0x0:
            if fragmented_opcode is None:
                raise ValueError("Frame de continuación inesperado.")
            fragments.append(payload)
            if sum(map(len, fragments)) > MAX_WEBSOCKET_MESSAGE:
                raise ValueError("Mensaje WebSocket demasiado grande.")
            if fin:
                return fragmented_opcode, b"".join(fragments)
        else:
            raise ValueError(f"Opcode WebSocket no soportado: {opcode}.")


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Devuelve (alto, ancho) desde el primer marcador JPEG SOF válido."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None

    index = 2
    while index < len(data):
        if data[index] != 0xFF:
            return None
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            return None
        marker = data[index]
        index += 1

        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA or index + 2 > len(data):
            return None

        segment_length = int.from_bytes(data[index:index + 2], "big")
        if segment_length < 2 or index + segment_length > len(data):
            return None
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if segment_length < 7:
                return None
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return (height, width) if height and width else None
        index += segment_length
    return None


class RecognitionSession:
    """Pipeline aislado de reconocimiento para una conexión WebSocket."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        lang: str,
        no_voice: bool = False,
        emit: Callable[[dict[str, Any]], None] | None = None,
        tracker: Any | None = None,
        predictor: Any | None = None,
        text_buffer: Any | None = None,
        gemini_proc: Any | None = None,
        speech_out: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        hands_down_seconds: float | None = None,
    ) -> None:
        if not no_voice and speech_out is None:
            raise ValueError("RecognitionSession requiere SpeechOutput compartido o no_voice=True.")
        self.cfg = cfg
        self.lang = lang
        self.no_voice = no_voice
        self.emit = emit or (lambda event: None)
        self.clock = clock
        self.tracker = tracker or _create_tracker(cfg)
        self.predictor = predictor
        self.text_buffer = text_buffer
        self.gemini_proc = gemini_proc
        self.speech_out = speech_out
        web_cfg = cfg.get("web", {})
        frame_rate_hz = float(web_cfg.get("frame_rate_hz", 10))
        if any(value is None for value in (predictor, text_buffer, gemini_proc)):
            components = _create_run_components(
                cfg,
                lang,
                async_playback=True,
                warn_untrained=False,
                frame_rate_hz=frame_rate_hz,
                create_speech=False,
            )
            self.predictor = predictor if predictor is not None else components[0]
            self.text_buffer = text_buffer if text_buffer is not None else components[1]
            self.gemini_proc = gemini_proc if gemini_proc is not None else components[2]

        global _untrained_warning_emitted
        with _warning_lock:
            if not getattr(self.predictor, "is_ready", True) and not _untrained_warning_emitted:
                LOGGER.warning(
                    "[AVISO] Aún no existe un modelo entrenado en 'models/'. "
                    "El sistema mostrará los landmarks en vivo, pero para reconocer letras "
                    "ejecuta primero: `python -m app collect` y `python -m app train`."
                )
                _untrained_warning_emitted = True

        self.hands_down_seconds = (
            float(hands_down_seconds)
            if hands_down_seconds is not None
            else float(web_cfg.get("hands_down_seconds", 1.5))
        )
        self.mirror_frames = bool(cfg.get("camera", {}).get("mirror_display", True))
        self.last_timestamp_ms = -1
        self.started_at = clock()
        self.last_no_hand_at: float | None = None
        self.hands_detected = False
        self.hands_seen_since_final = False
        self.jpeg_warning_emitted = False
        self.last_buffer_text = self.text_buffer.text
        self.last_quality: tuple[bool, bool] | None = None
        self.last_quality_at: float | None = None

    def _emit(self, tipo: str, **fields: Any) -> None:
        self.emit({"tipo": tipo, **fields})

    def _warn_invalid_jpeg(self) -> None:
        if not self.jpeg_warning_emitted:
            LOGGER.warning("Se omite un frame JPEG inválido o con resolución excesiva.")
            self.jpeg_warning_emitted = True

    def process_jpeg(self, jpeg: bytes) -> None:
        """Decodifica y procesa una imagen JPEG recibida del navegador."""
        try:
            dimensions = jpeg_dimensions(jpeg)
            if (
                len(jpeg) > MAX_WEBSOCKET_MESSAGE
                or dimensions is None
                or dimensions[0] * dimensions[1] > MAX_JPEG_PIXELS
            ):
                self._warn_invalid_jpeg()
                return
            encoded = np.frombuffer(jpeg, dtype=np.uint8)
            try:
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            except Exception:
                self._warn_invalid_jpeg()
                return
            if frame is None:
                self._warn_invalid_jpeg()
                return
            if self.mirror_frames:
                frame = cv2.flip(frame, 1)
            now = self.clock()
            timestamp_ms = max(self.last_timestamp_ms + 1, int((now - self.started_at) * 1000))
            self.last_timestamp_ms = timestamp_ms
            perception = self.tracker.process_frame(frame, timestamp_ms=timestamp_ms)
            primary = perception.primary_hand
            prediction = (
                self.predictor.predict(primary.landmarks, handedness=primary.handedness)
                if primary is not None
                else self.predictor.predict(None)
            )
            buffer_state = self.text_buffer.update(
                letter=prediction.letter,
                confidence=prediction.confidence,
                active_gestures=perception.active_gestures,
            )
            self._process_events(frame, perception, buffer_state, now)
        except Exception:
            LOGGER.exception("Error al procesar un frame de reconocimiento.")

    def _process_events(self, frame, perception, buffer_state, now: float) -> None:
        has_hands = bool(perception.hands)
        if has_hands and not self.hands_detected:
            self._emit("manos", detectadas=True)
        if has_hands:
            self.hands_seen_since_final = True
        self.hands_detected = has_hands
        if has_hands:
            self.last_no_hand_at = None
        elif self.last_no_hand_at is None:
            self.last_no_hand_at = now

        current_text = self.text_buffer.text
        if current_text != self.last_buffer_text:
            self._emit("parcial", texto=current_text)
            self.last_buffer_text = current_text

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = getattr(getattr(perception, "face", None), "landmarks", None)
        pose = getattr(perception, "pose_landmarks", None)
        quality = (face is not None or pose is not None, float(gray.mean()) >= 60)
        if quality != self.last_quality and (
            self.last_quality_at is None or now - self.last_quality_at >= 1.0
        ):
            self._emit("calidad", encuadre=quality[0], luz=quality[1])
            self.last_quality = quality
            self.last_quality_at = now

        if buffer_state.triggered_action == "speak" and (
            self.text_buffer.text.strip() or self.hands_seen_since_final
        ):
            self._finalize()
        elif (
            self.last_no_hand_at is not None
            and now - self.last_no_hand_at >= self.hands_down_seconds
        ):
            self._finalize()

    def _finalize(self) -> None:
        raw = self.text_buffer.text.strip()
        if not raw:
            if self.hands_seen_since_final:
                reason = (
                    "sin_modelo"
                    if not getattr(self.predictor, "is_ready", True)
                    else "sin_letras"
                )
                self._emit("no_reconocido", motivo=reason)
            self.hands_seen_since_final = False
            self.last_no_hand_at = None
            return
        cleaned = self.gemini_proc.clean_text(raw, lang=self.lang)
        self._emit("final", texto=cleaned)
        if not self.no_voice:
            self.speech_out.speak(cleaned, lang=self.lang)
        self.text_buffer.clear()
        self.last_buffer_text = self.text_buffer.text
        self.last_no_hand_at = None
        self.hands_seen_since_final = False

    def close(self) -> None:
        self.tracker.close()


class UnifiedWebHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    lexicon: Lexicon
    config: dict[str, Any]
    lang: str
    no_voice: bool
    alphabet: str
    speech_out: Any | None
    frame_interval_ms: int
    capture_width: int

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".glb": "model/gltf-binary",
        ".json": "application/json",
    }

    def translate_path(self, path: str) -> str:
        parsed_path = unquote(urlparse(path).path)
        if parsed_path == "/avatar" or parsed_path.startswith("/avatar/"):
            relative = parsed_path.removeprefix("/avatar").lstrip("/")
            target = (AVATAR_WEB_DIR / relative).resolve()
            if target != AVATAR_WEB_DIR and AVATAR_WEB_DIR not in target.parents:
                return str(AVATAR_WEB_DIR / "__not_found__")
            return str(target)
        return super().translate_path(path)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self) -> None:
        index_path = FRONTEND_DIR / "index.html"
        body = index_path.read_bytes()
        needle = b'<script type="module" src="js/main.js"></script>'
        injection = (
            b"<script>window.AULA_CONFIG={DEMO_MODE:false,API_URL:'',"
            b"FRAME_INTERVAL_MS:"
            + str(self.frame_interval_ms).encode("ascii")
            + b",CAPTURE_WIDTH:"
            + str(self.capture_width).encode("ascii")
            + b","
            b"RECOGNITION_WS_URL:(location.protocol==='https:'?'wss://':'ws://')"
            b"+location.host+'/reconocer'};</script>"
        )
        body = body.replace(needle, injection + needle, 1)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/reconocer":
            self._serve_websocket()
        elif url.path in ("/", "/index.html"):
            self._serve_index()
        elif url.path == "/salud":
            self._send_json({"ok": True})
        elif url.path == "/api/translate":
            text = parse_qs(url.query).get("text", [""])[0]
            self._send_json(translate_text(self.lexicon, text))
        elif url.path == "/api/lexicon":
            self._send_json(list_lexicon(self.lexicon))
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/traducir-texto":
            self.send_error(404)
            return
        try:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Falta Content-Length.")
            length = int(raw_length)
            if length < 0:
                raise ValueError("Content-Length negativo.")
        except (ValueError, TypeError):
            self._send_json({"error": "Content-Length inválido."}, 400)
            return
        if length > 64 * 1024:
            self._send_json({"error": "El cuerpo de la solicitud excede 64 KiB."}, 413)
            return

        try:
            request = json.loads(self.rfile.read(length))
            text = request.get("texto") if isinstance(request, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("El texto debe ser una cadena no vacía.")
            text = text.strip()[:500]
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "JSON inválido o texto vacío."}, 400)
            return

        result = translate_text(self.lexicon, text)
        self._send_json(
            {
                "texto_sencillo": text,
                "glosa": result["glosa"],
                "secuencia": result["secuencia"],
                "animacion_url": None,
            }
        )

    def _send_ws_frame(self, payload: bytes, opcode: int) -> None:
        self.wfile.write(encode_websocket_frame(payload, opcode))
        self.wfile.flush()

    def _serve_websocket(self) -> None:
        origin = self.headers.get("Origin")
        if origin is not None and urlparse(origin).netloc != self.headers.get("Host"):
            self.send_error(403, "Origin no coincide con Host.")
            return
        if self.headers.get("Upgrade", "").lower() != "websocket":
            self.send_error(426, "Se requiere una conexión WebSocket.")
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "Falta Sec-WebSocket-Key.")
            return
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", websocket_accept(key))
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True

        spec = get_alphabet_spec(self.config, self.alphabet)
        session = None
        try:
            session = RecognitionSession(
                self.config,
                lang=self.lang,
                no_voice=self.no_voice,
                speech_out=self.speech_out,
                emit=lambda event: self._send_ws_frame(
                    json.dumps(event, ensure_ascii=False).encode("utf-8"), 0x1
                ),
            )
            LOGGER.info("Sesión de reconocimiento iniciada con alfabeto %s.", spec["code"])
            while True:
                message = read_websocket_message(self.rfile, self._send_ws_frame)
                if message is None:
                    break
                opcode, payload = message
                if opcode == 0x2:
                    session.process_jpeg(payload)
                elif opcode == 0x1:
                    LOGGER.debug("Se ignora mensaje de texto recibido en /reconocer.")
        except (EOFError, ConnectionError, OSError):
            LOGGER.debug("Conexión WebSocket de reconocimiento cerrada.")
        except Exception:
            LOGGER.exception("Error en la conexión WebSocket de reconocimiento.")
        finally:
            if session is not None:
                session.close()

    def log_message(self, fmt: str, *args) -> None:
        LOGGER.debug(fmt, *args)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    config_path: str | Path = "config.yaml",
    config: dict[str, Any] | None = None,
    lexicon: Lexicon | None = None,
    alphabet: str | None = None,
    lang: str | None = None,
    no_voice: bool = False,
) -> ThreadingHTTPServer:
    """Crea un servidor sin iniciarlo; útil para pruebas y embebido."""
    ensure_handshape_library()
    cfg = config if config is not None else load_config(config_path)
    lex = lexicon or Lexicon.load()
    spec = get_alphabet_spec(cfg, alphabet)
    language = lang or cfg.get("speech", {}).get("language", "es")
    frame_rate_hz = float(cfg.get("web", {}).get("frame_rate_hz", 10))
    if frame_rate_hz <= 0:
        raise ValueError("web.frame_rate_hz debe ser mayor que cero.")
    capture_width = int(cfg.get("camera", {}).get("width", 640))
    shared_speech = (
        None if no_voice else _create_speech_output(cfg, language, async_playback=True)
    )
    configured_handler = type(
        "ConfiguredUnifiedWebHandler",
        (UnifiedWebHandler,),
        {
            "lexicon": lex,
            "config": cfg,
            "lang": language,
            "no_voice": no_voice,
            "alphabet": spec["code"],
            "speech_out": shared_speech,
            "frame_interval_ms": round(1000 / frame_rate_hz),
            "capture_width": capture_width,
        },
    )
    handler = partial(configured_handler, directory=str(FRONTEND_DIR))
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(
    host: str = "127.0.0.1",
    port: int | None = None,
    *,
    config_path: str | Path = "config.yaml",
    alphabet: str | None = None,
    lang: str | None = None,
    no_voice: bool = False,
) -> None:
    cfg = load_config(config_path)
    configured_port = int(cfg.get("web", {}).get("port", 8000))
    server = create_server(
        host,
        configured_port if port is None else port,
        config=cfg,
        alphabet=alphabet,
        lang=lang,
        no_voice=no_voice,
    )
    actual_host, actual_port = server.server_address[:2]
    print(f"[web] Aplicación unificada en http://{actual_host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aplicación web unificada de Aula Accesible.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alphabet", default=None)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--no-voice", action="store_true")
    args = parser.parse_args(argv)
    serve(args.host, args.port, config_path=args.config, alphabet=args.alphabet, lang=args.lang,
          no_voice=args.no_voice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
