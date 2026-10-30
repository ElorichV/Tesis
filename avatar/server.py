"""
Servidor local del avatar LSM (sólo biblioteca estándar).

    python -m avatar.server --port 8765        # o: python -m app avatar

Rutas:
    /                      visor Three.js (avatar/web)
    /api/translate?text=   texto -> glosa LSM -> secuencia de señas (JSON)
    /api/lexicon           términos disponibles en el léxico
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from avatar.lexicon import Lexicon

WEB_DIR = Path(__file__).resolve().parent / "web"
LOGGER = logging.getLogger(__name__)


def translate_text(lexicon: Lexicon, text: str) -> dict:
    return lexicon.translate(text[:500])


def list_lexicon(lexicon: Lexicon) -> list[dict]:
    return sorted(
        (
            {
                "termino": entry["termino"],
                "glosa": entry.get("glosa"),
                "categoria": entry.get("categoria"),
            }
            for entry in lexicon.entries.values()
        ),
        key=lambda entry: entry["termino"],
    )


class AvatarHandler(SimpleHTTPRequestHandler):
    lexicon: Lexicon

    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".glb": "model/gltf-binary", ".json": "application/json"}

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/api/translate":
            text = parse_qs(url.query).get("text", [""])[0][:500]
            self._json(translate_text(self.lexicon, text))
        elif url.path == "/api/lexicon":
            self._json(list_lexicon(self.lexicon))
        else:
            super().do_GET()

    def log_message(self, fmt: str, *args) -> None:
        LOGGER.debug(fmt, *args)


def ensure_handshape_library() -> None:
    lib = WEB_DIR / "assets" / "handshapes.json"
    if not lib.exists():
        from avatar.handshapes import build_library

        build_library(lib)


def serve(host: str = "127.0.0.1", port: int = 8765, lexicon: Lexicon | None = None) -> None:
    ensure_handshape_library()
    AvatarHandler.lexicon = lexicon or Lexicon.load()
    handler = partial(AvatarHandler, directory=str(WEB_DIR))
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"[avatar] Visor LSM en http://{host}:{port}  ({len(AvatarHandler.lexicon.entries)} señas en el léxico)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Visor del avatar LSM")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
