"""
Ingesta de diccionarios de LSM en PDF (p. ej. "Manos con Voz", SEP/CONAPRED) con PyMuPDF.

Extrae, por página, los bloques de texto de cada término (término en mayúsculas/negritas seguido de
su descripción de movimiento) y las imágenes asociadas. Genera `data/lexicon_<nombre>.extra.json`
con el formato del documento de la tesis; `Lexicon.load()` lo combina automáticamente y convierte
cada `descripcion_movimiento` en animación con `parse_description`.

    pip install pymupdf
    python -m avatar.ingest_pdf diccionario_lsm.pdf --fuente "Manos con Voz (SEP/CONAPRED, 2011)" \
        --vigencia 2011 --paginas 10-400
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from avatar.lexicon import DATA_DIR

TERM_RE = re.compile(r"^[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ \-()]{1,40}$")
MOVEMENT_HINTS = ("mano", "dedo", "palma", "índice", "pulgar", "movimiento", "mueve", "toca", "puño")


def parse_page_text(lines: list[str]) -> list[tuple[str, str]]:
    """Agrupa líneas en pares (término, descripción). Un término es una línea en mayúsculas."""
    pairs, term, desc = [], None, []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if TERM_RE.match(line) and not line.isdigit():
            if term and desc:
                pairs.append((term, " ".join(desc)))
            term, desc = line.title() if len(line) > 1 else line, []
        elif term:
            desc.append(line)
    if term and desc:
        pairs.append((term, " ".join(desc)))
    return [(t, d) for t, d in pairs if any(h in d.lower() for h in MOVEMENT_HINTS)]


def ingest(pdf_path: Path, fuente: str, vigencia: str, pages: range | None, images_dir: Path | None,
           categoria: str = "general") -> list[dict]:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    entries = []
    for pno in pages or range(len(doc)):
        if pno >= len(doc):
            break
        page = doc[pno]
        pairs = parse_page_text(page.get_text("text").splitlines())
        image_paths = []
        if images_dir is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(page.get_images(full=True)):
                pix = pymupdf.Pixmap(doc, img[0])
                if pix.n - pix.alpha > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                out = images_dir / f"p{pno + 1:04d}_{i}.png"
                pix.save(out)
                image_paths.append(str(out))
        for k, (term, desc) in enumerate(pairs):
            entries.append({
                "termino": term.lower(), "glosa": term.upper(), "categoria": categoria,
                "descripcion_movimiento": desc,
                "imagen_url": image_paths[k] if k < len(image_paths) else None,
                "fuente_oficial": f"{fuente}, p. {pno + 1}", "vigencia": vigencia,
                "revisado_por_interprete": False,
            })
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingesta de diccionario LSM en PDF")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--fuente", required=True)
    ap.add_argument("--vigencia", default="")
    ap.add_argument("--categoria", default="general")
    ap.add_argument("--paginas", default=None, help="Rango 1-indexado, p. ej. 10-200")
    ap.add_argument("--imagenes", type=Path, default=None, help="Carpeta para extraer imágenes")
    ap.add_argument("--nombre", default=None, help="Sufijo del archivo de salida")
    args = ap.parse_args(argv)
    pages = None
    if args.paginas:
        a, b = (int(x) for x in args.paginas.split("-"))
        pages = range(a - 1, b)
    entries = ingest(args.pdf, args.fuente, args.vigencia, pages, args.imagenes, args.categoria)
    name = args.nombre or re.sub(r"\W+", "_", args.pdf.stem.lower())
    out = DATA_DIR / f"lexicon_{name}.extra.json"
    out.write_text(json.dumps({"fuente": args.fuente, "senas": entries}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"[ingest] {len(entries)} términos -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
