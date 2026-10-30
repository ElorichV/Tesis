"""
Texto en español -> glosa LSM -> secuencia de señas para el avatar.

Flujo (según el documento de la tesis):
1. Normalización y tokenización del texto.
2. Glosa LSM: se eliminan artículos/preposiciones, se lematiza, los marcadores de tiempo van al
   inicio y los pronombres interrogativos al final (orden Tiempo-Tópico-Comentario).
3. Cada `tokens_clave` se busca en el léxico (`data/lexicon_lsm.json`). Si no existe una seña
   propia, se usa el alfabeto dactilológico mexicano (`data/alphabet_lsm.json`).
4. Cada seña se representa como fotogramas clave por mano: configuración (`shape`), ubicación
   (`loc`), orientación (`palma`/`dedos`) y desplazamientos; si la entrada sólo trae
   `descripcion_movimiento` (p. ej. ingerida de un PDF), `parse_description` la convierte.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
LEXICON_PATH = DATA_DIR / "lexicon_lsm.json"
ALPHABET_PATH = DATA_DIR / "alphabet_lsm.json"

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "de", "del", "al", "a", "en", "y", "e",
    "o", "u", "que", "con", "por", "para", "se", "su", "sus", "es", "son", "esta", "estan", "este",
    "esto", "eso", "ser", "estar", "muy", "mis", "tus", "estoy", "estas", "estamos", "estaba", "estuve",
    "soy", "eres", "somos", "era", "fue", "hay", "le", "les", "me", "te", "nos",
}
ENCLITICS = ("melo", "selo", "nos", "me", "te", "se", "lo", "la", "le", "los", "las", "les")
# Palabras que sí tienen seña aunque parezcan funcionales.
KEEP = {"no", "si", "yo", "tu", "ella", "nosotros", "ustedes", "ellos", "mi"}
TIME_WORDS = {"ayer", "hoy", "mañana", "antes", "despues", "futuro", "pasado", "ahora", "siempre", "nunca",
              "noche", "tarde"}
QUESTION_WORDS = {"que", "quien", "como", "donde", "cuando", "cuanto", "cual", "por_que"}
VERB_SUFFIXES = [
    ("amos", ["ar"]), ("emos", ["er"]), ("imos", ["ir"]), ("aste", ["ar"]), ("iste", ["er", "ir"]),
    ("aron", ["ar"]), ("ieron", ["er", "ir"]), ("ando", ["ar"]), ("iendo", ["er", "ir"]),
    ("aba", ["ar"]), ("ia", ["er", "ir"]), ("as", ["ar"]), ("es", ["er", "ir"]), ("an", ["ar"]),
    ("en", ["er", "ir"]), ("o", ["ar", "er", "ir"]), ("a", ["ar"]), ("e", ["er", "ir", "ar"]),
    ("é", ["ar"]), ("ó", ["ar"]), ("io", ["er", "ir"]),
]

LOCATIONS = {
    "cabeza": "cabeza", "frente": "frente", "sien": "sien", "ojo": "ojo", "ojos": "ojo", "nariz": "nariz",
    "boca": "boca", "labios": "boca", "labio": "boca", "barbilla": "barbilla", "menton": "barbilla",
    "mejilla": "mejilla", "cachete": "mejilla", "oreja": "oreja", "oido": "oreja", "cuello": "cuello",
    "hombro": "hombro", "pecho": "pecho", "corazon": "pecho", "estomago": "estomago", "vientre": "estomago",
    "cintura": "cintura", "espacio neutral": "neutral", "frente al cuerpo": "neutral", "neutral": "neutral",
}
PALM = {"adelante": "frente", "al frente": "frente", "afuera": "frente", "frente": "frente",
        "atras": "adentro", "adentro": "adentro", "al cuerpo": "adentro", "hacia el cuerpo": "adentro",
        "abajo": "abajo", "arriba": "arriba", "izquierda": "lado", "derecha": "lado", "lado": "lado",
        "al lado": "lado"}
DIRECTIONS = {"adelante": [0, 0, 1], "atras": [0, 0, -1], "arriba": [0, 1, 0], "abajo": [0, -1, 0],
              "derecha": [-1, 0, 0], "izquierda": [1, 0, 0], "afuera": [-1, 0, 0], "adentro": [1, 0, 0]}


def strip_accents(text: str) -> str:
    text = text.replace("ñ", "\0").replace("Ñ", "\1")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.replace("\0", "ñ").replace("\1", "Ñ")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text.lower())).strip()


@dataclass
class Lexicon:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    alphabet: dict[str, dict[str, Any]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, lexicon_path: Path = LEXICON_PATH, alphabet_path: Path = ALPHABET_PATH) -> "Lexicon":
        lex = cls()
        for path in [lexicon_path, *sorted(DATA_DIR.glob("lexicon_*.extra.json"))]:
            if Path(path).exists():
                lex.add_entries(json.loads(Path(path).read_text(encoding="utf-8"))["senas"])
        alpha = json.loads(Path(alphabet_path).read_text(encoding="utf-8"))
        lex.alphabet = {k.upper(): v for k, v in alpha["letras"].items()}
        return lex

    def add_entries(self, entries: list[dict[str, Any]]) -> None:
        for e in entries:
            key = normalize(e["termino"])
            if "keyframes" not in e and e.get("descripcion_movimiento"):
                e = {**e, "keyframes": parse_description(e["descripcion_movimiento"]),
                     "generado_desde_descripcion": True}
            self.entries.setdefault(key, e)
            for alias in e.get("sinonimos", []):
                self.aliases.setdefault(normalize(alias), key)

    def lookup(self, word: str) -> dict[str, Any] | None:
        w = normalize(word)
        for cand in self.lemma_candidates(w):
            if cand in self.entries:
                return self.entries[cand]
            if cand in self.aliases:
                return self.entries[self.aliases[cand]]
        return None

    @staticmethod
    def lemma_candidates(w: str) -> list[str]:
        cands = [w]
        bases = [w]
        for enc in ENCLITICS:
            if w.endswith(enc) and len(w) - len(enc) >= 3:
                base = strip_accents(w[: -len(enc)])
                cands.append(base)
                bases.append(base)
        for b in bases:
            cands.extend(Lexicon._inflections(b))
        return list(dict.fromkeys(cands))

    @staticmethod
    def _inflections(w: str) -> list[str]:
        cands = []
        if w.endswith("es") and len(w) > 4:
            cands.append(w[:-2])
        if w.endswith("s") and len(w) > 3:
            cands.append(w[:-1])
        for suf, endings in VERB_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 2:
                stem = strip_accents(w[: -len(suf)])
                stems = [stem] + [stem[::-1].replace(a[::-1], b[::-1], 1)[::-1]
                                  for a, b in (("ue", "u"), ("ue", "o"), ("ie", "e"), ("i", "e")) if a in stem]
                cands.extend(st + e for st in stems for e in endings)
        if w.endswith("a") and len(w) > 3:
            cands.append(w[:-1] + "o")
        return cands

    # ------------------------------------------------------------------ glosa
    def gloss(self, text: str) -> list[str]:
        # "él" (pronombre) se distingue de "el" (artículo) antes de quitar acentos.
        t = re.sub(r"(?<![\wáéíóúñ])él(?![\wáéíóúñ])", "ele_pron", text.lower())
        t = normalize(t)
        t = re.sub(r"por\s+que", "por_que", t)
        words = re.findall(r"[a-zñ_]+|\d+", t)
        question = "?" in text or "¿" in text
        times, body, qs = [], [], []
        i = 0
        while i < len(words):
            phrase = None
            for n in (3, 2):
                cand = " ".join(words[i:i + n])
                if len(words) - i >= n and (cand in self.entries or cand in self.aliases):
                    phrase = cand
                    break
            if phrase:
                body.append(phrase)
                i += len(phrase.split())
                continue
            w = words[i]
            i += 1
            if w == "ele_pron":
                body.append("él")
            elif w in TIME_WORDS:
                times.append(w)
            elif question and w in QUESTION_WORDS:
                qs.append(w)
            elif w in STOPWORDS and w not in KEEP:
                continue
            else:
                body.append(w)
        return [w.upper() for w in times + body + qs]

    def fingerspell(self, word: str) -> list[dict[str, Any]]:
        letters = []
        w = word.upper()
        i = 0
        while i < len(w):
            ch = w[i:i + 2] if w[i:i + 2] in ("LL", "RR", "CH") and w[i:i + 2] in self.alphabet else w[i]
            i += len(ch)
            ch = strip_accents(ch) if ch != "Ñ" else ch
            if ch in self.alphabet:
                letters.append({"letra": ch, **self.alphabet[ch]})
            elif ch.isdigit() and f"NUM_{ch}" in self.alphabet:
                letters.append({"letra": ch, **self.alphabet[f"NUM_{ch}"]})
        return letters

    def translate(self, text: str) -> dict[str, Any]:
        """Devuelve la glosa y la secuencia de señas lista para el visor."""
        tokens = self.gloss(text)
        sequence = []
        for tok in tokens:
            entry = self.lookup(tok)
            if entry is not None:
                sequence.append({"tipo": "sena", "glosa": entry.get("glosa", tok), "token": tok,
                                 "termino": entry["termino"], "duracion": entry.get("duracion", 1.0),
                                 "keyframes": entry["keyframes"],
                                 "fuente": entry.get("fuente_oficial")})
            else:
                letters = self.fingerspell(tok)
                if letters:
                    sequence.append({"tipo": "deletreo", "glosa": "dm-" + tok, "token": tok,
                                     "letras": letters})
        return {"texto": text, "glosa": " ".join(tokens), "tokens_clave": tokens, "secuencia": sequence}


# ---------------------------------------------------------------------- parser de descripciones
_SHAPE_PATTERNS = [
    (r"(?:forma|configuracion|letra) (?:de )?(?:la )?(?:letra )?[\"']?([a-zñ])\b",
     lambda m: f"LSM_{m.group(1).upper()}"),
    (r"mano(?:s)? (?:en|con) [\"']?([a-zñ])\b(?![\"']? (?:la|el|los|las)\b)", lambda m: f"LSM_{m.group(1).upper()}"),
    (r"mano(?:s)? (?:abierta|extendida|plana)s?", lambda m: "plana"),
    (r"dedos (?:abiertos|separados|extendidos y separados)", lambda m: "5"),
    (r"(?:pu[ñn]os?|manos? cerradas?)", lambda m: "puño"),
    (r"(?:dedo )?indice (?:extendido|estirado)|senala", lambda m: "1"),
    (r"(?:dedos )?indice y (?:dedo )?medio (?:extendidos|separados)", lambda m: "2"),
    (r"pulgar (?:extendido|arriba)", lambda m: "pulgar"),
    (r"(?:pinza|indice y pulgar (?:juntos|unidos|se tocan))", lambda m: "pinza"),
    (r"(?:garra|dedos flexionados|dedos curvos)", lambda m: "5_garra"),
    (r"(?:mano )?(?:en )?forma de c|mano curva", lambda m: "curva"),
    (r"yemas juntas|dedos juntos en punta|\bbolsa\b", lambda m: "bolsa"),
]


def _find_shape(text: str) -> str | None:
    for pat, fn in _SHAPE_PATTERNS:
        m = re.search(pat, text)
        if m:
            return fn(m)
    return None


def _find_location(text: str) -> str | None:
    best = None
    for word, loc in LOCATIONS.items():
        pat = r"\bfrente\b(?!\s+a)" if word == "frente" else r"\b" + word + r"\b"
        m = re.search(pat, text)
        if m and (best is None or (m.start(), -len(word)) < best[:2]):
            best = (m.start(), -len(word), loc)
    return best[2] if best else None


def _find_palm(text: str) -> str | None:
    m = re.search(r"palma(?:s)? (?:hacia |viendo hacia |mirando hacia |hacia el |hacia la |al )?"
                  r"(al frente|adelante|afuera|atras|adentro|al cuerpo|el cuerpo|abajo|arriba|"
                  r"izquierda|derecha|al lado|lado)", text)
    if not m:
        return None
    key = m.group(1)
    return PALM.get(key, "adentro" if "cuerpo" in key else None)


def _find_moves(text: str) -> list[dict[str, Any]]:
    moves = []
    for m in re.finditer(r"(?:mueve|movimiento|desliza|desplaza|lleva|baja|sube|avanza|hacia)\w*"
                         r"(?: \w+){0,3}? (?:hacia )?(adelante|atras|arriba|abajo|derecha|izquierda|afuera|adentro)",
                         text):
        moves.append({"tipo": "linea", "dir": DIRECTIONS[m.group(1)]})
    if re.search(r"\bbaja\b", text) and not any(mv["dir"] == [0, -1, 0] for mv in moves):
        moves.append({"tipo": "linea", "dir": [0, -1, 0]})
    if re.search(r"\bsube\b", text) and not any(mv["dir"] == [0, 1, 0] for mv in moves):
        moves.append({"tipo": "linea", "dir": [0, 1, 0]})
    if re.search(r"circul|circular|gira", text):
        moves.append({"tipo": "circulo"})
    if re.search(r"agita|oscila|sacude|vaiven|ondula|mueve los dedos", text):
        moves.append({"tipo": "oscilar"})
    if re.search(r"toca|golpe|contacto|apoya", text):
        moves.append({"tipo": "tocar"})
    return moves


def parse_description(desc: str) -> list[dict[str, Any]]:
    """Convierte una descripción textual del movimiento (formato del diccionario SEP) en fotogramas."""
    text = normalize(desc)
    both = bool(re.search(r"ambas manos|las dos manos|dos manos", text))
    shape = _find_shape(text) or "plana"
    loc = _find_location(text) or "neutral"
    palm = _find_palm(text) or ("adentro" if loc not in ("neutral", "cintura") else "frente")
    moves = _find_moves(text)
    repeat = 2 if re.search(r"dos veces|repite|repetid|varias veces", text) else 1
    if re.search(r"(?:cierra|se cierra|cerrando)", text):
        final_shape = "puño" if shape in ("plana", "5") else "bolsa"
    elif re.search(r"(?:abre|se abre|abriendo)", text):
        final_shape = "5"
    else:
        final_shape = shape
    hand0 = {"shape": shape, "loc": loc, "palma": palm, "dedos": "arriba"}
    frames = [{"t": 0.0, "R": hand0}]
    offset = [0.0, 0.0, 0.0]
    t = 0.0
    for _ in range(repeat):
        for mv in moves or [{"tipo": "tocar"}]:
            t += 0.5
            if mv["tipo"] == "linea":
                offset = [o + 0.12 * d for o, d in zip(offset, mv["dir"])]
                frames.append({"t": t, "R": {**hand0, "shape": final_shape, "offset": list(offset)}})
            elif mv["tipo"] == "circulo":
                frames.append({"t": t, "R": {**hand0, "offset": list(offset), "mov": {"tipo": "circulo"}}})
            elif mv["tipo"] == "oscilar":
                frames.append({"t": t, "R": {**hand0, "offset": list(offset), "mov": {"tipo": "oscilar"}}})
            else:
                frames.append({"t": t, "R": {**hand0, "shape": final_shape, "offset": [0, 0, 0.04]}})
                t += 0.3
                frames.append({"t": t, "R": {**hand0, "shape": final_shape}})
        if repeat > 1:
            t += 0.3
            offset = [0.0, 0.0, 0.0]
            frames.append({"t": t, "R": hand0})
    if both:
        for f in frames:
            f["L"] = {**f["R"], "espejo": True}
    return frames
