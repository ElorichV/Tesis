"""Post-procesador opcional de texto con el SDK oficial google-genai."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class GeminiTextPostProcessor:
    """
    Corrige secuencias de letras deletreadas en señas (p. ej. "H O L A" o "HOLA COMO ESTAS")
    a palabras/frases naturales en el idioma configurado usando `google-genai`.
    Nunca recibe ni envía imágenes ni video; opera exclusivamente sobre cadenas de texto.
    Si no hay `GEMINI_API_KEY` o falla la red, devuelve el texto original limpio sin bloquear.
    """

    def __init__(
        self,
        enabled: bool = True,
        default_model: str = "gemini-2.5-flash",
        timeout_seconds: float = 8.0,
    ) -> None:
        load_dotenv()
        self.enabled = enabled
        self.model_name = os.getenv("GEMINI_MODEL", default_model).strip() or default_model
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.timeout_seconds = timeout_seconds
        self._client = None

        if self.enabled and self.api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
                logger.info("GeminiTextPostProcessor activo con modelo '%s'.", self.model_name)
            except Exception as exc:
                logger.warning("No se pudo inicializar google-genai (%s). Usando modo local.", exc)
                self._client = None
        else:
            logger.info("GeminiTextPostProcessor desactivado (sin GEMINI_API_KEY o deshabilitado).")

    @property
    def is_available(self) -> bool:
        return self.enabled and self._client is not None

    def clean_text(self, raw_text: str, lang: str = "es") -> str:
        """
        Convierte una secuencia de letras deletreadas en una frase limpia.
        Si Gemini no está disponible o ocurre cualquier error, normaliza localmente.
        """
        fallback = self._local_fallback(raw_text)
        if not fallback or not self.is_available:
            return fallback

        lang_label = "español" if lang.lower().startswith("es") else "inglés"
        prompt = (
            f"Eres un asistente de accesibilidad para un deletreador de lengua de señas en {lang_label}. "
            "Convierte la siguiente secuencia de letras reconocidas en palabras o una frase natural "
            "corrigiendo pequeños errores de deletreo o espacios faltantes, sin agregar explicaciones, "
            "sin comillas y conservando exactamente el sentido original.\n"
            f"Entrada: {raw_text.strip()}\n"
            "Salida:"
        )
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            cleaned = (response.text or "").strip().strip('"').strip("'")
            return cleaned if cleaned else fallback
        except Exception as exc:
            logger.warning("Fallo al consultar Gemini (%s). Usando texto directo: '%s'", exc, fallback)
            return fallback

    @staticmethod
    def _local_fallback(raw_text: str) -> str:
        """Normalización local cuando Gemini no está configurado o sin red."""
        stripped = " ".join(raw_text.strip().split())
        if not stripped:
            return ""
        tokens = stripped.split(" ")
        # Si todos los tokens son letras individuales ("H O L A"), las une en una palabra ("hola")
        if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
            return "".join(tokens).lower()
        return stripped.lower()

