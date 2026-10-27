"""
Adaptador `SpeechOutput` que reutiliza `voice-translate/main-vosk-argos-tts.py`
sin modificar el repositorio original.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
import threading
from types import ModuleType

logger = logging.getLogger(__name__)


class SpeechOutput:
    """
    Interfaz unificada `SpeechOutput.speak(text: str, lang: str = "es")` que reutiliza
    los componentes `TTS` (Coqui TTS), `argostranslate` y `playsound` importados desde
    el módulo `voice-translate/main-vosk-argos-tts.py`.
    """

    DEFAULT_MODELS = {
        "es": "tts_models/es/mai/tacotron2-DDC",
        "en": "tts_models/en/ljspeech/tacotron2-DDC_ph",
    }

    def __init__(
        self,
        voice_translate_dir: str | Path = "./voice-translate",
        default_lang: str = "es",
        output_wav: str = "output.wav",
        tts_model_es: str | None = None,
        tts_model_en: str | None = None,
        translate_es_to_en: bool = False,
        async_playback: bool = True,
    ) -> None:
        self.voice_translate_dir = Path(voice_translate_dir).resolve()
        self.default_lang = default_lang
        self.output_wav = output_wav
        self.models_by_lang = {
            "es": tts_model_es or self.DEFAULT_MODELS["es"],
            "en": tts_model_en or self.DEFAULT_MODELS["en"],
        }
        self.translate_es_to_en = translate_es_to_en
        self.async_playback = async_playback

        self._vt_module: ModuleType | None = None
        self._vt_load_attempted = False
        self._tts_instances: dict[str, object] = {}
        self._argos_ready = False
        self._lock = threading.Lock()

    def _ensure_vt_module(self) -> ModuleType | None:
        if not self._vt_load_attempted:
            self._vt_load_attempted = True
            self._vt_module = self._load_voice_translate_module()
        return self._vt_module

    def _load_voice_translate_module(self) -> ModuleType | None:
        """
        Importa `main-vosk-argos-tts.py` desde el repositorio `voice-translate` como módulo
        (sin ejecutar su bloque `__main__`), reutilizando sus dependencias `TTS`, `playsound`
        y `argostranslate`.
        """
        script_path = self.voice_translate_dir / "main-vosk-argos-tts.py"
        if not script_path.exists():
            logger.warning(
                "No se encontró %s. SpeechOutput usará modo stub (consola).",
                script_path,
            )
            return None

        try:
            spec = importlib.util.spec_from_file_location("voice_translate_tts", str(script_path))
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info("Módulo voice-translate cargado desde %s", script_path)
            return module
        except Exception as exc:
            logger.warning(
                "No se pudo importar main-vosk-argos-tts.py directamente (%s). "
                "Se usará fallback seguro.",
                exc,
            )
            return None

    def _get_tts_engine(self, lang: str):
        """Obtiene o inicializa bajo demanda el motor Coqui TTS del módulo voice-translate."""
        lang_key = "en" if lang.lower().startswith("en") else "es"
        if lang_key in self._tts_instances:
            return self._tts_instances[lang_key]

        vt_mod = self._ensure_vt_module()
        if vt_mod is None or not hasattr(vt_mod, "TTS"):
            return None

        model_name = self.models_by_lang.get(lang_key, self.DEFAULT_MODELS["es"])
        logger.info("Inicializando Coqui TTS (voice-translate) con modelo '%s'...", model_name)
        tts_cls = getattr(vt_mod, "TTS")
        engine = tts_cls(model_name=model_name)
        self._tts_instances[lang_key] = engine
        return engine

    def _maybe_translate(self, text: str, target_lang: str) -> str:
        """
        Si se solicita `lang="en"` con `translate_es_to_en=True`, reutiliza `argostranslate`
        del módulo `voice-translate` tal como lo hace `main-vosk-argos-tts.py`.
        """
        if not self.translate_es_to_en or not target_lang.lower().startswith("en"):
            return text
        vt_mod = self._ensure_vt_module()
        if vt_mod is None:
            return text

        try:
            pkg = getattr(vt_mod, "package")
            translate_fn = getattr(vt_mod, "translate")
            from_code = getattr(vt_mod, "from_code", "es")
            to_code = getattr(vt_mod, "to_code", "en")

            if not self._argos_ready:
                pkg.update_package_index()
                available = pkg.get_available_packages()
                pkg_to_install = next(
                    filter(lambda x: x.from_code == from_code and x.to_code == to_code, available),
                    None,
                )
                if pkg_to_install is not None:
                    pkg.install_from_path(pkg_to_install.download())
                self._argos_ready = True

            return translate_fn(text, from_code, to_code)
        except Exception as exc:
            logger.warning("No se pudo traducir con Argos Translate (%s). Usando texto original.", exc)
            return text

    def speak(self, text: str, lang: str = "es") -> bool:
        """
        Convierte `text` en voz reutilizando el pipeline TTS + reproducción de `voice-translate`.
        """
        clean = text.strip()
        if not clean:
            return False

        target_lang = lang or self.default_lang
        print(f"\n[VOZ ({target_lang})] -> {clean}", flush=True)

        if self.async_playback:
            thread = threading.Thread(
                target=self._speak_sync,
                args=(clean, target_lang),
                daemon=True,
            )
            thread.start()
            return True

        return self._speak_sync(clean, target_lang)

    def _speak_sync(self, text: str, lang: str) -> bool:
        with self._lock:
            vt_mod = self._ensure_vt_module()
            if vt_mod is None:
                logger.info("[STUB VOZ] (%s): %s", lang, text)
                return False

            try:
                spoken_text = self._maybe_translate(text, lang)
                tts = self._get_tts_engine(lang)
                if tts is None:
                    return False

                wav_path = os.path.abspath(self.output_wav)
                # Misma llamada que en voice-translate/main-vosk-argos-tts.py (líneas 127-131)
                tts.tts_to_file(text=spoken_text, file_path=wav_path)
                self._play_wav(wav_path)
                return True
            except Exception as exc:
                logger.warning("Error durante síntesis/reproducción TTS (%s).", exc)
                return False

    def _play_wav(self, wav_path: str) -> None:
        """
        Reproduce el archivo WAV usando `playsound.playsound` de `voice-translate`,
        con respaldo nativo (`winsound` en Windows) si MCI falla con rutas largas.
        """
        try:
            if self._vt_module is not None and hasattr(self._vt_module, "playsound"):
                self._vt_module.playsound.playsound(wav_path)
                return
        except Exception as exc:
            logger.debug("playsound lanzó excepción (%s); probando fallback winsound.", exc)

        if os.name == "nt":
            import winsound

            winsound.PlaySound(wav_path, winsound.SND_FILENAME)

