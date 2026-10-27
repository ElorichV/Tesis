"""Módulo de construcción de texto estable y post-procesamiento opcional con Gemini."""

from text.buffer import BufferUpdateResult, StableTextBuffer
from text.gemini_postprocessor import GeminiTextPostProcessor

__all__ = [
    "BufferUpdateResult",
    "GeminiTextPostProcessor",
    "StableTextBuffer",
]

