"""Módulo de recolección de muestras, entrenamiento y predicción de señas."""

from recognition.collector import SampleCollector, get_collection_guidance
from recognition.predictor import PredictionOutput, SignPredictor
from recognition.trainer import TrainingReport, train_models

__all__ = [
    "PredictionOutput",
    "SampleCollector",
    "SignPredictor",
    "TrainingReport",
    "get_collection_guidance",
    "train_models",
]

