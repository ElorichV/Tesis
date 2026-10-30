"""CLI principal (`collect | train | run`) para el traductor de lenguaje de señas a voz."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import get_alphabet_spec, load_config
from capture.camera import CameraError, CameraStream, FPSCounter
from recognition.bootstrap import ensure_bootstrap_static_csv
from recognition.collector import SampleCollector, get_collection_guidance
from recognition.predictor import SignPredictor
from recognition.trainer import format_confusion_matrix, train_models
from speech.voice_adapter import SpeechOutput
from text.buffer import StableTextBuffer
from text.gemini_postprocessor import GeminiTextPostProcessor


def _print_alphabet_summary(spec: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"ALFABETO ACTIVO: {spec['code']} - {spec['name']}")
    print(f"  Letras ESTÁTICAS ({len(spec['static'])}): {', '.join(spec['static'])}")
    print(f"  Letras DINÁMICAS ({len(spec['dynamic'])}): {', '.join(spec['dynamic'])}")
    print("=" * 72)


def _print_mediapipe_install_hint(exc: Exception) -> None:
    if isinstance(exc, ImportError) and "mediapipe" in str(exc).lower():
        print(
            r"Instala dependencias: .\.venv\Scripts\python.exe -m pip install -r requirements.txt",
            file=sys.stderr,
        )


def _create_tracker(cfg: dict[str, Any]) -> Any:
    try:
        from perception.mediapipe_tracker import PerceptionTracker
    except ImportError as exc:
        _print_mediapipe_install_hint(exc)
        raise

    p_cfg = cfg.get("perception", {})
    return PerceptionTracker(
        models_dir=p_cfg.get("models_dir", "models/mediapipe"),
        max_num_hands=int(p_cfg.get("max_num_hands", 2)),
        min_hand_detection_confidence=float(p_cfg.get("min_hand_detection_confidence", 0.55)),
        min_hand_presence_confidence=float(p_cfg.get("min_hand_presence_confidence", 0.55)),
        min_tracking_confidence=float(p_cfg.get("min_tracking_confidence", 0.50)),
        enable_face=bool(p_cfg.get("enable_face", True)),
        enable_pose=bool(p_cfg.get("enable_pose", True)),
        blendshape_thresholds=p_cfg.get("face_blendshape_thresholds"),
    )


def _ensure_recognition_ready(cfg: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Prepara modelos de percepción y crea el clasificador público si hace falta."""
    try:
        from perception.mediapipe_tracker import ensure_model_downloaded

        p_cfg = cfg.get("perception", {})
        models_dir = Path(p_cfg.get("models_dir", "models/mediapipe"))
        model_files = [("hand_landmarker.task", True)]
        model_files.extend(
            [
                ("face_landmarker.task", bool(p_cfg.get("enable_face", True))),
                ("pose_landmarker_lite.task", bool(p_cfg.get("enable_pose", True))),
            ]
        )
        for filename, enabled in model_files:
            if enabled:
                ensure_model_downloaded(models_dir, filename)

        rec_cfg = cfg.get("recognition", {})
        static_model_path = Path(rec_cfg.get("model_path", "models/sign_classifier.joblib"))
        if static_model_path.exists():
            return True
        if not rec_cfg.get("bootstrap_public_dataset", False):
            return False

        print(
            "[MediaPipe] No hay modelo entrenado: generando un modelo inicial con el "
            "dataset público ASL (CC-BY-4.0)…"
        )
        bootstrap_csv = ensure_bootstrap_static_csv(cfg, spec)
        reports = train_models(
            static_csv=rec_cfg.get("dataset_csv", "data/landmarks_static.csv"),
            dynamic_csv=rec_cfg.get("dynamic_dataset_csv", "data/landmarks_dynamic.csv"),
            static_model_path=static_model_path,
            dynamic_model_path=rec_cfg.get(
                "dynamic_model_path", "models/sign_dynamic_classifier.joblib"
            ),
            classifier_type=rec_cfg.get("classifier_type", "mlp"),
            alphabet=spec["code"],
            bootstrap_static_csv=bootstrap_csv,
        )
        if not reports:
            raise RuntimeError("No se generó el reporte del modelo estático.")
        report = reports[0]
        print(
            f"[MediaPipe] Accuracy del modelo inicial: {report.accuracy:.2%}; "
            f"clases: {', '.join(report.classes)}"
        )
        return static_model_path.exists()
    except Exception as exc:
        _print_mediapipe_install_hint(exc)
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"[Aviso] No se pudo preparar el reconocimiento: {detail}", file=sys.stderr)
        return False


def _create_camera(cfg: dict[str, Any], camera_override: int | None = None) -> CameraStream:
    c_cfg = cfg.get("camera", {})
    idx = camera_override if camera_override is not None else int(c_cfg.get("device_index", 0))
    return CameraStream(
        device_index=idx,
        width=int(c_cfg.get("width", 640)),
        height=int(c_cfg.get("height", 480)),
        fps=int(c_cfg.get("fps", 30)),
        use_dshow_on_windows=bool(c_cfg.get("use_dshow_on_windows", True)),
        mirror_display=bool(c_cfg.get("mirror_display", True)),
    )


def _text_buffer_frame_counts(
    cfg: dict[str, Any],
    frame_rate_hz: float | None = None,
) -> dict[str, int]:
    """Calcula umbrales del buffer, ajustándolos si cambia la tasa de frames."""
    txt_cfg = cfg.get("text", {})
    counts = {
        "stability_frames": int(txt_cfg.get("stability_frames", 12)),
        "release_frames": int(txt_cfg.get("release_frames", 4)),
        "control_gesture_hold_frames": int(txt_cfg.get("control_gesture_hold_frames", 20)),
    }
    if frame_rate_hz is None:
        return counts

    camera_fps = float(cfg.get("camera", {}).get("fps", 30))
    if camera_fps <= 0 or frame_rate_hz <= 0:
        raise ValueError("Las tasas de frames deben ser mayores que cero.")
    scale = frame_rate_hz / camera_fps
    counts["stability_frames"] = max(2, round(counts["stability_frames"] * scale))
    counts["release_frames"] = max(1, round(counts["release_frames"] * scale))
    counts["control_gesture_hold_frames"] = max(
        1, round(counts["control_gesture_hold_frames"] * scale)
    )
    return counts


def _dynamic_window_frame_count(
    cfg: dict[str, Any],
    frame_rate_hz: float | None = None,
) -> int:
    """Ajusta la ventana temporal del predictor a la tasa de frames web."""
    base_frames = int(cfg.get("features", {}).get("dynamic_window_frames", 25))
    if frame_rate_hz is None:
        return base_frames

    camera_fps = float(cfg.get("camera", {}).get("fps", 30))
    if camera_fps <= 0 or frame_rate_hz <= 0:
        raise ValueError("Las tasas de frames deben ser mayores que cero.")
    return max(3, round(base_frames * frame_rate_hz / camera_fps))


def _create_speech_output(
    cfg: dict[str, Any],
    target_lang: str,
    async_playback: bool = True,
) -> SpeechOutput:
    """Construye el único adaptador de voz compartido por un servidor web."""
    sp_cfg = cfg.get("speech", {})
    return SpeechOutput(
        voice_translate_dir=sp_cfg.get("voice_translate_dir", "./voice-translate"),
        default_lang=target_lang,
        output_wav=sp_cfg.get("output_wav", "output.wav"),
        tts_model_es=sp_cfg.get("tts_model_es"),
        tts_model_en=sp_cfg.get("tts_model_en"),
        async_playback=async_playback,
    )


def _create_run_components(
    cfg: dict[str, Any],
    target_lang: str,
    async_playback: bool = True,
    *,
    warn_untrained: bool = True,
    frame_rate_hz: float | None = None,
    create_speech: bool = True,
) -> tuple[SignPredictor, StableTextBuffer, GeminiTextPostProcessor, SpeechOutput | None]:
    """Construye los componentes compartidos por los modos `run` y `web`."""
    rec_cfg = cfg.get("recognition", {})
    feat_cfg = cfg.get("features", {})
    txt_cfg = cfg.get("text", {})
    gem_cfg = cfg.get("gemini", {})

    predictor = SignPredictor(
        static_model_path=rec_cfg.get("model_path", "models/sign_classifier.joblib"),
        dynamic_model_path=rec_cfg.get(
            "dynamic_model_path", "models/sign_dynamic_classifier.joblib"
        ),
        canonical_hand=feat_cfg.get("canonical_hand", "Right"),
        dynamic_window_frames=_dynamic_window_frame_count(cfg, frame_rate_hz),
        dynamic_min_motion=float(feat_cfg.get("dynamic_min_motion", 0.08)),
    )
    if warn_untrained and not predictor.is_ready:
        print(
            "[AVISO] Aún no existe un modelo entrenado en 'models/'. "
            "El sistema mostrará los landmarks en vivo, pero para reconocer letras "
            "ejecuta primero: `python -m app collect` y `python -m app train`."
        )

    text_buffer = StableTextBuffer(
        **_text_buffer_frame_counts(cfg, frame_rate_hz=frame_rate_hz),
        confidence_threshold=float(txt_cfg.get("confidence_threshold", 0.70)),
        gestures_config=txt_cfg.get("gestures"),
    )
    gemini_proc = GeminiTextPostProcessor(
        enabled=bool(gem_cfg.get("enabled", True)),
        default_model=str(gem_cfg.get("model", "gemini-2.5-flash")),
        timeout_seconds=float(gem_cfg.get("timeout_seconds", 8.0)),
    )
    speech_out = (
        _create_speech_output(cfg, target_lang, async_playback=async_playback)
        if create_speech
        else None
    )
    return predictor, text_buffer, gemini_proc, speech_out


def cmd_collect(args: argparse.Namespace) -> int:
    """Modo de captura interactiva de landmarks (estáticos y dinámicos) con la webcam."""
    try:
        from app.overlay import draw_collect_hud, draw_perception_overlay
    except ImportError as exc:
        _print_mediapipe_install_hint(exc)
        print(f"[ERROR DE PERCEPCIÓN] {exc}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    spec = get_alphabet_spec(cfg, args.alphabet)
    _print_alphabet_summary(spec)

    rec_cfg = cfg.get("recognition", {})
    feat_cfg = cfg.get("features", {})
    collector = SampleCollector(
        static_csv_path=rec_cfg.get("dataset_csv", "data/landmarks_static.csv"),
        dynamic_csv_path=rec_cfg.get("dynamic_dataset_csv", "data/landmarks_dynamic.csv"),
        canonical_hand=feat_cfg.get("canonical_hand", "Right"),
    )

    all_letters: list[str] = spec["static"] + spec["dynamic"]
    dynamic_set = set(spec["dynamic"])

    current_letter = (args.letter or all_letters[0]).strip().upper()
    if current_letter not in all_letters:
        all_letters.append(current_letter)

    static_target = int(args.samples or rec_cfg.get("samples_per_letter", 150))
    dynamic_target = int(rec_cfg.get("dynamic_sequences_per_letter", 20))
    dyn_win_frames = int(feat_cfg.get("dynamic_window_frames", 25))

    counts_static = collector.count_samples_by_label(is_dynamic=False)
    counts_dyn = collector.count_samples_by_label(is_dynamic=True)
    if counts_static or counts_dyn:
        print("Muestras ya guardadas:")
        if counts_static:
            print(f"  Estáticas: {counts_static}")
        if counts_dyn:
            print(f"  Dinámicas: {counts_dyn}")

    try:
        camera = _create_camera(cfg, args.camera)
        camera.open()
    except CameraError as exc:
        print(f"[ERROR DE CÁMARA] {exc}", file=sys.stderr)
        return 1

    tracker = _create_tracker(cfg)
    fps_counter = FPSCounter()

    recording = False
    batch_captured = 0
    dyn_frames_buffer: list[np.ndarray] = []

    print(
        "\nControles en ventana 'Sign Translator - Collect':\n"
        "  - Pulsa una tecla [A-Z] o [Tab] para cambiar la letra objetivo.\n"
        "  - Pulsa [Espacio] para iniciar/pausar la captura del lote de muestras.\n"
        "  - Pulsa [Q] o [ESC] para salir.\n"
    )

    try:
        while True:
            frame = camera.read()
            fps = fps_counter.tick()
            ts_ms = int(time.perf_counter() * 1000)

            perception = tracker.process_frame(frame, timestamp_ms=ts_ms)
            primary = perception.primary_hand
            is_dyn = current_letter in dynamic_set
            target_total = dynamic_target if is_dyn else static_target

            if recording and primary is not None:
                if is_dyn:
                    dyn_frames_buffer.append(primary.landmarks.copy())
                    if len(dyn_frames_buffer) >= dyn_win_frames:
                        collector.save_dynamic_sequence(
                            current_letter, dyn_frames_buffer, handedness=primary.handedness
                        )
                        dyn_frames_buffer.clear()
                        batch_captured += 1
                else:
                    collector.save_static_sample(
                        current_letter, primary.landmarks, handedness=primary.handedness
                    )
                    batch_captured += 1

                if batch_captured >= target_total:
                    print(
                        f"[OK] Captura completada para letra '{current_letter}': "
                        f"{batch_captured} muestras guardadas."
                    )
                    recording = False
                    batch_captured = 0
                    dyn_frames_buffer.clear()

            guidance = get_collection_guidance(batch_captured, target_total, is_dynamic=is_dyn)
            if is_dyn and recording:
                guidance = (
                    f"Trazando secuencia ({len(dyn_frames_buffer)}/{dyn_win_frames} frames)... "
                    + guidance
                )

            canvas = draw_perception_overlay(frame, perception)
            canvas = draw_collect_hud(
                canvas=canvas,
                fps=fps,
                alphabet_code=spec["code"],
                target_letter=current_letter,
                is_dynamic=is_dyn,
                recording=recording,
                captured_count=batch_captured,
                target_count=target_total,
                guidance_text=guidance,
                hand_detected=primary is not None,
            )

            cv2.imshow("Sign Translator - Collect", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break
            elif key == ord(" "):
                recording = not recording
                dyn_frames_buffer.clear()
                if recording:
                    batch_captured = 0
                    print(f"[CAPTURA] Iniciando captura para letra '{current_letter}'...")
                else:
                    print("[CAPTURA] Pausada por el usuario.")
            elif key == 9:  # Tab: siguiente letra del alfabeto
                idx = (all_letters.index(current_letter) + 1) % len(all_letters)
                current_letter = all_letters[idx]
                recording = False
                batch_captured = 0
                dyn_frames_buffer.clear()
                print(f"[LETRA] Seleccionada: '{current_letter}'")
            elif ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
                ch = chr(key).upper()
                if ch != "Q":
                    current_letter = ch
                    recording = False
                    batch_captured = 0
                    dyn_frames_buffer.clear()
                    print(f"[LETRA] Seleccionada: '{current_letter}'")

            if args.max_frames and fps_counter._timestamps and len(fps_counter._timestamps) >= args.max_frames:
                break
    finally:
        tracker.close()
        camera.close()
        cv2.destroyAllWindows()

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Entrena los modelos de clasificación a partir de los CSV recolectados."""
    cfg = load_config(args.config)
    spec = get_alphabet_spec(cfg, args.alphabet)
    rec_cfg = cfg.get("recognition", {})
    clf_type = args.classifier or rec_cfg.get("classifier_type", "mlp")

    try:
        bootstrap_csv = None
        if (
            not getattr(args, "no_bootstrap", False)
            and rec_cfg.get("bootstrap_public_dataset", False)
        ):
            bootstrap_csv = ensure_bootstrap_static_csv(cfg, spec)
        reports = train_models(
            static_csv=rec_cfg.get("dataset_csv", "data/landmarks_static.csv"),
            dynamic_csv=rec_cfg.get("dynamic_dataset_csv", "data/landmarks_dynamic.csv"),
            static_model_path=rec_cfg.get("model_path", "models/sign_classifier.joblib"),
            dynamic_model_path=rec_cfg.get(
                "dynamic_model_path", "models/sign_dynamic_classifier.joblib"
            ),
            classifier_type=clf_type,
            alphabet=spec["code"],
            bootstrap_static_csv=bootstrap_csv,
        )
    except Exception as exc:
        _print_mediapipe_install_hint(exc)
        print(f"[ERROR DE ENTRENAMIENTO] {exc}", file=sys.stderr)
        return 1

    for rep in reports:
        print("\n" + "=" * 72)
        print(f"REPORTE DE ENTRENAMIENTO - {rep.phase}")
        print("=" * 72)
        print(f"Modelo guardado en : {rep.model_path}")
        print(f"Total de muestras  : {rep.num_samples}")
        print(f"Clases ({len(rep.classes)})       : {', '.join(rep.classes)}")
        print(f"Accuracy validación: {rep.accuracy:.2%}\n")
        print("Matriz de Confusión (filas = real, columnas = predicción):")
        print(format_confusion_matrix(rep.classes, rep.confusion_matrix))
        print("\nDetalle por clase:")
        print(rep.classification_summary)

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Ejecuta el traductor de señas a texto y voz en tiempo real."""
    cfg = load_config(args.config)
    spec = get_alphabet_spec(cfg, args.alphabet)
    use_synthetic = bool(args.headless_benchmark)
    if not use_synthetic:
        _ensure_recognition_ready(cfg, spec)

    try:
        from app.overlay import draw_perception_overlay, draw_run_hud
    except ImportError as exc:
        _print_mediapipe_install_hint(exc)
        print(f"[ERROR DE PERCEPCIÓN] {exc}", file=sys.stderr)
        return 1

    _print_alphabet_summary(spec)

    sp_cfg = cfg.get("speech", {})

    target_lang = args.lang or sp_cfg.get("language", "es")
    predictor, text_buffer, gemini_proc, speech_out = _create_run_components(
        cfg, target_lang, async_playback=not args.headless_benchmark
    )

    try:
        tracker = _create_tracker(cfg)
    except Exception as exc:
        _print_mediapipe_install_hint(exc)
        print(f"[ERROR DE PERCEPCIÓN] {exc}", file=sys.stderr)
        return 1
    fps_counter = FPSCounter()
    last_spoken = ""

    def confirm_and_speak() -> None:
        nonlocal last_spoken
        raw = text_buffer.text.strip()
        if not raw:
            return
        cleaned = gemini_proc.clean_text(raw, lang=target_lang)
        last_spoken = cleaned
        print(f"[TEXTO CONFIRMADO] Crudo: '{raw}' -> Procesado: '{cleaned}'")
        speech_out.speak(cleaned, lang=target_lang)
        text_buffer.clear()

    camera: CameraStream | None = None
    if not use_synthetic:
        try:
            camera = _create_camera(cfg, args.camera)
            camera.open()
        except CameraError as exc:
            print(f"[ERROR DE CÁMARA] {exc}", file=sys.stderr)
            tracker.close()
            return 1

    frame_count = 0
    fps_samples: list[float] = []

    try:
        while True:
            if use_synthetic:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                assert camera is not None
                frame = camera.read()

            frame_count += 1
            ts_ms = int(time.perf_counter() * 1000)

            perception = tracker.process_frame(frame, timestamp_ms=ts_ms)
            primary = perception.primary_hand

            if primary is not None:
                pred = predictor.predict(primary.landmarks, handedness=primary.handedness)
            else:
                pred = predictor.predict(None)

            buf_state = text_buffer.update(
                letter=pred.letter,
                confidence=pred.confidence,
                active_gestures=perception.active_gestures,
            )

            if buf_state.committed_letter:
                print(
                    f"[LETRA ACEPTADA] +{buf_state.committed_letter} "
                    f"-> Buffer: '{buf_state.current_text}'"
                )

            if buf_state.triggered_action == "speak":
                confirm_and_speak()

            fps = fps_counter.tick()
            if frame_count > 3 and fps > 0:
                fps_samples.append(fps)

            canvas = draw_perception_overlay(frame, perception)
            canvas = draw_run_hud(
                canvas=canvas,
                fps=fps,
                prediction=pred,
                buffer_state=buf_state,
                alphabet_code=spec["code"],
                last_spoken=last_spoken,
            )

            if not use_synthetic:
                cv2.imshow("Sign Translator - Live", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                elif key == ord(" "):
                    text_buffer.add_space()
                elif key in (8, 127):  # Backspace
                    text_buffer.backspace()
                elif key in (10, 13):  # Enter -> Hablar
                    confirm_and_speak()
                elif key in (ord("c"), ord("C")):
                    text_buffer.clear()

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        tracker.close()
        if camera is not None:
            camera.close()
        if not use_synthetic:
            cv2.destroyAllWindows()

    if fps_samples:
        avg_fps = float(np.mean(fps_samples))
        print(f"\n[RENDIMIENTO] FPS promedio en CPU: {avg_fps:.1f} FPS ({frame_count} frames)")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Traductor de lenguaje de señas a voz en tiempo real con MediaPipe.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta al archivo de configuración YAML (por defecto: config.yaml).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: collect
    p_collect = subparsers.add_parser(
        "collect",
        help="Captura muestras de landmarks desde la webcam y las guarda en CSV.",
    )
    p_collect.add_argument("--letter", type=str, default=None, help="Letra inicial a capturar.")
    p_collect.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Número de muestras por lote (por defecto: 150 en config.yaml).",
    )
    p_collect.add_argument(
        "--alphabet",
        type=str,
        default=None,
        help="Alfabeto a usar (ASL o LSM).",
    )
    p_collect.add_argument("--camera", type=int, default=None, help="Índice de cámara web.")
    p_collect.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    p_collect.set_defaults(func=cmd_collect)

    # Subcomando: train
    p_train = subparsers.add_parser(
        "train",
        help="Entrena el clasificador de señas y muestra accuracy + matriz de confusión.",
    )
    p_train.add_argument(
        "--classifier",
        choices=["mlp", "knn"],
        default=None,
        help="Tipo de clasificador: 'mlp' o 'knn'.",
    )
    p_train.add_argument("--alphabet", type=str, default=None, help="Alfabeto (ASL o LSM).")
    p_train.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Entrena solo con las muestras CSV propias.",
    )
    p_train.set_defaults(func=cmd_train)

    # Subcomando: run
    p_run = subparsers.add_parser(
        "run",
        help="Ejecuta la traducción de lenguaje de señas a voz en vivo.",
    )
    p_run.add_argument("--alphabet", type=str, default=None, help="Alfabeto (ASL o LSM).")
    p_run.add_argument("--lang", type=str, default=None, help="Idioma de salida de voz (es o en).")
    p_run.add_argument("--camera", type=int, default=None, help="Índice de cámara web.")
    p_run.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Detiene la ejecución tras N frames (útil para pruebas/benchmarks).",
    )
    p_run.add_argument(
        "--headless-benchmark",
        action="store_true",
        help="Ejecuta el pipeline completo en CPU sin ventana para medir FPS en entornos sin cámara.",
    )
    p_run.set_defaults(func=cmd_run)

    p_avatar = subparsers.add_parser(
        "avatar", help="Visor web del avatar 3D que interpreta texto en LSM"
    )
    p_avatar.add_argument("--host", default="127.0.0.1")
    p_avatar.add_argument("--port", type=int, default=8765)
    p_avatar.set_defaults(func=cmd_avatar)

    p_web = subparsers.add_parser(
        "web", help="Inicia la aplicación web unificada con avatar y reconocimiento en vivo."
    )
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=None)
    p_web.add_argument("--no-voice", action="store_true", help="Desactiva la salida de voz.")
    p_web.add_argument("--alphabet", type=str, default=None, help="Alfabeto (ASL o LSM).")
    p_web.add_argument("--lang", type=str, default=None, help="Idioma de salida de voz (es o en).")
    p_web.set_defaults(func=cmd_web)

    return parser


def cmd_avatar(args: argparse.Namespace) -> int:
    from avatar.server import serve

    serve(args.host, args.port)
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from app.web import serve

    serve(
        host=args.host,
        port=args.port,
        config_path=args.config,
        alphabet=args.alphabet,
        lang=args.lang,
        no_voice=args.no_voice,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
