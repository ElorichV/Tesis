"""Buffer de estabilidad para construcción de texto y gestos de control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BufferUpdateResult:
    """Estado devuelto en cada frame al actualizar el buffer de texto."""

    committed_letter: str | None
    triggered_action: str | None  # "space" | "backspace" | "speak" | None
    current_text: str
    candidate_letter: str | None
    candidate_progress: float
    active_control_gesture: str | None
    gesture_progress: float
    locked_letter: str | None


class StableTextBuffer:
    """
    Acumula letras reconocidas aplicando:
    1. Filtro de estabilidad temporal: exige `stability_frames` frames consecutivos de la
       misma letra con `confidence >= confidence_threshold`.
    2. Bloqueo anti-repetición: tras aceptar una letra, exige que la mano desaparezca o
       cambie durante `release_frames` frames antes de permitir repetir la misma letra.
    3. Gestos de control sostenidos: espacio, borrar última letra y confirmar/hablar.
    """

    def __init__(
        self,
        stability_frames: int = 12,
        confidence_threshold: float = 0.70,
        release_frames: int = 4,
        control_gesture_hold_frames: int = 20,
        gestures_config: dict[str, str] | None = None,
    ) -> None:
        if stability_frames < 1:
            raise ValueError("stability_frames debe ser >= 1")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold debe estar entre 0.0 y 1.0")
        if release_frames < 1:
            raise ValueError("release_frames debe ser >= 1")

        self.stability_frames = stability_frames
        self.confidence_threshold = confidence_threshold
        self.release_frames = release_frames
        self.control_gesture_hold_frames = max(1, control_gesture_hold_frames)

        default_gestures = {
            "space": "mouth_open",
            "backspace": "brow_down",
            "speak": "brow_up",
        }
        self.gestures_config = gestures_config or default_gestures

        self._chars: list[str] = []
        self._candidate_letter: str | None = None
        self._candidate_streak: int = 0
        self._locked_letter: str | None = None
        self._release_counter: int = 0

        self._current_gesture_action: str | None = None
        self._gesture_streak: int = 0
        self._gesture_locked: bool = False

    @property
    def text(self) -> str:
        return "".join(self._chars)

    @property
    def locked_letter(self) -> str | None:
        return self._locked_letter

    def add_space(self) -> bool:
        """Agrega un espacio si el texto no está vacío ni termina ya en espacio."""
        if self._chars and self._chars[-1] != " ":
            self._chars.append(" ")
            self._locked_letter = None
            return True
        return False

    def backspace(self) -> bool:
        """Borra el último carácter del buffer."""
        if self._chars:
            self._chars.pop()
            self._locked_letter = None
            return True
        return False

    def clear(self) -> None:
        """Limpia por completo el texto y reinicia contadores de estabilidad."""
        self._chars.clear()
        self._candidate_letter = None
        self._candidate_streak = 0
        self._locked_letter = None
        self._release_counter = 0
        self._current_gesture_action = None
        self._gesture_streak = 0
        self._gesture_locked = False

    def update(
        self,
        letter: str | None,
        confidence: float,
        active_gestures: set[str] | None = None,
    ) -> BufferUpdateResult:
        """
        Actualiza el buffer con la predicción del frame actual y los gestos de control activos.
        """
        active_gestures = active_gestures or set()
        committed_letter: str | None = None
        triggered_action: str | None = None

        # 1. Evaluar gestos de control (speak, backspace, space)
        matched_action: str | None = None
        for action in ("speak", "backspace", "space"):
            gesture_name = self.gestures_config.get(action, "none")
            if gesture_name and gesture_name != "none" and gesture_name in active_gestures:
                matched_action = action
                break

        if matched_action is None:
            self._current_gesture_action = None
            self._gesture_streak = 0
            self._gesture_locked = False
        else:
            if matched_action == self._current_gesture_action:
                if not self._gesture_locked:
                    self._gesture_streak += 1
                    if self._gesture_streak >= self.control_gesture_hold_frames:
                        triggered_action = matched_action
                        self._gesture_locked = True
                        self._gesture_streak = 0
                        if matched_action == "space":
                            self.add_space()
                        elif matched_action == "backspace":
                            self.backspace()
            else:
                self._current_gesture_action = matched_action
                self._gesture_streak = 1
                self._gesture_locked = False

        gesture_progress = (
            min(1.0, self._gesture_streak / self.control_gesture_hold_frames)
            if self._current_gesture_action and not self._gesture_locked
            else 0.0
        )

        # 2. Evaluar liberación del bloqueo anti-repetición
        valid_prediction = (
            letter is not None
            and len(letter.strip()) > 0
            and confidence >= self.confidence_threshold
        )
        clean_letter = letter.strip().upper() if (letter and valid_prediction) else None

        if self._locked_letter is not None:
            if clean_letter != self._locked_letter:
                self._release_counter += 1
                if self._release_counter >= self.release_frames:
                    self._locked_letter = None
                    self._release_counter = 0
            else:
                self._release_counter = 0

        # 3. Evaluar estabilidad de la letra candidata
        if clean_letter is None:
            self._candidate_letter = None
            self._candidate_streak = 0
        elif clean_letter == self._locked_letter:
            # Sigue mostrando la misma letra recién aceptada sin haber soltado/cambiado
            self._candidate_letter = None
            self._candidate_streak = 0
        else:
            if clean_letter == self._candidate_letter:
                self._candidate_streak += 1
            else:
                self._candidate_letter = clean_letter
                self._candidate_streak = 1

            if self._candidate_streak >= self.stability_frames:
                self._chars.append(clean_letter)
                committed_letter = clean_letter
                self._locked_letter = clean_letter
                self._release_counter = 0
                self._candidate_letter = None
                self._candidate_streak = 0

        candidate_progress = (
            min(1.0, self._candidate_streak / self.stability_frames)
            if self._candidate_letter is not None
            else 0.0
        )

        return BufferUpdateResult(
            committed_letter=committed_letter,
            triggered_action=triggered_action,
            current_text=self.text,
            candidate_letter=self._candidate_letter,
            candidate_progress=candidate_progress,
            active_control_gesture=self._current_gesture_action,
            gesture_progress=gesture_progress,
            locked_letter=self._locked_letter,
        )

