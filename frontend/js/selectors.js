/**
 * Valores derivados del estado. Funciones puras, sin DOM.
 */

/** Último mensaje del maestro. */
export function getLastTeacherMessage(state) {
  for (let i = state.messages.length - 1; i >= 0; i -= 1) {
    if (state.messages[i].author === 'maestro') return state.messages[i];
  }
  return null;
}

export const isSigning = (state) => state.signPhase === 'signing' || state.signPhase === 'ending';
export const isListening = (state) => state.mic === 'listening';

/**
 * Indicador de turno compartido por ambos usuarios.
 * @returns {{ variant: 'alumno'|'maestro'|'neutral', avatar: string, text: string, sub: string }}
 */
export function getTurn(state) {
  const last = getLastTeacherMessage(state);

  if (isSigning(state)) {
    const ending = state.signPhase === 'ending';
    return {
      variant: 'alumno',
      avatar: 'A',
      text: ending ? 'El alumno terminó de señar' : 'Ahora seña el alumno',
      sub: ending ? 'El mensaje se enviará en un momento.' : 'Maestro: espera a que termine.',
    };
  }
  if (isListening(state)) {
    const ending = state.micPhase === 'ending';
    return {
      variant: 'maestro',
      avatar: 'M',
      text: ending ? 'El maestro terminó de hablar' : 'El maestro está hablando',
      sub: ending ? 'El mensaje se enviará en un momento.' : 'Alumno: las señas aparecen mientras habla.',
    };
  }
  if (last && last.animation === 'generating') {
    return { variant: 'maestro', avatar: 'M', text: 'Preparando señas para el alumno', sub: 'Alumno: mira la animación.' };
  }
  if (state.draft.length) {
    return { variant: 'maestro', avatar: 'M', text: 'El maestro está escribiendo', sub: 'Alumno: espera su mensaje.' };
  }
  if (state.camera === 'active') {
    return { variant: 'neutral', avatar: '·', text: 'Turno libre', sub: 'El alumno puede señar o el maestro escribir.' };
  }
  if (state.camera === 'paused') {
    return { variant: 'neutral', avatar: '·', text: 'Cámara en pausa', sub: 'Solo el maestro puede escribir.' };
  }
  return { variant: 'neutral', avatar: '·', text: 'Activa la cámara para empezar', sub: 'El alumno seña; el maestro escribe o dicta.' };
}

/**
 * Texto que ve el alumno debajo de la animación.
 * Español sencillo por defecto; «Ver original» muestra el texto exacto.
 */
export function getSubtitle(state) {
  const last = getLastTeacherMessage(state);

  if (isListening(state)) {
    return { label: 'En vivo · el maestro está hablando', text: `${state.draft}…`, canToggle: false };
  }
  if (!last) return { label: 'Mensaje del maestro', text: '—', canToggle: false };
  if (state.subtitleMode === 'original') {
    return { label: 'Texto original del maestro', text: last.text, canToggle: Boolean(last.simple) };
  }
  if (last.simple) return { label: 'Español sencillo', text: last.simple, canToggle: true };
  return { label: 'Texto original · la versión sencilla la dará el servicio', text: last.text, canToggle: false };
}

/** Vista del escenario de la animación. */
export function getStageView(state) {
  if (isListening(state)) return 'live';
  const last = getLastTeacherMessage(state);
  if (!last || !last.animation) return 'idle';
  return last.animation; // generating | ready | unavailable | error
}

/** Indicadores de calidad de la cámara (texto + tipo de ícono). */
export function getChecks(state) {
  const signing = isSigning(state);
  return [
    state.quality.framing
      ? { kind: 'ok', text: 'Estás en el encuadre' }
      : { kind: 'warn', text: 'Acércate: que se vean tu cara y tus manos' },
    signing
      ? { kind: 'ok', text: 'Manos detectadas' }
      : { kind: 'idle', text: 'Levanta las manos para señar' },
    state.quality.light
      ? { kind: 'ok', text: 'Luz suficiente' }
      : { kind: 'warn', text: 'Falta luz: busca un lugar más iluminado' },
  ];
}

export const CAMERA_LABELS = {
  off: 'Apagada',
  asking: 'Pidiendo permiso',
  active: 'Activa',
  paused: 'En pausa',
  denied: 'Sin permiso',
  nocam: 'Sin cámara',
};

/** Vista dentro del recuadro de la cámara. */
export function getCameraView(state) {
  switch (state.camera) {
    case 'active': return state.cameraView; // video | points | hidden
    case 'denied':
    case 'nocam': return 'unavailable';
    default: return state.camera; // off | asking | paused
  }
}
