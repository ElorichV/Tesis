/**
 * Estado único de la aplicación.
 * Los módulos de UI se suscriben y se vuelven a pintar cuando cambia.
 */

/**
 * @typedef {'off'|'asking'|'active'|'paused'|'denied'|'nocam'} CameraStatus
 * @typedef {'video'|'points'|'hidden'} CameraView
 * @typedef {'generating'|'ready'|'unavailable'|'error'} AnimationStatus
 *
 * @typedef {Object} Message
 * @property {string} id
 * @property {'alumno'|'maestro'|'sistema'} author
 * @property {'lsm'|'texto'|'voz'} [via]
 * @property {string} text
 * @property {string} [simple]        Español sencillo para el alumno
 * @property {string} time            HH:MM
 * @property {AnimationStatus} [animation]
 * @property {string} [animationUrl]  Video de la animación en LSM
 */

export const initialState = Object.freeze({
  // Cámara del alumno
  camera: /** @type {CameraStatus} */ ('off'),
  cameraView: /** @type {CameraView} */ ('video'),
  quality: { framing: true, light: true },

  // Reconocimiento de señas: null | 'signing' | 'ending'
  signPhase: null,
  interim: '',

  // Dictado del maestro: 'off' | 'listening' | 'denied'
  mic: 'off',
  micPhase: null, // null | 'speaking' | 'ending'

  // Conversación
  messages: /** @type {Message[]} */ ([]),
  draft: '',
  draftError: null, // null | 'long'

  // Preferencias del alumno
  subtitleMode: 'simple', // 'simple' | 'original'
  largeText: false,
  speed: '1',
  writerOpen: false,

  // Sistema
  online: true,
  toast: null, // { text, tone: 'ok'|'info'|'warn', undo: boolean }
  undoSnapshot: null,
  announcement: { id: 0, text: '' },

  // Demo
  demoOpen: false,
  failNextAnimation: false,
});

/**
 * Crea un almacén mínimo con suscripción.
 * @param {object} initial
 */
export function createStore(initial) {
  let state = { ...initial };
  const listeners = new Set();

  return {
    get: () => state,

    /** @param {object|((s: object) => object)} patch */
    set(patch) {
      const prev = state;
      const next = typeof patch === 'function' ? patch(state) : patch;
      state = { ...state, ...next };
      listeners.forEach((fn) => fn(state, prev));
    },

    /** @param {(state: object, prev: object) => void} fn */
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
