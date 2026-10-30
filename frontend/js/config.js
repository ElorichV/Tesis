/**
 * Configuración de Voz abierta.
 *
 * Para cambiar valores sin tocar este archivo, define antes de cargar main.js:
 *   <script>window.AULA_CONFIG = { DEMO_MODE: false, API_URL: 'http://localhost:8000' };</script>
 *
 * Nunca pongas aquí claves de API: este archivo llega al navegador.
 */
const DEFAULTS = {
  /** true = respuestas simuladas y panel «Demo» visible. */
  DEMO_MODE: true,

  /** URL base del backend en Python (la define el equipo de backend). */
  API_URL: '',

  /** WebSocket del reconocimiento de señas en vivo. */
  RECOGNITION_WS_URL: '',

  /** Límite de caracteres por mensaje del maestro. */
  MAX_MESSAGE_LENGTH: 300,

  /** Silencio o manos abajo antes del envío automático (igual a $auto-send-ms en Sass). */
  AUTO_SEND_DELAY_MS: 1400,

  /** Intervalo de envío de cuadros JPEG al reconocimiento en vivo. */
  FRAME_INTERVAL_MS: 100,

  /** Ancho de captura JPEG enviado al reconocimiento en vivo. */
  CAPTURE_WIDTH: 640,

  /** Tiempos de la simulación. */
  DEMO: {
    CAMERA_PERMISSION_MS: 1000,
    SIGN_WORD_MS: 420,
    DICTATION_WORD_MS: 380,
    ANIMATION_MS: 1500,
  },
};

export const CONFIG = Object.freeze({
  ...DEFAULTS,
  ...(window.AULA_CONFIG || {}),
  DEMO: { ...DEFAULTS.DEMO, ...((window.AULA_CONFIG || {}).DEMO || {}) },
});
