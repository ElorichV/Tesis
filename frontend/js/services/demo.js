/**
 * Simuladores del modo demostración.
 * Emiten los MISMOS eventos que los servicios reales, así el controlador
 * no distingue entre demo y producción.
 */
import { CONFIG } from '../config.js';

/** Simula que el navegador concede el permiso de la cámara. */
export function simulateCameraPermission(onGranted) {
  const t = setTimeout(onGranted, CONFIG.DEMO.CAMERA_PERMISSION_MS);
  return () => clearTimeout(t);
}

/**
 * Simula el reconocimiento de señas: manos → parcial (palabra a palabra) → final.
 * @param {string} phrase
 * @param {{ fail?: boolean }} options
 * @param {(evt: object) => void} emit  mismos eventos que services/api.connectRecognition
 * @returns {() => void} cancelar
 */
export function simulateSigns(phrase, { fail = false }, emit) {
  const words = phrase.split(' ');
  let i = 0;
  emit({ type: 'hands' });

  const timer = setInterval(() => {
    i += 1;
    if (!fail && i <= words.length) {
      emit({ type: 'partial', text: words.slice(0, i).join(' ') });
      return;
    }
    if (fail && i < 3) return;
    clearInterval(timer);
    emit(fail ? { type: 'unrecognized' } : { type: 'final', text: phrase });
  }, CONFIG.DEMO.SIGN_WORD_MS);

  return () => clearInterval(timer);
}

/**
 * Simula el dictado: el texto crece palabra a palabra y luego hay silencio.
 * @param {string} phrase
 * @param {{ onPartial: (t: string) => void, onSilence: () => void }} handlers
 * @returns {() => void} cancelar
 */
export function simulateDictation(phrase, { onPartial, onSilence }) {
  const words = phrase.split(' ');
  let i = 0;
  const timer = setInterval(() => {
    i += 1;
    if (i <= words.length) {
      onPartial(words.slice(0, i).join(' '));
      return;
    }
    clearInterval(timer);
    onSilence();
  }, CONFIG.DEMO.DICTATION_WORD_MS);
  return () => clearInterval(timer);
}

/**
 * Simula la generación de la animación. En la demo no hay video real.
 * @returns {Promise<{ animation: 'unavailable' }>}
 */
export function simulateAnimation(text, { fail = false } = {}) {
  return new Promise((resolve, reject) => {
    setTimeout(() => (fail ? reject(new Error('demo: animación fallida')) : resolve({ animation: 'unavailable' })),
      CONFIG.DEMO.ANIMATION_MS);
  });
}
