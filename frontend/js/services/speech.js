/**
 * Dictado del maestro con la Web Speech API del navegador (Chrome/Edge).
 * Si backend ofrece su propio reconocimiento de voz, reemplaza este módulo
 * manteniendo los mismos callbacks.
 */

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;

export const isSupported = () => Boolean(Recognition);

/**
 * @param {{
 *   onPartial: (text: string) => void,  // texto acumulado mientras habla
 *   onSilence: () => void,              // dejó de hablar
 *   onError: (reason: 'denied'|'error') => void
 * }} handlers
 * @returns {() => void} función para detener
 */
export function startDictation({ onPartial, onSilence, onError }) {
  const rec = new Recognition();
  rec.lang = 'es-MX';
  rec.interimResults = true;
  rec.continuous = false;

  rec.addEventListener('result', (event) => {
    const text = Array.from(event.results).map((r) => r[0].transcript).join(' ').trim();
    onPartial(text);
  });
  rec.addEventListener('speechend', onSilence);
  rec.addEventListener('error', (event) => {
    onError(event.error === 'not-allowed' || event.error === 'service-not-allowed' ? 'denied' : 'error');
  });

  rec.start();
  return () => rec.abort();
}
