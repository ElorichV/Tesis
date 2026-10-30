/**
 * Dictado del maestro con la Web Speech API del navegador (Chrome/Edge).
 * Si backend ofrece su propio reconocimiento de voz, reemplaza este módulo
 * manteniendo los mismos callbacks.
 */

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;

export const isSupported = () => Boolean(Recognition);

function permissionErrorReason(error) {
  if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') return 'denied';
  if (error?.name === 'NotFoundError' || error?.name === 'OverconstrainedError') return 'nomic';
  if (error?.name === 'NotReadableError' || error?.name === 'AbortError') return 'busy';
  return 'error';
}

/**
 * @param {{
 *   onPartial: (text: string, isFinal: boolean) => void,
 *   onSilence: () => void,
 *   onEnd: () => void,
 *   onError: (reason: 'denied'|'nomic'|'network'|'nospeech'|'aborted'|'busy'|'insecure'|'error') => void
 * }} handlers
 * @returns {() => void} detiene el reconocimiento e ignora eventos posteriores
 */
export function startDictation({ onPartial, onSilence, onEnd, onError }) {
  const rec = new Recognition();
  rec.lang = 'es-MX';
  rec.interimResults = true;
  rec.continuous = false;
  let stopped = false;
  const failStart = (reason) => {
    if (stopped) return;
    stopped = true;
    onError(reason);
  };

  rec.addEventListener('result', (event) => {
    if (stopped) return;
    const results = Array.from(event.results);
    const text = results.map((result) => result[0].transcript).join(' ').trim();
    onPartial(text, Boolean(results[results.length - 1]?.isFinal));
  });
  rec.addEventListener('speechend', () => {
    if (!stopped) onSilence();
  });
  rec.addEventListener('end', () => {
    if (!stopped) onEnd?.();
  });
  rec.addEventListener('error', (event) => {
    if (stopped) return;
    const reasons = {
      'not-allowed': 'denied',
      'service-not-allowed': 'denied',
      'audio-capture': 'nomic',
      network: 'network',
      'no-speech': 'nospeech',
      aborted: 'aborted',
    };
    const reason = Object.prototype.hasOwnProperty.call(reasons, event.error)
      ? reasons[event.error]
      : 'error';
    onError(reason);
  });

  const stop = () => {
    if (stopped) return;
    stopped = true;
    try {
      rec.abort();
    } catch {}
  };

  const startRecognition = () => {
    if (stopped) return;
    try {
      rec.start();
    } catch {
      failStart('error');
    }
  };

  const mediaDevices = globalThis.navigator?.mediaDevices;
  if (!mediaDevices?.getUserMedia) {
    failStart('insecure');
    return stop;
  }

  let permissionRequest;
  try {
    permissionRequest = mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    failStart(permissionErrorReason(error));
    return stop;
  }
  Promise.resolve(permissionRequest)
    .then((stream) => {
      stream.getTracks().forEach((track) => {
        try { track.stop(); } catch {}
      });
      startRecognition();
    })
    .catch((error) => failStart(permissionErrorReason(error)));

  return stop;
}
