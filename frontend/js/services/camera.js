/**
 * Cámara del alumno (getUserMedia).
 * Solo se usa fuera del modo demostración.
 */
import { CONFIG } from '../config.js';

let stream = null;
let frameTimer = null;

/**
 * Enciende la cámara y la muestra en el <video>.
 * @param {HTMLVideoElement} video
 */
export async function start(video) {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play().catch(() => {});
}

/** Apaga la cámara y libera el dispositivo. */
export function stop() {
  clearInterval(frameTimer);
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
}

/** Pausa o reanuda el envío de video sin pedir permiso de nuevo. */
export function setPaused(paused) {
  stream?.getVideoTracks().forEach((t) => { t.enabled = !paused; });
}

/**
 * Traduce un error de getUserMedia al estado de la interfaz.
 * @returns {'denied'|'nocam'}
 */
export function errorToStatus(err) {
  return err && (err.name === 'NotAllowedError' || err.name === 'SecurityError') ? 'denied' : 'nocam';
}

/**
 * Envía cuadros JPEG al reconocimiento cada `intervalMs`.
 * Si backend prefiere puntos del cuerpo (MediaPipe), reemplaza esta función.
 * @param {HTMLVideoElement} video
 * @param {(blob: Blob) => void} onFrame
 */
export function streamFrames(video, onFrame, intervalMs = 200) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  clearInterval(frameTimer);
  frameTimer = setInterval(() => {
    if (!stream || video.readyState < 2) return;
    canvas.width = CONFIG.CAPTURE_WIDTH;
    canvas.height = Math.round((video.videoHeight / video.videoWidth) * CONFIG.CAPTURE_WIDTH) || 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => blob && onFrame(blob), 'image/jpeg', 0.7);
  }, intervalMs);
}
