/**
 * Cliente del backend en Python.
 * Contrato PROPUESTO: confírmalo con el equipo de backend y ajusta aquí.
 * Todas las rutas parten de CONFIG.API_URL.
 */
import { CONFIG } from '../config.js';

const url = (path) => `${CONFIG.API_URL.replace(/\/$/, '')}${path}`;

/**
 * GET /salud → 200 si el servicio responde.
 * La interfaz solo muestra «Conectado» después de esta comprobación.
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
  try {
    const res = await fetch(url('/salud'), { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * POST /traducir-texto
 * Petición:  { "texto": "Hoy vamos a ver la fotosíntesis." }
 * Respuesta: {
 *   "texto_sencillo": "Hoy vamos a estudiar la fotosíntesis.",
 *   "animacion_url": "https://…/animacion.mp4"   // o null si no hay animación
 * }
 * @param {string} text
 * @param {{ signal?: AbortSignal }} [options]
 */
export async function translateText(text, { signal } = {}) {
  const res = await fetch(url('/traducir-texto'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texto: text }),
    signal,
  });
  if (!res.ok) throw new Error(`translateText: HTTP ${res.status}`);
  return res.json();
}

/**
 * WebSocket de reconocimiento de señas en vivo (CONFIG.RECOGNITION_WS_URL).
 *
 * El navegador envía cuadros de la cámara (Blob JPEG) o puntos del cuerpo.
 * El servidor responde con mensajes JSON:
 *   { "tipo": "manos",        "detectadas": true }
 *   { "tipo": "parcial",      "texto": "¿Qué es la…" }
 *   { "tipo": "final",        "texto": "¿Qué es la fotosíntesis?" }
 *   { "tipo": "no_reconocido", "motivo": "sin_modelo" }
 *   { "tipo": "calidad",      "encuadre": true, "luz": false }
 *
 * @param {(evt: {type: string, text?: string, reason?: string, framing?: boolean, light?: boolean}) => void} onEvent
 * @param {() => void} onDisconnect
 */
export function connectRecognition(onEvent, onDisconnect) {
  const ws = new WebSocket(CONFIG.RECOGNITION_WS_URL);
  ws.binaryType = 'blob';

  ws.addEventListener('message', ({ data }) => {
    let msg;
    try { msg = JSON.parse(data); } catch { return; }
    const map = {
      manos: () => msg.detectadas && onEvent({ type: 'hands' }),
      parcial: () => onEvent({ type: 'partial', text: msg.texto }),
      final: () => onEvent({ type: 'final', text: msg.texto }),
      no_reconocido: () => onEvent({ type: 'unrecognized', reason: msg.motivo }),
      calidad: () => onEvent({ type: 'quality', framing: msg.encuadre, light: msg.luz }),
    };
    map[msg.tipo]?.();
  });
  ws.addEventListener('close', onDisconnect);
  ws.addEventListener('error', onDisconnect);

  return {
    send(frame) { if (ws.readyState === WebSocket.OPEN) ws.send(frame); },
    close() { ws.close(); },
  };
}
