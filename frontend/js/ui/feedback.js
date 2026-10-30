/**
 * Avisos globales: región aria-live, mensaje temporal y franja sin conexión.
 */
import { $, setIcon } from '../utils/dom.js';

/**
 * @param {Document} doc
 * @param {ReturnType<import('../state.js').createStore>} store
 */
export function initFeedback(doc, store) {
  const el = {
    live: $('[data-live-region]', doc),
    toast: $('[data-toast]', doc),
    toastText: $('[data-toast-text]', doc),
    toastIcon: $('[data-toast-icon]', doc),
    toastUndo: $('[data-toast-undo]', doc),
    offline: $('[data-offline-banner]', doc),
  };

  function render(s, prev) {
    // Lector de pantalla: se vacía primero para que repita mensajes iguales
    if (!prev || prev.announcement.id !== s.announcement.id) {
      el.live.textContent = '';
      requestAnimationFrame(() => { el.live.textContent = s.announcement.text; });
    }

    // Mensaje temporal
    el.toast.hidden = !s.toast;
    if (s.toast) {
      el.toastText.textContent = s.toast.text;
      el.toast.classList.toggle('toast--warn', s.toast.tone === 'warn');
      setIcon(el.toastIcon, s.toast.tone === 'warn' ? 'icon-alert' : 'icon-check');
      el.toastUndo.hidden = !s.toast.undo;
    }

    // Sin conexión
    el.offline.hidden = s.online;
  }

  store.subscribe(render);
  render(store.get());
}
