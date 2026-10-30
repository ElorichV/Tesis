/**
 * Utilidades pequeñas de DOM.
 */

/** querySelector con raíz opcional. */
export const $ = (selector, root = document) => root.querySelector(selector);

/** querySelectorAll como arreglo. */
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

/** Muestra un elemento y oculta sus hermanos con el mismo atributo de vista. */
export function showView(elements, attr, value) {
  elements.forEach((el) => { el.hidden = el.getAttribute(attr) !== value; });
}

/** Cambia el ícono de un <use href="#..."> */
export function setIcon(useEl, id) {
  if (useEl) useEl.setAttribute('href', `#${id}`);
}

/** Hora actual en formato HH:MM. */
export function nowHHMM() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/** Identificador corto único. */
let counter = 0;
export const uid = (prefix = 'm') => `${prefix}${Date.now().toString(36)}${(counter += 1)}`;

/** Copia texto al portapapeles con respaldo. */
export async function copyText(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* usa el respaldo */ }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

/** true si la persona pidió reducir el movimiento. */
export const prefersReducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
