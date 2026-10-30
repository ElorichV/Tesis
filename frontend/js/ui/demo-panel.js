/**
 * Controles de demostración. Solo existen con CONFIG.DEMO_MODE = true.
 */
import { CONFIG } from '../config.js';
import { $, $$ } from '../utils/dom.js';

/**
 * @param {Document} doc
 * @param {ReturnType<import('../state.js').createStore>} store
 * @param {ReturnType<import('../controller.js').createController>} controller
 */
export function initDemoPanel(doc, store, controller) {
  const toggle = $('[data-demo-toggle]', doc);
  const panel = $('[data-demo-panel]', doc);
  if (!CONFIG.DEMO_MODE || !toggle || !panel) return;

  toggle.hidden = false;

  const close = (returnFocus = false) => {
    controller.toggleDemo(false);
    if (returnFocus) toggle.focus();
  };

  toggle.addEventListener('click', () => controller.toggleDemo());
  $('[data-demo-close]', panel).addEventListener('click', () => close(true));
  panel.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') close(true);
  });

  $$('[data-demo]', panel).forEach((button) => {
    button.addEventListener('click', () => {
      controller.demo[button.dataset.demo]?.();
      close();
    });
  });

  const failCheckbox = $('[data-demo-fail-animation]', panel);
  failCheckbox.addEventListener('change', () => controller.setFailNextAnimation(failCheckbox.checked));

  function render(s, prev) {
    panel.hidden = !s.demoOpen;
    toggle.setAttribute('aria-expanded', String(s.demoOpen));
    failCheckbox.checked = s.failNextAnimation;
    if (s.demoOpen && prev && !prev.demoOpen) $('[data-demo]', panel).focus();
  }

  store.subscribe(render);
  render(store.get());
}
