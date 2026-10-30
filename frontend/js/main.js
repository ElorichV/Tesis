/**
 * Voz abierta · punto de entrada del traductor en vivo.
 */
import { $ } from './utils/dom.js';
import { createStore, initialState } from './state.js';
import { createController } from './controller.js';
import { initAlumno } from './ui/alumno.js';
import { initMaestro } from './ui/maestro.js';
import { initFeedback } from './ui/feedback.js';
import { initDemoPanel } from './ui/demo-panel.js';

const store = createStore(initialState);

const controller = createController(store, {
  stageVideo: $('[data-stage-video]'),
  cameraVideo: $('[data-camera-video]'),
  stageAvatar: $('[data-stage-avatar]'),
});

initAlumno($('.panel--alumno'), store, controller);
initMaestro($('.panel--maestro'), store, controller);
initFeedback(document, store);
initDemoPanel(document, store, controller);

// Un solo manejador para todos los botones con data-action="…"
document.addEventListener('click', (event) => {
  const button = event.target.closest('[data-action]');
  if (!button || button.disabled) return;
  controller.actions[button.dataset.action]?.();
});

controller.init();
