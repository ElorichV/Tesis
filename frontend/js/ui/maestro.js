/**
 * Panel del maestro: conversación y campo para escribir o dictar.
 */
import { CONFIG } from '../config.js';
import { $, setIcon } from '../utils/dom.js';
import { isListening, isSigning } from '../selectors.js';

const AUTHOR = {
  alumno: { label: 'Alumno', avatar: 'A', avatarClass: 'avatar--alumno' },
  maestro: { label: 'Maestro', avatar: 'M', avatarClass: 'avatar--maestro' },
  sistema: { label: 'Aviso', avatar: '!', avatarClass: 'avatar--system' },
};

const VIA = { lsm: 'Interpretado de LSM', texto: 'Escribió', voz: 'Dictó' };

const ANIMATION_STATUS = {
  generating: 'Preparando señas…',
  ready: 'Señas enviadas al alumno',
  unavailable: CONFIG.DEMO_MODE ? 'Animación no disponible en la demo' : 'Animación no disponible',
  error: 'No se generaron las señas',
};

const MAX_INPUT_HEIGHT = 160;

/**
 * @param {HTMLElement} root  sección .panel--maestro
 * @param {ReturnType<import('../state.js').createStore>} store
 * @param {ReturnType<import('../controller.js').createController>} controller
 */
export function initMaestro(root, store, controller) {
  const el = {
    pauseButton: $('[data-action="teacher-pause-camera"]', root),
    copyButton: $('[data-action="copy-conversation"]', root),
    clearButton: $('[data-action="clear-conversation"]', root),
    conversation: $('[data-conversation]', root),
    template: $('#message-template'),
    composer: $('[data-composer]', root),
    input: $('[data-composer-input]', root),
    mic: $('[data-action="toggle-mic"]', root),
    micIcon: $('[data-mic-icon]', root),
    send: $('[data-composer-send]', root),
    feedback: $('[data-composer-feedback]', root),
    progress: $('[data-composer-progress]', root),
  };

  // ---- Campo de texto: crece con el contenido --------------------------------
  function fitInput() {
    el.input.style.height = 'auto';
    el.input.style.height = `${Math.min(el.input.scrollHeight, MAX_INPUT_HEIGHT)}px`;
  }

  el.input.addEventListener('input', () => {
    controller.setDraft(el.input.value);
    fitInput();
  });

  el.input.addEventListener('keydown', (event) => {
    // Enter envía; Mayús + Enter salta de línea
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      controller.sendTeacherMessage();
    }
  });

  el.composer.addEventListener('submit', (event) => {
    event.preventDefault();
    controller.sendTeacherMessage();
  });

  // ---- Conversación -------------------------------------------------------------

  function buildMessage(message) {
    const author = AUTHOR[message.author];
    const node = el.template.content.firstElementChild.cloneNode(true);
    if (message.author === 'sistema') node.classList.add('message--system');

    const avatar = node.querySelector('[data-message-avatar]');
    avatar.classList.add(author.avatarClass);
    avatar.textContent = author.avatar;

    node.querySelector('[data-message-author]').textContent = author.label;
    node.querySelector('[data-message-via]').textContent = message.via ? VIA[message.via] : '';
    node.querySelector('[data-message-time]').textContent = message.time || '';
    node.querySelector('[data-message-text]').textContent = message.text;

    if (message.author === 'maestro' && message.animation) {
      const status = node.querySelector('[data-message-status]');
      status.hidden = false;
      status.textContent = ANIMATION_STATUS[message.animation];
    }
    return node;
  }

  function buildInterim(s) {
    const ending = s.signPhase === 'ending';
    const node = buildMessage({ author: 'alumno', text: ending ? (s.interim || '…') : `${s.interim}…` });
    node.classList.add('message--interim');
    node.querySelector('[data-message-via]').textContent = ending ? 'Terminando…' : 'Señando…';

    const discard = document.createElement('button');
    discard.type = 'button';
    discard.className = 'text-button message__action';
    discard.dataset.action = 'discard-signs';
    discard.textContent = 'Descartar';
    node.querySelector('.message__text').after(discard);
    return node;
  }

  function buildEmpty() {
    const li = document.createElement('li');
    li.className = 'conversation__empty';
    li.textContent = 'La conversación aparecerá aquí.';
    return li;
  }

  function renderConversation(s, prev) {
    const changed = !prev || prev.messages !== s.messages || prev.interim !== s.interim || prev.signPhase !== s.signPhase;
    if (!changed) return;

    const nodes = s.messages.map(buildMessage);
    if (isSigning(s)) nodes.push(buildInterim(s));
    el.conversation.replaceChildren(...(nodes.length ? nodes : [buildEmpty()]));
    el.conversation.scrollTop = el.conversation.scrollHeight;
  }

  // ---- Feedback debajo del campo ---------------------------------------------------

  function hint(text, { warn = false, alert = false, pulse = false } = {}) {
    const p = document.createElement('p');
    p.className = `hint${warn ? ' hint--warn' : ''}`;
    if (alert) p.setAttribute('role', 'alert');
    if (pulse) {
      const dot = document.createElement('span');
      dot.className = 'pulse-dot';
      dot.setAttribute('aria-hidden', 'true');
      p.append(dot);
    }
    p.append(text);
    return p;
  }

  function renderFeedback(s) {
    const items = [];
    if (isListening(s) && s.micPhase !== 'ending') items.push(hint('Escuchando… habla con frases cortas.', { warn: true, pulse: true }));
    if (s.mic === 'denied') items.push(hint('Sin permiso para el micrófono. Puedes escribir.', { warn: true, alert: true }));
    if (s.draftError === 'long') {
      const extra = s.draft.length - CONFIG.MAX_MESSAGE_LENGTH;
      items.push(hint(`Sobran ${extra.toLocaleString('es-MX')} caracteres. Usa frases cortas.`, { warn: true, alert: true }));
    }
    if (isSigning(s) && !isListening(s)) items.push(hint('El alumno está señando. Espera a que termine.', { warn: true }));
    el.feedback.replaceChildren(...items);
  }

  // ---- Pintado general ----------------------------------------------------------------

  function render(s, prev) {
    const listening = isListening(s);
    const cameraOn = s.camera === 'active' || s.camera === 'paused';

    // Encabezado
    el.pauseButton.disabled = !cameraOn;
    el.pauseButton.textContent = s.camera === 'paused' ? 'Reanudar cámara' : 'Pausar cámara';
    el.copyButton.disabled = !s.messages.length;
    el.clearButton.disabled = !s.messages.length;

    renderConversation(s, prev);

    // Campo
    if (el.input.value !== s.draft) {
      el.input.value = s.draft;
      fitInput();
    }
    el.input.readOnly = listening;
    el.input.placeholder = listening ? 'Escuchando…' : 'Escribe o dicta una frase corta…';
    el.input.setAttribute('aria-invalid', String(Boolean(s.draftError)));
    el.composer.classList.toggle('composer--listening', listening);
    el.composer.classList.toggle('composer--invalid', Boolean(s.draftError));

    // Micrófono
    el.mic.disabled = !s.online;
    el.mic.setAttribute('aria-pressed', String(listening));
    const micLabel = listening ? 'Detener dictado y enviar' : 'Dictar con el micrófono';
    el.mic.setAttribute('aria-label', micLabel);
    el.mic.title = micLabel;
    setIcon(el.micIcon, listening ? 'icon-stop' : 'icon-mic');

    // Enviar
    el.send.disabled = !s.online || listening || !s.draft.trim();

    renderFeedback(s);
    el.progress.hidden = !(listening && s.micPhase === 'ending');
  }

  store.subscribe(render);
  render(store.get());
}
