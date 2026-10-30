/**
 * Panel del alumno: animación, cámara, turno, texto adaptado y controles.
 */
import { CONFIG } from '../config.js';
import { $, $$, showView } from '../utils/dom.js';
import {
  CAMERA_LABELS,
  getCameraView,
  getChecks,
  getLastTeacherMessage,
  getStageView,
  getSubtitle,
  getTurn,
  isSigning,
} from '../selectors.js';

const AVATAR_BY_VARIANT = { alumno: 'avatar--alumno', maestro: 'avatar--maestro', neutral: 'avatar--neutral' };
const ICON_BY_CHECK = { ok: 'icon-check', warn: 'icon-alert', idle: 'icon-circle' };
const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * @param {HTMLElement} root  sección .panel--alumno
 * @param {ReturnType<import('../state.js').createStore>} store
 * @param {ReturnType<import('../controller.js').createController>} controller
 */
export function initAlumno(root, store, controller) {
  const el = {
    stage: $('[data-stage]', root),
    stageViews: $$('[data-stage-view]', root),
    stageLiveText: $('[data-stage-live-text]', root),
    stageVideo: $('[data-stage-video]', root),
    stageAvatar: $('[data-stage-avatar]', root),

    camera: $('[data-camera]', root),
    cameraViews: $$('[data-camera-view]', root),
    cameraChip: $('[data-camera-chip]', root),
    cameraVideo: $('[data-camera-video]', root),

    turn: $('[data-turn]', root),
    turnAvatar: $('[data-turn-avatar]', root),
    turnText: $('[data-turn-text]', root),
    turnSub: $('[data-turn-sub]', root),

    speedButtons: $$('[data-speed]', root),

    subtitleLabel: $('[data-subtitle-label]', root),
    subtitleText: $('[data-subtitle-text]', root),
    toggleOriginal: $('[data-action="toggle-original"]', root),
    toggleLarge: $('[data-action="toggle-large-text"]', root),

    confirm: $('[data-confirm]', root),
    confirmLabel: $('[data-confirm-label]', root),
    confirmText: $('[data-confirm-text]', root),
    confirmProgress: $('[data-confirm-progress]', root),

    cameraLabel: $('[data-camera-label]', root),
    cameraGroups: $$('[data-camera-group]', root),
    cameraDemoNote: $('[data-camera-demo-note]', root),
    checklist: $('[data-checklist]', root),
    cameraOptions: $$('[data-camera-option]', root),
    pauseButton: $('[data-camera-pause-label]', root),

    writer: $('[data-writer]', root),
    writerInput: $('[data-writer-input]', root),
    writerToggle: $('[data-action="toggle-writer"]', root),
  };

  el.cameraDemoNote.hidden = !CONFIG.DEMO_MODE;

  // ---- Eventos propios del panel ---------------------------------------------
  el.speedButtons.forEach((button) => {
    button.addEventListener('click', () => controller.setSpeed(button.dataset.speed));
  });
  el.cameraOptions.forEach((button) => {
    button.addEventListener('click', () => controller.setCameraView(button.dataset.cameraOption));
  });
  el.writer.addEventListener('submit', (event) => {
    event.preventDefault();
    if (controller.sendStudentText(el.writerInput.value)) el.writerInput.value = '';
    el.writerInput.focus();
  });

  // ---- Pintado -------------------------------------------------------------------

  function renderStage(s) {
    const view = getStageView(s);
    const last = getLastTeacherMessage(s);
    const hasAvatar = view === 'ready' && Boolean(last?.secuencia?.length);
    const hasVideo = view === 'ready' && !hasAvatar && Boolean(last?.animationUrl);

    showView(el.stageViews, 'data-stage-view', hasAvatar || hasVideo ? '' : view === 'ready' ? 'unavailable' : view);
    el.stageVideo.hidden = !hasVideo;
    el.stageAvatar.hidden = !hasAvatar;
    if (hasVideo && el.stageVideo.getAttribute('src') !== last.animationUrl) {
      el.stageVideo.src = last.animationUrl;
      el.stageVideo.playbackRate = Number(s.speed);
    }
    el.stage.classList.toggle('stage--live', view === 'live');
    el.stage.classList.toggle('stage--error', view === 'error');
    el.stageLiveText.textContent = `${s.draft}…`;
  }

  function renderTurn(s) {
    const turn = getTurn(s);
    el.turn.className = `turn turn--${turn.variant}`;
    el.turnAvatar.className = `avatar avatar--sm ${AVATAR_BY_VARIANT[turn.variant]}`;
    el.turnAvatar.textContent = turn.avatar;
    el.turnText.textContent = turn.text;
    el.turnSub.textContent = turn.sub;
  }

  function renderPlayer(s) {
    el.speedButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.speed === s.speed)));
  }

  function renderSubtitle(s) {
    const sub = getSubtitle(s);
    el.subtitleLabel.textContent = sub.label;
    el.subtitleText.textContent = sub.text;
    el.subtitleText.classList.toggle('subtitle__text--large', s.largeText);

    const showingOriginal = s.subtitleMode === 'original';
    el.toggleOriginal.disabled = !sub.canToggle;
    el.toggleOriginal.setAttribute('aria-pressed', String(showingOriginal));
    el.toggleOriginal.textContent = showingOriginal ? 'Ver sencillo' : 'Ver original';
    el.toggleLarge.setAttribute('aria-pressed', String(s.largeText));
  }

  function renderConfirm(s) {
    const signing = isSigning(s);
    const ending = s.signPhase === 'ending';
    el.confirm.hidden = !signing;
    el.confirmLabel.textContent = ending ? 'Entendimos:' : 'Interpretando tus señas…';
    el.confirmText.textContent = ending ? (s.interim ? `«${s.interim}»` : '—') : `${s.interim}…`;
    el.confirmText.classList.toggle('confirm__text--large', s.largeText);
    el.confirmProgress.hidden = !ending; // al mostrarse, la animación CSS reinicia
  }

  function renderCamera(s) {
    const view = getCameraView(s);
    const signingNow = s.signPhase === 'signing';

    showView(el.cameraViews, 'data-camera-view', view);
    el.cameraVideo.hidden = CONFIG.DEMO_MODE || view !== 'video';
    el.camera.classList.toggle('camera--active', s.camera === 'active' && s.cameraView !== 'hidden');
    el.camera.classList.toggle('camera--warn', s.camera === 'denied' || s.camera === 'nocam');
    el.camera.setAttribute('aria-label', `Cámara: ${CAMERA_LABELS[s.camera].toLowerCase()}`);

    const showChip = s.camera === 'active' && s.cameraView !== 'hidden';
    el.cameraChip.hidden = !showChip;
    el.cameraChip.className = `live-chip${signingNow ? '' : ' live-chip--idle'}`;
    el.cameraChip.textContent = signingNow ? 'Manos detectadas' : `En vivo${CONFIG.DEMO_MODE ? ' · simulado' : ''}`;
  }

  function renderCameraControls(s, prev) {
    el.cameraLabel.textContent = CAMERA_LABELS[s.camera];

    const groupByStatus = { off: 'off', asking: '', active: 'active', paused: 'active', denied: 'denied', nocam: 'nocam' };
    showView(el.cameraGroups, 'data-camera-group', groupByStatus[s.camera]);

    el.cameraOptions.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.cameraOption === s.cameraView)));
    el.pauseButton.textContent = s.camera === 'paused' ? 'Reanudar' : 'Pausar';

    const checksChanged = !prev || prev.quality !== s.quality || prev.signPhase !== s.signPhase || !el.checklist.children.length;
    if (checksChanged) renderChecklist(getChecks(s));
  }

  function renderChecklist(checks) {
    el.checklist.replaceChildren(...checks.map((check) => {
      const li = document.createElement('li');
      li.className = `checklist__item checklist__item--${check.kind}`;

      const svg = document.createElementNS(SVG_NS, 'svg');
      svg.setAttribute('class', 'icon icon--sm');
      svg.setAttribute('aria-hidden', 'true');
      const use = document.createElementNS(SVG_NS, 'use');
      use.setAttribute('href', `#${ICON_BY_CHECK[check.kind]}`);
      svg.append(use);

      const text = document.createElement('span');
      text.textContent = check.text;
      li.append(svg, text);
      return li;
    }));
  }

  function renderWriter(s, prev) {
    el.writer.hidden = !s.writerOpen;
    el.writerToggle.textContent = s.writerOpen ? 'Ocultar campo de texto' : 'Prefiero escribir';
    el.writerToggle.setAttribute('aria-expanded', String(s.writerOpen));
    if (s.writerOpen && prev && !prev.writerOpen) el.writerInput.focus();
  }

  function render(s, prev) {
    renderStage(s);
    renderTurn(s);
    renderPlayer(s);
    renderSubtitle(s);
    renderConfirm(s);
    renderCamera(s);
    renderCameraControls(s, prev);
    renderWriter(s, prev);
  }

  store.subscribe(render);
  render(store.get());
}
