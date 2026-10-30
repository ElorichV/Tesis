/**
 * Controlador: toda la lógica de la app.
 * La UI solo llama acciones de aquí y pinta el estado; nunca cambia el estado directo.
 */
import { CONFIG } from './config.js';
import { initialState } from './state.js';
import { getLastTeacherMessage, isListening, isSigning } from './selectors.js';
import { nowHHMM, uid, copyText } from './utils/dom.js';
import * as api from './services/api.js';
import * as camera from './services/camera.js';
import * as avatar from './services/avatar.js';
import * as speech from './services/speech.js';
import * as demo from './services/demo.js';
import { EXAMPLE_CONVERSATION, SIGN_SCRIPT, DICTATION_SCRIPT, SIMPLE_SPANISH } from './data/demo-data.js';

/**
 * @param {ReturnType<import('./state.js').createStore>} store
 * @param {{ stageVideo: HTMLVideoElement, cameraVideo: HTMLVideoElement, stageAvatar: HTMLElement }} media
 */
export function createController(store, { stageVideo, cameraVideo, stageAvatar }) {
  const { get, set } = store;
  const timers = { toast: 0, signEnd: 0, micEnd: 0 };
  let cancelSigns = null;
  let cancelDictation = null;
  let cancelPermission = null;
  let recognition = null;
  let signIndex = 0;
  let dictationIndex = 0;

  if (!CONFIG.DEMO_MODE && stageAvatar) avatar.init(stageAvatar);

  // ------------------------------------------------------------------------
  // Utilidades internas
  // ------------------------------------------------------------------------

  /** Anuncio para lectores de pantalla (región aria-live). */
  function announce(text) {
    set((s) => ({ announcement: { id: s.announcement.id + 1, text } }));
  }

  function showToast(text, tone = 'info', { undo = false, ms = 4000 } = {}) {
    clearTimeout(timers.toast);
    set({ toast: { text, tone, undo } });
    announce(text);
    timers.toast = setTimeout(() => set({ toast: null, undoSnapshot: null }), ms);
  }

  function addMessage(message) {
    const full = { id: uid(), time: nowHHMM(), ...message };
    set((s) => ({ messages: [...s.messages, full] }));
    return full.id;
  }

  function updateMessage(id, patch) {
    set((s) => ({ messages: s.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)) }));
  }

  // ------------------------------------------------------------------------
  // Cámara del alumno
  // ------------------------------------------------------------------------

  async function cameraOn() {
    set({ camera: 'asking' });
    announce('Esperando permiso de la cámara.');

    if (CONFIG.DEMO_MODE) {
      cancelPermission = demo.simulateCameraPermission(() => {
        set({ camera: 'active' });
        announce('Cámara activa. Esperando señas.');
      });
      return;
    }

    try {
      await camera.start(cameraVideo);
      set({ camera: 'active' });
      announce('Cámara activa. Esperando señas.');
      startRecognition();
    } catch (err) {
      const status = camera.errorToStatus(err);
      set({ camera: status });
      announce(status === 'denied' ? 'Sin permiso para la cámara.' : 'No encontramos una cámara.');
    }
  }

  function cameraTogglePause() {
    const { camera: status } = get();
    if (status === 'active') {
      stopSigning();
      camera.setPaused(true);
      set({ camera: 'paused', signPhase: null, interim: '' });
      announce('Cámara en pausa.');
    } else if (status === 'paused') {
      camera.setPaused(false);
      set({ camera: 'active' });
      announce('Cámara activa.');
    }
  }

  /** @param {'video'|'points'|'hidden'} view */
  function setCameraView(view) {
    set({ cameraView: view });
    const text = { video: 'Vista de video.', points: 'Vista de solo puntos.', hidden: 'Vista oculta. Los indicadores siguen activos.' };
    announce(text[view]);
  }

  // ------------------------------------------------------------------------
  // Reconocimiento de señas (mismos eventos en demo y producción)
  // ------------------------------------------------------------------------

  function handleRecognition(evt) {
    const s = get();
    if (s.camera !== 'active') return;

    switch (evt.type) {
      case 'hands':
        if (!isSigning(s)) {
          set({ signPhase: 'signing', interim: '' });
          announce('El alumno empezó a señar.');
        }
        break;
      case 'partial':
        set({ signPhase: 'signing', interim: evt.text });
        break;
      case 'final':
        set({ signPhase: 'ending', interim: evt.text });
        scheduleSignCommit(evt.text);
        break;
      case 'unrecognized':
        set({ signPhase: 'ending', interim: '' });
        scheduleSignCommit(null, evt.reason);
        break;
      case 'quality':
        set({ quality: { framing: evt.framing ?? s.quality.framing, light: evt.light ?? s.quality.light } });
        break;
      default:
        break;
    }
  }

  /** Envío automático tras bajar las manos (se puede cancelar con «Cancelar»). */
  function scheduleSignCommit(text, reason) {
    clearTimeout(timers.signEnd);
    timers.signEnd = setTimeout(() => {
      set({ signPhase: null, interim: '' });
      if (text) {
        addMessage({ author: 'alumno', via: 'lsm', text });
        announce(`El alumno dijo: ${text}`);
      } else {
        const message = reason === 'sin_modelo'
          ? 'No hay un modelo de señas entrenado. Ejecuta «python -m app collect» y luego «python -m app train».'
          : 'No reconocimos ninguna letra. Sostén cada letra quieta un momento, con la mano dentro del cuadro.';
        addMessage({ author: 'sistema', text: message });
        announce(message);
      }
    }, CONFIG.AUTO_SEND_DELAY_MS);
  }

  function stopSigning() {
    cancelSigns?.();
    cancelSigns = null;
    clearTimeout(timers.signEnd);
  }

  function discardSigns() {
    stopSigning();
    set({ signPhase: null, interim: '' });
    announce('Se descartó la interpretación.');
  }

  function startRecognition() {
    if (!CONFIG.RECOGNITION_WS_URL) return;
    recognition?.close();
    recognition = api.connectRecognition(handleRecognition, () => set({ online: false }));
    camera.streamFrames(cameraVideo, (frame) => recognition?.send(frame), CONFIG.FRAME_INTERVAL_MS);
  }

  // ------------------------------------------------------------------------
  // Maestro: escribir, dictar y generar señas
  // ------------------------------------------------------------------------

  function setDraft(text) {
    const patch = { draft: text };
    if (get().draftError === 'long' && text.length <= CONFIG.MAX_MESSAGE_LENGTH) patch.draftError = null;
    set(patch);
  }

  /**
   * Envía un mensaje del maestro y pide la animación.
   * @param {string} [text]  si se omite, usa el borrador
   * @param {'texto'|'voz'} [via]
   */
  async function sendTeacherMessage(text, via = 'texto') {
    const s = get();
    const content = (text ?? s.draft).trim();
    if (!s.online || !content || isListening(s)) return;
    if (content.length > CONFIG.MAX_MESSAGE_LENGTH) {
      set({ draftError: 'long' });
      return;
    }

    const id = addMessage({ author: 'maestro', via, text: content, animation: 'generating' });
    if (text === undefined) set({ draft: '', draftError: null });
    announce('Preparando señas para el alumno.');
    await requestAnimation(id, content);
  }

  async function requestAnimation(id, text) {
    try {
      if (CONFIG.DEMO_MODE) {
        const fail = get().failNextAnimation;
        set({ failNextAnimation: false });
        const res = await demo.simulateAnimation(text, { fail });
        updateMessage(id, { simple: SIMPLE_SPANISH[text], animation: res.animation });
      } else {
        const res = await api.translateText(text);
        const secuencia = Array.isArray(res.secuencia) ? res.secuencia : [];
        avatar.setSpeed(get().speed);
        const avatarAvailable = secuencia.length ? await avatar.play(secuencia) : false;
        updateMessage(id, {
          simple: res.texto_sencillo || undefined,
          animationUrl: res.animacion_url || undefined,
          secuencia: avatarAvailable ? secuencia : undefined,
          animation: avatarAvailable || res.animacion_url ? 'ready' : 'unavailable',
        });
      }
      announce('Mensaje listo para el alumno.');
    } catch (err) {
      updateMessage(id, { animation: 'error' });
      announce('No se pudieron generar las señas.');
      if (err instanceof TypeError) set({ online: false }); // fallo de red
    }
  }

  function retryAnimation() {
    const last = getLastTeacherMessage(get());
    if (!last) return;
    updateMessage(last.id, { animation: 'generating' });
    requestAnimation(last.id, last.text);
  }

  // ---- Dictado --------------------------------------------------------------

  function toggleMic() {
    if (isListening(get())) finishDictation();
    else startDictation();
  }

  function startDictation() {
    stopDictation();
    set({ mic: 'listening', micPhase: 'speaking', draft: '', draftError: null });
    announce('Micrófono activo. Habla con frases cortas.');

    const handlers = {
      onPartial: (text, isFinal) => {
        set({ draft: text });
        clearTimeout(timers.micEnd);
        if (isFinal) {
          set({ micPhase: 'ending' });
          timers.micEnd = setTimeout(finishDictation, CONFIG.AUTO_SEND_DELAY_MS);
        } else {
          set({ micPhase: 'speaking' });
        }
      },
      onSilence: () => {
        set({ micPhase: 'ending' });
        clearTimeout(timers.micEnd);
        timers.micEnd = setTimeout(finishDictation, CONFIG.AUTO_SEND_DELAY_MS);
      },
      onEnd: () => {
        if (isListening(get())) {
          clearTimeout(timers.micEnd);
          finishDictation();
        }
      },
      onError: (reason) => {
        if (reason === 'aborted') return;
        stopDictation();
        if (reason === 'denied') {
          set({ mic: 'denied', micPhase: null });
          showToast('Permite el micrófono en el candado de la barra de direcciones.', 'warn');
          announce('Sin permiso para el micrófono.');
        } else {
          set({ mic: 'off', micPhase: null });
          const messages = {
            nospeech: 'No se escuchó nada. Intenta de nuevo.',
            network: 'El dictado necesita conexión a internet (Chrome usa el servicio de Google). Escribe el mensaje o revisa tu conexión.',
            nomic: 'No se encontró un micrófono. Revisa que esté conectado y permitido en Windows.',
            busy: 'Otra aplicación está usando el micrófono. Ciérrala e intenta de nuevo.',
            insecure: 'El navegador solo permite el micrófono en http://127.0.0.1:8000 o https.',
          };
          if (messages[reason]) showToast(messages[reason], 'warn');
          else announce('Se interrumpió el dictado.');
        }
      },
    };

    if (CONFIG.DEMO_MODE) {
      const phrase = DICTATION_SCRIPT[dictationIndex % DICTATION_SCRIPT.length];
      dictationIndex += 1;
      cancelDictation = demo.simulateDictation(phrase, handlers);
    } else if (speech.isSupported()) {
      cancelDictation = speech.startDictation(handlers);
    } else {
      set({ mic: 'off', micPhase: null });
      showToast('Tu navegador no permite dictar. Usa Chrome o escribe el mensaje.', 'warn');
    }
  }

  function stopDictation() {
    cancelDictation?.();
    cancelDictation = null;
    clearTimeout(timers.micEnd);
  }

  /** Termina el dictado y envía lo dicho (Detener o silencio). */
  function finishDictation() {
    stopDictation();
    const text = get().draft.trim();
    set({ mic: 'off', micPhase: null, draft: '' });
    if (text) sendTeacherMessage(text, 'voz');
    else announce('Micrófono apagado.');
  }

  // ------------------------------------------------------------------------
  // Alumno: texto alterno y preferencias de lectura
  // ------------------------------------------------------------------------

  function toggleWriter() {
    set((s) => ({ writerOpen: !s.writerOpen }));
  }

  /** @returns {boolean} true si se envió */
  function sendStudentText(text) {
    const content = text.trim();
    if (!content) return false;
    addMessage({ author: 'alumno', via: 'texto', text: content });
    announce('Mensaje enviado al maestro.');
    return true;
  }

  function toggleOriginal() {
    set((s) => ({ subtitleMode: s.subtitleMode === 'original' ? 'simple' : 'original' }));
  }

  function toggleLargeText() {
    set((s) => ({ largeText: !s.largeText }));
  }

  function setSpeed(value) {
    set({ speed: value });
    if (stageVideo) stageVideo.playbackRate = Number(value);
    avatar.setSpeed(value);
    announce(`Velocidad ${value}×`);
  }

  function animationReady() {
    const last = getLastTeacherMessage(get());
    return Boolean(
      last?.animation === 'ready'
      && (last.secuencia?.length || last.animationUrl),
    );
  }

  function repeatAnimation() {
    if (animationReady()) {
      const last = getLastTeacherMessage(get());
      if (last.secuencia?.length) avatar.replay();
      else {
        stageVideo.currentTime = 0;
        stageVideo.play();
      }
    } else {
      showToast('Repetir estará disponible cuando el servicio entregue la animación.');
    }
  }

  function expandAnimation() {
    if (!animationReady()) {
      showToast('Ampliar estará disponible cuando el servicio entregue la animación.');
      return;
    }
    const last = getLastTeacherMessage(get());
    if (last.secuencia?.length && stageAvatar?.requestFullscreen) stageAvatar.requestFullscreen();
    else if (stageVideo.requestFullscreen) stageVideo.requestFullscreen();
    else showToast('Ampliar estará disponible cuando el servicio entregue la animación.');
  }

  // ------------------------------------------------------------------------
  // Conversación y sistema
  // ------------------------------------------------------------------------

  async function copyConversation() {
    const label = { maestro: 'Maestro', alumno: 'Alumno', sistema: 'Aviso' };
    const text = get().messages.map((m) => `[${m.time}] ${label[m.author]}: ${m.text}`).join('\n');
    const ok = await copyText(text);
    showToast(ok ? 'Conversación copiada.' : 'No se pudo copiar.', ok ? 'ok' : 'warn');
  }

  function clearConversation() {
    const { messages } = get();
    if (!messages.length) return;
    set({ messages: [], undoSnapshot: { messages } });
    showToast('Borramos la conversación.', 'info', { undo: true, ms: 10000 });
  }

  function undo() {
    const snapshot = get().undoSnapshot;
    if (!snapshot) return;
    clearTimeout(timers.toast);
    set({ ...snapshot, undoSnapshot: null, toast: null });
    announce('Recuperamos la conversación.');
  }

  function closeToast() {
    clearTimeout(timers.toast);
    set({ toast: null });
  }

  async function reconnect() {
    const ok = CONFIG.DEMO_MODE ? true : await api.checkHealth();
    set({ online: ok });
    if (ok && get().camera === 'active') startRecognition();
    announce(ok ? 'Conexión recuperada.' : 'Seguimos sin conexión.');
  }

  function reset() {
    stopSigning();
    stopDictation();
    cancelPermission?.();
    camera.stop();
    clearTimeout(timers.toast);
    signIndex = 0;
    dictationIndex = 0;
    set({ ...initialState, announcement: { id: get().announcement.id + 1, text: 'Demostración reiniciada.' } });
  }

  // ------------------------------------------------------------------------
  // Controles de demostración
  // ------------------------------------------------------------------------

  function simulateSigns(fail) {
    if (get().camera !== 'active') set({ camera: 'active' });
    stopSigning();
    const phrase = SIGN_SCRIPT[signIndex % SIGN_SCRIPT.length];
    if (!fail) signIndex += 1;
    cancelSigns = demo.simulateSigns(phrase, { fail }, handleRecognition);
  }

  function ensureCameraActive() {
    if (get().camera !== 'active' && get().camera !== 'paused') set({ camera: 'active' });
  }

  const demoActions = {
    signs: () => simulateSigns(false),
    'signs-fail': () => simulateSigns(true),
    dictation: startDictation,
    'mic-denied': () => { stopDictation(); set({ mic: 'denied', micPhase: null }); },
    'camera-denied': () => { stopSigning(); set({ camera: 'denied', signPhase: null, interim: '' }); },
    'camera-none': () => { stopSigning(); set({ camera: 'nocam', signPhase: null, interim: '' }); },
    'low-light': () => { ensureCameraActive(); set((s) => ({ quality: { ...s.quality, light: !s.quality.light } })); },
    framing: () => { ensureCameraActive(); set((s) => ({ quality: { ...s.quality, framing: !s.quality.framing } })); },
    offline: () => { stopDictation(); set({ online: false, mic: 'off', micPhase: null }); },
    example: () => {
      reset();
      set({
        camera: 'active',
        messages: EXAMPLE_CONVERSATION.map((m) => ({ id: uid(), simple: SIMPLE_SPANISH[m.text], ...m })),
      });
    },
    reset,
  };

  // ------------------------------------------------------------------------
  // Arranque
  // ------------------------------------------------------------------------

  async function init() {
    if (!CONFIG.DEMO_MODE) {
      set({ online: await api.checkHealth() });
    }
  }

  return {
    init,
    /** Acciones disparadas por botones con data-action="…" */
    actions: {
      'camera-on': cameraOn,
      'camera-toggle-pause': cameraTogglePause,
      'teacher-pause-camera': cameraTogglePause,
      'discard-signs': discardSigns,
      'retry-animation': retryAnimation,
      'repeat-animation': repeatAnimation,
      'expand-animation': expandAnimation,
      'toggle-original': toggleOriginal,
      'toggle-large-text': toggleLargeText,
      'toggle-writer': toggleWriter,
      'toggle-mic': toggleMic,
      'copy-conversation': copyConversation,
      'clear-conversation': clearConversation,
      undo,
      'close-toast': closeToast,
      reconnect,
    },
    setCameraView,
    setSpeed,
    setDraft,
    sendTeacherMessage: () => sendTeacherMessage(),
    sendStudentText,
    demo: demoActions,
    setFailNextAnimation: (value) => set({ failNextAnimation: value }),
    toggleDemo: (open) => set((s) => ({ demoOpen: open ?? !s.demoOpen })),
  };
}
