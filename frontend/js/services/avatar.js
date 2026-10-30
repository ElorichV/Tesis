let viewer = null;
let initialization = null;
let speed = 1;
let available = false;
const endCallbacks = new Set();

export function init(container) {
  if (!initialization) {
    initialization = import('/avatar/js/viewer.js')
      .then(({ createAvatarViewer }) => createAvatarViewer(container, { base: '/avatar/' }))
      .then((instance) => {
        viewer = instance;
        available = true;
        viewer.setSpeed(speed);
        endCallbacks.forEach((callback) => viewer.onEnd(callback));
        return true;
      })
      .catch((error) => {
        available = false;
        console.error('No se pudo iniciar el avatar LSM.', error);
        return false;
      });
  }
  return initialization;
}

export async function play(secuencia) {
  if (!initialization || !await initialization || !available || !secuencia?.length) return false;
  viewer.setSpeed(speed);
  viewer.playSequence(secuencia);
  return true;
}

export function replay() {
  return available && viewer.replay();
}

export function setSpeed(value) {
  speed = Number(value);
  viewer?.setSpeed(speed);
}

export function isReady() {
  return available && Boolean(viewer);
}

export function onEnd(callback) {
  endCallbacks.add(callback);
  viewer?.onEnd(callback);
  return () => {
    endCallbacks.delete(callback);
  };
}
