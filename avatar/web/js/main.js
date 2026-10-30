import * as THREE from 'three';
import { buildTimeline, sampleTimeline } from './signer.js';
import { createAvatarViewer } from './viewer.js';

const $ = id => document.getElementById(id);
const container = $('viewport');
let viewer;

async function translate(text) {
  const response = await fetch(`/api/translate?text=${encodeURIComponent(text)}`);
  return response.json();
}

async function play(text) {
  const result = await translate(text);
  $('glossText').textContent = result.glosa || '—';
  viewer.playSequence(result.secuencia);
}

window.avatarPlay = play;

function installDebugHooks() {
  window.avatarCamera = (x, y, z, tx, ty, tz) => {
    viewer.camera.position.set(x, y, z);
    viewer.controls.target.set(tx, ty, tz);
    viewer.controls.update();
    viewer.renderer.render(viewer.scene, viewer.camera);
  };
  window.avatarDebug = () => ({
    ready: Boolean(viewer.signer),
    playing: viewer.playing,
    clock: viewer.clock,
    duration: viewer.timeline?.duration,
    clips: viewer.timeline?.clips.map(clip => [clip.t, clip.label]),
  });
  window.avatarProbe = async (text) => {
    const result = await translate(text);
    const timeline = buildTimeline(viewer.signer, result.secuencia);
    const signer = viewer.signer;
    const angle = quaternion => 2 * Math.acos(Math.min(1, Math.abs(quaternion.w))) * 180 / Math.PI;
    const output = [];
    const previous = {};
    signer.prevTheta = null;
    signer.swivel = null;
    signer.prevTB = null;
    signer.prevWrist = null;
    signer.prevHandQ = null;
    signer.prevPerp = null;
    for (let time = 0; time < timeline.duration; time += 1 / 60) {
      const poses = sampleTimeline(signer, timeline, time);
      signer.dt = 1 / 60;
      signer.apply(poses);
      signer.dt = 0;
      for (const side of ['R', 'L']) {
        const arm = signer.arms[side];
        const twist = signer.rest[arm.twist.name].local.clone().invert().multiply(arm.twist.quaternion);
        const handDelta = signer.rest[arm.hand.name].local.clone().invert().multiply(arm.hand.quaternion);
        const handWorld = arm.hand.getWorldQuaternion(new THREE.Quaternion());
        const lowerWorld = arm.lower.getWorldQuaternion(new THREE.Quaternion());
        const previousSide = previous[side];
        if (previousSide) {
          output.push({
            t: +time.toFixed(3),
            s: side,
            label: poses.label,
            twist: angle(twist),
            wrist: angle(handDelta),
            dHand: angle(previousSide.handWorld.clone().invert().multiply(handWorld)),
            dLower: angle(previousSide.lowerWorld.clone().invert().multiply(lowerWorld)),
            dTwist: angle(previousSide.twist.clone().invert().multiply(twist)),
          });
        }
        previous[side] = { handWorld, lowerWorld, twist };
      }
    }
    return output;
  };
  window.avatarSeek = async (text, time) => {
    if (text !== null) {
      const result = await translate(text);
      $('glossText').textContent = result.glosa;
      viewer.timeline = buildTimeline(viewer.signer, result.secuencia);
    }
    const label = viewer.seek(null, time);
    return label;
  };
}

$('form').addEventListener('submit', event => {
  event.preventDefault();
  play($('text').value);
});
$('stop').addEventListener('click', () => viewer.stop());
$('speed').addEventListener('input', event => {
  const speed = +event.target.value;
  viewer.setSpeed(speed);
  $('speedv').textContent = speed.toFixed(1) + '×';
});

function chips(element, items, onClick) {
  element.innerHTML = '';
  for (const [label, value] of items) {
    const chip = document.createElement('span');
    chip.textContent = label;
    chip.onclick = () => onClick(value);
    element.appendChild(chip);
  }
}

async function init() {
  viewer = await createAvatarViewer(container, { lexicon: true });
  installDebugHooks();
  viewer.onLabel(label => { $('currentText').textContent = label || '—'; });
  const lib = viewer.lib;
  const lexicon = viewer.terms;
  chips($('alphabet'), 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'.split('').map(letter => [letter, letter]), letter => play(letter));
  $('lexCount').textContent = lexicon.length;
  chips($('lexicon'), lexicon.map(entry => [entry.termino, entry.termino]), term => {
    $('text').value = term;
    play(term);
  });
  chips($('shapes'), Object.keys(lib.handshapes)
    .filter(name => !name.startsWith('ASL_') && !name.startsWith('fallback_'))
    .map(name => [name, name]), name => {
      viewer.playSequence([{
        tipo: 'sena',
        glosa: name,
        keyframes: [
          { t: 0, R: { shape: name, loc: 'deletreo', palma: 'frente', dedos: 'arriba' } },
          { t: 1.2, R: { shape: name, loc: 'deletreo', palma: 'frente', dedos: 'arriba' } },
        ],
      }]);
      $('glossText').textContent = name;
    });
  $('status').textContent = `Avatar listo · ${Object.keys(lib.handshapes).length} configuraciones de mano`;
  const query = new URLSearchParams(location.search).get('text');
  if (query) {
    $('text').value = query;
    play(query);
  }
}

init().catch(error => {
  $('status').textContent = 'Error: ' + error.message;
  console.error(error);
});
