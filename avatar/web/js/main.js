import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Signer, buildTimeline, sampleTimeline } from './signer.js';

const $ = id => document.getElementById(id);
const container = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x2b3038);
const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 20);
camera.position.set(0, 1.42, 1.9);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.38, 0);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xfff4e8, 0x40464f, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(1.2, 2.4, 2.0);
key.castShadow = true;
key.shadow.bias = -0.0004;
key.shadow.normalBias = 0.02;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -1; key.shadow.camera.right = 1; key.shadow.camera.top = 1; key.shadow.camera.bottom = -1;
scene.add(key);
const rim = new THREE.DirectionalLight(0x9ec1ff, 1.0);
rim.position.set(-1.5, 2.0, -1.5);
scene.add(rim);

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let signer = null, timeline = null, clock = 0, speed = 1, playing = false, chest = null;

async function translate(text) {
  const r = await fetch(`/api/translate?text=${encodeURIComponent(text)}`);
  return r.json();
}

async function play(text) {
  const res = await translate(text);
  $('glossText').textContent = res.glosa || '—';
  timeline = buildTimeline(signer, res.secuencia);
  clock = 0;
  playing = true;
}
window.avatarPlay = play;
window.avatarCamera = (x, y, z, tx, ty, tz) => { camera.position.set(x, y, z); controls.target.set(tx, ty, tz); controls.update(); renderer.render(scene, camera); };
window.avatarDebug = () => ({ ready: !!signer, playing, clock, duration: timeline?.duration });
window.avatarSeek = async (text, t) => {
  if (text !== null) { const res = await translate(text); timeline = buildTimeline(signer, res.secuencia); $('glossText').textContent = res.glosa; }
  playing = false; clock = t;
  const poses = sampleTimeline(signer, timeline, t); signer.apply(poses);
  $('currentText').textContent = poses.label || '—';
  renderer.render(scene, camera);
  return poses.label;
};

$('form').addEventListener('submit', e => { e.preventDefault(); play($('text').value); });
$('stop').addEventListener('click', () => { playing = false; timeline = null; });
$('speed').addEventListener('input', e => { speed = +e.target.value; $('speedv').textContent = speed.toFixed(1) + '×'; });

function chips(el, items, onClick) {
  el.innerHTML = '';
  for (const [label, value] of items) {
    const s = document.createElement('span');
    s.textContent = label;
    s.onclick = () => onClick(value);
    el.appendChild(s);
  }
}

async function init() {
  const [gltf, lib, lex] = await Promise.all([
    new GLTFLoader().loadAsync('assets/avatar.glb'),
    fetch('assets/handshapes.json').then(r => r.json()),
    fetch('/api/lexicon').then(r => r.json()).catch(() => []),
  ]);
  gltf.scene.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false; } });
  scene.add(gltf.scene);
  signer = new Signer(gltf.scene, lib);
  chest = signer.bones.chest;
  chips($('alphabet'), 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'.split('').map(c => [c, c]), c => play(c));
  $('lexCount').textContent = lex.length;
  chips($('lexicon'), lex.map(e => [e.termino, e.termino]), t => { $('text').value = t; play(t); });
  chips($('shapes'), Object.keys(lib.handshapes).filter(n => !n.startsWith('ASL_') && !n.startsWith('fallback_')).map(n => [n, n]), n => {
    timeline = buildTimeline(signer, [{ tipo: 'sena', glosa: n, keyframes: [{ t: 0, R: { shape: n, loc: 'deletreo', palma: 'frente', dedos: 'arriba' } }, { t: 1.2, R: { shape: n, loc: 'deletreo', palma: 'frente', dedos: 'arriba' } }] }]);
    clock = 0; playing = true; $('glossText').textContent = n;
  });
  $('status').textContent = `Avatar listo · ${Object.keys(lib.handshapes).length} configuraciones de mano`;
  const q = new URLSearchParams(location.search).get('text');
  if (q) { $('text').value = q; play(q); }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (signer) {
    if (chest) {
      const b = Math.sin(now / 1000 * 1.6) * 0.006;
      chest.quaternion.copy(signer.rest.chest.local).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(b, 0, 0)));
      chest.updateMatrixWorld(true);
    }
    let poses;
    if (timeline && playing) {
      clock += dt * speed;
      poses = sampleTimeline(signer, timeline, clock);
      $('currentText').textContent = poses.label || '—';
      if (clock > timeline.duration) playing = false;
    } else if (timeline) {
      poses = sampleTimeline(signer, timeline, Math.min(clock, timeline.duration));
    } else {
      poses = { R: signer.restPose('R'), L: signer.restPose('L') };
    }
    signer.apply(poses);
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
init().catch(err => { $('status').textContent = 'Error: ' + err.message; console.error(err); });
