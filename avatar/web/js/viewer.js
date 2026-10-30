import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Signer, buildTimeline, sampleTimeline } from './signer.js';

const MATERIAL_LOOK = {
  Skin: { roughness: 0.52, sheen: 0.35, sheenRoughness: 0.55, sheenColor: new THREE.Color(0xffb59a), clearcoat: 0.04, clearcoatRoughness: 0.6 },
  Hair: { color: new THREE.Color(0x1f3f8f), metalness: 0.3, roughness: 0.45, clearcoat: 0.2, clearcoatRoughness: 0.45, sheen: 0.5, sheenColor: new THREE.Color(0x6f8fd8), envMapIntensity: 0.6 },
  Lip: { roughness: 0.36, clearcoat: 0.35, clearcoatRoughness: 0.3, sheen: 0.2, sheenColor: new THREE.Color(0xff9a9a) },
  EyeWhite: { roughness: 0.12, clearcoat: 1.0, clearcoatRoughness: 0.02 },
  Iris: { roughness: 0.2, clearcoat: 1.0, clearcoatRoughness: 0.02 },
  Pupil: { roughness: 0.1, clearcoat: 1.0, clearcoatRoughness: 0.02 },
  Nail: { roughness: 0.22, clearcoat: 0.8, clearcoatRoughness: 0.1 },
  Shirt: { roughness: 0.9, sheen: 0.6, sheenRoughness: 0.8, sheenColor: new THREE.Color(0x46506a) },
  Metal: { metalness: 1.0, roughness: 0.15 },
  Brow: { roughness: 0.65 },
};

export async function createAvatarViewer(container, { base = '', lexicon = false } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x23272f);
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.55;
  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 20);
  camera.position.set(0, 1.42, 1.9);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.38, 0);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xfff4e8, 0x40464f, 0.55));
  const key = new THREE.DirectionalLight(0xfff1e6, 2.0);
  key.position.set(1.2, 2.4, 2.0);
  key.castShadow = true;
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.02;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -1;
  key.shadow.camera.right = 1;
  key.shadow.camera.top = 1;
  key.shadow.camera.bottom = -1;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xdfe8ff, 0.6);
  fill.position.set(-1.6, 1.6, 1.4);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0x9ec1ff, 1.3);
  rim.position.set(-1.5, 2.0, -1.5);
  scene.add(rim);

  function resize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  const [gltf, lib, terms] = await Promise.all([
    new GLTFLoader().loadAsync(`${base}assets/avatar.glb`),
    fetch(`${base}assets/handshapes.json`).then((response) => {
      if (!response.ok) throw new Error(`No se pudo cargar handshapes.json (${response.status}).`);
      return response.json();
    }),
    lexicon
      ? fetch(`${base}api/lexicon`).then((response) => response.ok ? response.json() : [])
      : Promise.resolve([]),
  ]);
  gltf.scene.traverse((object) => {
    if (!object.isMesh) return;
    object.castShadow = true;
    object.receiveShadow = true;
    object.frustumCulled = false;
    const look = MATERIAL_LOOK[object.material.name];
    if (look) {
      const material = new THREE.MeshPhysicalMaterial({
        color: object.material.color,
        side: object.material.name === 'Shirt' ? THREE.DoubleSide : THREE.FrontSide,
        ...look,
      });
      material.name = object.material.name;
      object.material = material;
    }
  });
  scene.add(gltf.scene);

  const signer = new Signer(gltf.scene, lib);
  const chest = signer.bones.chest;
  let timeline = null;
  let clock = 0;
  let speed = 1;
  let playing = false;
  let last = performance.now();
  let wasPlaying = false;
  const labelListeners = new Set();
  const endListeners = new Set();

  function playSequence(sequence) {
    timeline = buildTimeline(signer, sequence);
    clock = 0;
    playing = true;
    wasPlaying = true;
    return timeline;
  }

  function stop() {
    playing = false;
    timeline = null;
  }

  function replay() {
    if (!timeline) return false;
    clock = 0;
    playing = true;
    wasPlaying = true;
    return true;
  }

  function setSpeed(value) {
    speed = Number(value);
  }

  function seek(nextTimeline, time) {
    if (nextTimeline) timeline = nextTimeline;
    if (!timeline) return null;
    playing = false;
    wasPlaying = false;
    clock = time;
    const poses = sampleTimeline(signer, timeline, time);
    signer.prevQ = null;
    signer.prevTheta = null;
    signer.swivel = null;
    signer.prevTB = null;
    signer.prevWrist = null;
    signer.prevHandQ = null;
    signer.prevPerp = null;
    signer.apply(poses);
    labelListeners.forEach((callback) => callback(poses.label));
    renderer.render(scene, camera);
    return poses.label;
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (chest) {
      const breath = Math.sin(now / 1000 * 1.6) * 0.006;
      chest.quaternion.copy(signer.rest.chest.local).multiply(
        new THREE.Quaternion().setFromEuler(new THREE.Euler(breath, 0, 0)),
      );
      chest.updateMatrixWorld(true);
    }
    let poses;
    if (timeline && playing) {
      clock += dt * speed;
      poses = sampleTimeline(signer, timeline, clock);
      labelListeners.forEach((callback) => callback(poses.label));
      if (clock > timeline.duration) {
        playing = false;
        if (wasPlaying) endListeners.forEach((callback) => callback());
        wasPlaying = false;
      }
    } else if (timeline) {
      poses = sampleTimeline(signer, timeline, Math.min(clock, timeline.duration));
    } else {
      poses = { R: signer.restPose('R'), L: signer.restPose('L') };
    }
    signer.applySmoothed(poses, dt);
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return {
    signer,
    scene,
    camera,
    controls,
    renderer,
    lib,
    terms,
    playSequence,
    stop,
    replay,
    setSpeed,
    seek,
    get playing() { return playing; },
    get timeline() { return timeline; },
    set timeline(value) { timeline = value; },
    get clock() { return clock; },
    set clock(value) { clock = value; },
    get speed() { return speed; },
    onLabel(callback) { labelListeners.add(callback); return () => labelListeners.delete(callback); },
    onEnd(callback) { endListeners.add(callback); return () => endListeners.delete(callback); },
  };
}
