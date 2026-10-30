// Motor de animación de señas: IK de brazos, orientación de mano y configuraciones de dedos.
import * as THREE from 'three';

const V = (x, y, z) => new THREE.Vector3(x, y, z);

// Ubicaciones (glTF, metros) para la mano derecha; la izquierda se espeja en X.
export const LOCATIONS = {
  neutral: V(-0.09, 1.22, 0.30), deletreo: V(-0.14, 1.33, 0.27), reposo: V(-0.22, 0.93, 0.08),
  frente: V(0.0, 1.70, 0.13), sien: V(-0.075, 1.68, 0.07), ojo: V(-0.035, 1.655, 0.13),
  nariz: V(0.0, 1.63, 0.15), boca: V(0.0, 1.595, 0.135), barbilla: V(0.0, 1.565, 0.12),
  mejilla: V(-0.065, 1.61, 0.10), oreja: V(-0.10, 1.63, 0.0), cabeza: V(0.0, 1.80, 0.03),
  cuello: V(0.0, 1.49, 0.09), pecho: V(-0.02, 1.32, 0.17), hombro: V(-0.15, 1.39, 0.10),
  estomago: V(0.0, 1.14, 0.18), cintura: V(-0.1, 1.05, 0.15),
};

// Direcciones con nombre para la mano derecha ("lado" = hacia la línea media).
const DIRS = {
  arriba: V(0, 1, 0), abajo: V(0, -1, 0), frente: V(0, 0, 1), adentro: V(0, 0, -1),
  lado: V(1, 0, 0), afuera: V(-1, 0, 0),
};
const POINTING_SHAPES = new Set(['1', 'LSM_D', 'LSM_X', 'LSM_G', 'LSM_L', 'LSM_I', 'LSM_J', 'pinza', 'LSM_Z']);
const FINGER_BONES = ['thumb_01', 'thumb_02', 'thumb_03', 'meta_index', 'meta_middle', 'meta_ring', 'meta_pinky',
  ...['index', 'middle', 'ring', 'pinky'].flatMap(f => [`${f}_01`, `${f}_02`, `${f}_03`])];
const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

function mirrorVec(v, side) { return side === 'L' ? V(-v.x, v.y, v.z) : v.clone(); }
function mirrorQuat(q, side) { return side === 'L' ? new THREE.Quaternion(q.x, -q.y, -q.z, q.w) : q.clone(); }

export class Signer {
  constructor(root, library) {
    this.lib = library;
    this.bones = {};
    root.traverse(o => { if (o.isBone) this.bones[o.name] = o; });
    root.updateMatrixWorld(true);
    this.rest = {};
    for (const [name, b] of Object.entries(this.bones)) {
      this.rest[name] = { local: b.quaternion.clone(), world: b.getWorldQuaternion(new THREE.Quaternion()),
                          pos: b.getWorldPosition(new THREE.Vector3()) };
    }
    this.arms = {};
    for (const s of ['R', 'L']) {
      const p = n => this.rest[`${n}_${s}`].pos;
      const handRestInv = this.rest[`hand_${s}`].world.clone().invert();
      const toLocal = v => v.clone().sub(p('hand')).applyQuaternion(handRestInv);
      this.arms[s] = {
        upper: this.bones[`upperarm_${s}`], lower: this.bones[`lowerarm_${s}`],
        twist: this.bones[`lowerarm_twist_${s}`], hand: this.bones[`hand_${s}`],
        a: p('upperarm').distanceTo(p('lowerarm')), b: p('lowerarm').distanceTo(p('hand')),
        restFinger: p('middle_01').clone().sub(p('hand')).normalize(),
        restPalm: V(0, -1, 0),
        palmContact: toLocal(p('middle_01').clone().lerp(p('hand'), 0.25).add(V(0, -0.025, 0))),
        tipContact: toLocal(this.rest[`index_03_${s}`].pos.clone().add(
          p('index_03').clone().sub(p('index_02')).normalize().multiplyScalar(0.02))),
      };
    }
    // Rotaciones de dedos por configuración, convertidas a espacio local de cada hueso.
    this.shapeQuats = {};
    for (const [name, hs] of Object.entries(library.handshapes)) {
      this.shapeQuats[name] = {};
      for (const s of ['R', 'L']) {
        const out = {};
        for (const bn of FINGER_BONES) {
          const r = hs.rotations[s][bn];
          const full = `${bn}_${s}`;
          if (!this.bones[full]) continue;
          if (!r) { out[full] = new THREE.Quaternion(); continue; }
          const Rl = new THREE.Quaternion(...r);
          const RW = this.rest[full].world;
          out[full] = RW.clone().invert().multiply(Rl).multiply(RW);
        }
        this.shapeQuats[name][s] = out;
      }
    }
    this.state = { R: this.restPose('R'), L: this.restPose('L') };
  }

  restPose(side) {
    return { pos: mirrorVec(LOCATIONS.reposo, side), rot: this.orientation(side, 'lado', 'abajo'),
             shape: 'plana', shapeB: null, shapeT: 0 };
  }

  // Rotación (delta en mundo) que lleva la mano de reposo a palma/dedos dados.
  orientation(side, palma = 'frente', dedos = 'arriba') {
    const arm = this.arms[side];
    const f = mirrorVec(DIRS[dedos] || DIRS.arriba, side);
    let p = mirrorVec(DIRS[palma] || DIRS.frente, side);
    if (Math.abs(f.dot(p)) > 0.95) p = mirrorVec(dedos === 'frente' || dedos === 'adentro' ? DIRS.abajo : DIRS.frente, side);
    p.sub(f.clone().multiplyScalar(f.dot(p))).normalize();
    const fr = arm.restFinger.clone();
    const pr = arm.restPalm.clone();
    pr.sub(fr.clone().multiplyScalar(fr.dot(pr))).normalize();
    const mt = new THREE.Matrix4().makeBasis(f, p, f.clone().cross(p));
    const mr = new THREE.Matrix4().makeBasis(fr, pr, fr.clone().cross(pr));
    return new THREE.Quaternion().setFromRotationMatrix(mt.multiply(mr.transpose()));
  }

  trainedOrientation(side, shape) {
    const hs = this.lib.handshapes[shape];
    if (!hs || !hs.orientation) return null;
    return mirrorQuat(new THREE.Quaternion(...hs.orientation), side);
  }

  // Pose objetivo de una mano a partir de un fotograma clave {shape, loc, palma, dedos, offset}.
  poseFor(side, h, useTrained = false) {
    if (!h) return this.restPose(side);
    const shape = this.lib.handshapes[h.shape] ? h.shape : 'plana';
    let rot = (useTrained && this.trainedOrientation(side, shape)) || this.orientation(side, h.palma, h.dedos);
    const base = mirrorVec(LOCATIONS[h.loc] || LOCATIONS.neutral, side);
    const off = h.offset ? mirrorVec(V(...h.offset), side) : V(0, 0, 0);
    const target = base.add(off);
    let pos;
    if (h.loc === 'deletreo') {
      pos = target;
    } else {
      const arm = this.arms[side];
      const contact = POINTING_SHAPES.has(shape) ? arm.tipContact : arm.palmContact;
      const handWorld = rot.clone().multiply(this.rest[`hand_${side}`].world);
      pos = target.sub(contact.clone().applyQuaternion(handWorld));
    }
    return { pos, rot, shape, mov: h.mov };
  }

  solveArm(side, wristTarget, handDelta) {
    const arm = this.arms[side];
    const S = arm.upper.getWorldPosition(new THREE.Vector3());
    let toT = wristTarget.clone().sub(S);
    const d = THREE.MathUtils.clamp(toT.length(), Math.abs(arm.a - arm.b) + 1e-3, arm.a + arm.b - 1e-3);
    const dir = toT.normalize();
    const pole = mirrorVec(V(-0.55, -1.0, -0.25), side).normalize();
    const perp = pole.sub(dir.clone().multiplyScalar(pole.dot(dir))).normalize();
    const cosA = (arm.a * arm.a + d * d - arm.b * arm.b) / (2 * arm.a * d);
    const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
    const elbow = S.clone().add(dir.clone().multiplyScalar(arm.a * cosA)).add(perp.clone().multiplyScalar(arm.a * sinA));
    const wrist = S.clone().add(dir.clone().multiplyScalar(d));

    const aimBone = (bone, from, to, bendRef) => {
      const parentW = bone.parent.getWorldQuaternion(new THREE.Quaternion());
      bone.quaternion.copy(this.rest[bone.name].local);
      bone.updateMatrixWorld(true);
      const curW = bone.getWorldQuaternion(new THREE.Quaternion());
      const curDir = V(0, 1, 0).applyQuaternion(curW);
      const want = to.clone().sub(from).normalize();
      let delta = new THREE.Quaternion().setFromUnitVectors(curDir, want);
      if (bendRef) {
        // Alinea el eje de flexión del codo: el antebrazo se dobla hacia `bendRef`.
        const restBend = mirrorVec(V(0, 0, 1), 'R').applyQuaternion(delta);
        const b0 = restBend.sub(want.clone().multiplyScalar(restBend.dot(want))).normalize();
        const b1 = bendRef.clone().sub(want.clone().multiplyScalar(bendRef.dot(want))).normalize();
        if (b0.lengthSq() > 1e-6 && b1.lengthSq() > 1e-6) {
          delta = new THREE.Quaternion().setFromUnitVectors(b0, b1).multiply(delta);
        }
      }
      const newW = delta.multiply(curW);
      bone.quaternion.copy(parentW.invert().multiply(newW));
      bone.updateMatrixWorld(true);
    };
    aimBone(arm.upper, S, elbow, wrist.clone().sub(elbow));
    aimBone(arm.lower, elbow, wrist, null);

    // Muñeca: reparte el giro (pronación/supinación) entre el hueso de torsión y la mano.
    const desired = handDelta.clone().multiply(this.rest[arm.hand.name].world);
    arm.twist.quaternion.copy(this.rest[arm.twist.name].local);
    arm.twist.updateMatrixWorld(true);
    const twistW = arm.twist.getWorldQuaternion(new THREE.Quaternion());
    const X = twistW.clone().invert().multiply(desired).multiply(this.rest[arm.hand.name].local.clone().invert());
    const tw = new THREE.Quaternion(0, X.y, 0, X.w).normalize();
    const half = new THREE.Quaternion().slerp(tw, 0.6);
    arm.twist.quaternion.multiply(half);
    arm.twist.updateMatrixWorld(true);
    const tW2 = arm.twist.getWorldQuaternion(new THREE.Quaternion());
    arm.hand.quaternion.copy(tW2.invert().multiply(desired));
    arm.hand.updateMatrixWorld(true);
  }

  applyFingers(side, shapeA, shapeB, t) {
    const qa = this.shapeQuats[shapeA]?.[side] || this.shapeQuats.plana[side];
    const qb = shapeB ? (this.shapeQuats[shapeB]?.[side] || qa) : qa;
    const tmp = new THREE.Quaternion();
    for (const full of Object.keys(qa)) {
      tmp.slerpQuaternions(qa[full], qb[full] || qa[full], t);
      this.bones[full].quaternion.copy(this.rest[full].local).multiply(tmp);
    }
  }

  apply(poses) {
    for (const side of ['R', 'L']) {
      const p = poses[side];
      this.solveArm(side, p.pos, p.rot);
      this.applyFingers(side, p.shape, p.shapeB, p.shapeT || 0);
    }
  }
}

function movementOffset(mov, u, side) {
  if (!mov) return V(0, 0, 0);
  const s = Math.sin(u * Math.PI);
  if (mov.tipo === 'circulo') {
    const a = u * Math.PI * 2;
    return mirrorVec(V(Math.cos(a) - 1, Math.sin(a), 0).multiplyScalar(0.035), side);
  }
  if (mov.tipo === 'oscilar') {
    const ax = mov.eje ? V(...mov.eje) : V(1, 0, 0);
    return mirrorVec(ax.multiplyScalar(0.03 * Math.sin(u * Math.PI * 6) * s), side);
  }
  if (mov.tipo === 'trazo' && mov.puntos?.length > 1) {
    const pts = mov.puntos.map(p => V(...p));
    const k = Math.min(pts.length - 2, Math.floor(u * (pts.length - 1)));
    const f = u * (pts.length - 1) - k;
    return mirrorVec(pts[k].clone().lerp(pts[k + 1], f), side);
  }
  return V(0, 0, 0);
}

// Convierte la respuesta de /api/translate en una línea de tiempo de poses.
export function buildTimeline(signer, sequence) {
  const clips = [];
  let t = 0.3;
  const TRANS = 0.35;
  const push = (label, frames, useTrained = false) => {
    // frames: [{t, R, L}] con t relativo
    const start = t + TRANS;
    for (const f of frames) {
      clips.push({ t: start + f.t, label, R: signer.poseFor('R', f.R, useTrained), L: f.L ? signer.poseFor('L', f.L) : null,
                   movR: f.R?.mov, movL: f.L?.mov });
    }
    t = start + frames[frames.length - 1].t + 0.15;
  };
  for (const item of sequence) {
    if (item.tipo === 'sena') {
      push(item.glosa, item.keyframes);
    } else {
      for (const L of item.letras) {
        const h = { shape: L.shape, loc: 'deletreo', palma: L.palma, dedos: L.dedos };
        const dur = L.mov ? 0.6 : 0.35;
        const frames = [{ t: 0, R: h }, { t: dur, R: { ...h, mov: L.mov } }];
        const before = t;
        push(`${item.glosa} · ${L.letra}`, frames, L.usar_orientacion_entrenada);
        t = before + 0.22 + dur + 0.1; // deletreo más fluido
      }
    }
  }
  clips.push({ t: t + 0.6, label: '', R: signer.restPose('R'), L: null });
  return { clips, duration: t + 0.8 };
}

export function sampleTimeline(signer, tl, time) {
  const out = {};
  for (const side of ['R', 'L']) {
    const keys = tl.clips.map(c => ({ t: c.t, p: c[side] || signer.restPose(side), mov: c[side === 'R' ? 'movR' : 'movL'], label: c.label }));
    let i = keys.findIndex(k => k.t > time);
    if (i === -1) i = keys.length;
    const prev = i > 0 ? keys[i - 1] : { t: 0, p: signer.restPose(side) };
    const next = i < keys.length ? keys[i] : prev;
    const span = Math.max(1e-3, next.t - prev.t);
    const u = i < keys.length ? THREE.MathUtils.clamp((time - prev.t) / span, 0, 1) : 1;
    const e = ease(u);
    const pos = prev.p.pos.clone().lerp(next.p.pos, e);
    if (next.mov && next !== prev) pos.add(movementOffset(next.mov, u, side));
    const rot = prev.p.rot.clone().slerp(next.p.rot, e);
    out[side] = { pos, rot, shape: prev.p.shape, shapeB: next.p.shape, shapeT: Math.min(1, e * 1.4) };
    if (side === 'R') out.label = (u > 0.0 && next.label) ? next.label : prev.label;
  }
  return out;
}
