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
// Perfil de mínimo jerk (velocidad y aceleración nulas en los extremos): movimiento humano suave.
const ease = t => t * t * t * (t * (6 * t - 15) + 10);
const DEG = Math.PI / 180;
// Límites anatómicos (grados): flexo-extensión y desviación radial/cubital de la muñeca,
// pronación/supinación del antebrazo y giro residual tolerado en la muñeca.
export const WRIST_LIMITS = { flex: 80, ext: 70, dev: 30, forearmTwist: 100, wristTwist: 10 };
// Volúmenes de colisión del cuerpo (glTF, metros): las manos no pueden entrar en ellos.
const BODY_VOLUMES = [
  { c: V(0, 1.64, 0.045), r: V(0.09, 0.12, 0.105) },   // cabeza
  { c: V(0, 1.49, 0.01), r: V(0.06, 0.07, 0.06) },     // cuello
  { c: V(0, 1.28, 0.06), r: V(0.15, 0.15, 0.135) },     // pecho
  { c: V(0, 1.07, 0.045), r: V(0.14, 0.15, 0.105) },    // abdomen
];
const HAND_MARGIN = 0.018;
const clampAbs = (v, m) => Math.max(-m, Math.min(m, v));
// Límite suave: lineal en el rango útil y se frena progresivamente cerca del tope (sin golpes).
function softLimit(v, lo, hi) {
  const knee = 0.8;
  if (v > hi * knee) return hi * knee + hi * (1 - knee) * Math.tanh((v - hi * knee) / (hi * (1 - knee)));
  if (v < -lo * knee) return -lo * knee - lo * (1 - knee) * Math.tanh((-v - lo * knee) / (lo * (1 - knee)));
  return v;
}
function axisAngle(q) {
  const w = THREE.MathUtils.clamp(q.w, -1, 1);
  const ang = 2 * Math.acos(Math.abs(w));
  const sgn = w < 0 ? -1 : 1;
  const s = Math.sqrt(Math.max(1e-12, 1 - w * w));
  return V(q.x * sgn / s, q.y * sgn / s, q.z * sgn / s).multiplyScalar(ang);
}
function fromRotVec(r) {
  const a = r.length();
  return a < 1e-9 ? new THREE.Quaternion() : new THREE.Quaternion().setFromAxisAngle(r.clone().divideScalar(a), a);
}
// Empuja un punto fuera de los volúmenes del cuerpo; devuelve el desplazamiento aplicado.
function pushOut(p, margin) {
  const d = V(0, 0, 0);
  for (const b of BODY_VOLUMES) {
    const r = b.r.clone().addScalar(margin);
    const q = p.clone().add(d).sub(b.c).divide(r);
    const len = q.length();
    if (len < 1 && len > 1e-6) {
      const surf = b.c.clone().add(q.multiplyScalar(1 / len).multiply(r));
      d.add(surf.sub(p.clone().add(d)));
    }
  }
  return d;
}

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
        palmOffset: p('middle_01').clone().sub(p('hand')).multiplyScalar(0.55),
        tipOffset: this.rest[`middle_03_${s}`].pos.clone().sub(p('hand')).multiplyScalar(1.08),
        extraOffsets: ['thumb_03', 'index_03', 'pinky_03'].map(n => this.rest[`${n}_${s}`].pos.clone().sub(p('hand')).multiplyScalar(1.06)),
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
    for (const s of ['R', 'L']) {
      const twInv = this.rest[`lowerarm_twist_${s}`].world.clone().invert();
      const n = this.arms[s].restPalm.clone().applyQuaternion(twInv);
      n.y = 0;
      this.arms[s].palmLocal = n.normalize();
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

  // Codo "vivo": como una persona, gira el brazo alrededor del eje hombro-muñeca (swivel) para que
  // el antebrazo no tenga que torcerse más allá de lo anatómico. Se elige el swivel con menor
  // torsión requerida, penalizando alejarse del anterior para que el codo se mueva con continuidad.
  solveArmOrganic(side, wristTarget, handDelta) {
    this.prevWrist = this.prevWrist || {};
    const pw = this.prevWrist[side];
    if (this.dt && pw) {  // velocidad máxima de la muñeca (~2.5 m/s): sin saltos al esquivar el cuerpo
      const d = wristTarget.clone().sub(pw);
      const maxD = 2.5 * this.dt;
      if (d.length() > maxD) wristTarget = pw.clone().add(d.setLength(maxD));
    }
    this.prevWrist[side] = wristTarget.clone();
    this.swivel = this.swivel || {};
    const prev = this.swivel[side] ?? 0;
    let best = prev, bestCost = Infinity;
    for (let k = -6; k <= 6; k++) {
      const phi = k * 10 * DEG;
      const th = this.solveArm(side, wristTarget, handDelta, phi, false);
      const cost = Math.abs(th) + 0.35 * Math.abs(phi) + 0.6 * Math.abs(phi - prev);
      if (cost < bestCost) { bestCost = cost; best = phi; }
    }
    // Con dt (reproducción) el codo y la torsión cambian con velocidad acotada: sin chasquidos.
    if (this.dt) best = prev + clampAbs(best - prev, 160 * DEG * this.dt);
    this.swivel[side] = best;
    this.solveArm(side, wristTarget, handDelta, best, true);
  }

  solveArm(side, wristTarget, handDelta, swivel = 0, commit = true) {
    const arm = this.arms[side];
    const S = arm.upper.getWorldPosition(new THREE.Vector3());
    let toT = wristTarget.clone().sub(S);
    const d = THREE.MathUtils.clamp(toT.length(), Math.abs(arm.a - arm.b) + 1e-3, arm.a + arm.b - 1e-3);
    const dir = toT.normalize();
    const pole = mirrorVec(V(-0.55, -1.0, -0.25), side).normalize()
      .applyAxisAngle(dir, side === 'L' ? -swivel : swivel);
    // Si la muñeca queda casi en la línea del polo (mano junto a la oreja), el codo apunta hacia
    // afuera y atrás de forma continua en lugar de voltearse.
    const align = Math.abs(pole.dot(dir));
    if (align > 0.6) {
      const k = THREE.MathUtils.smoothstep(align, 0.6, 0.95);
      pole.lerp(mirrorVec(V(-1.0, 0.0, -0.45), side).normalize(), k).normalize();
    }
    let perp = pole.sub(dir.clone().multiplyScalar(pole.dot(dir))).normalize();
    // El plano del codo gira con velocidad acotada (~300°/s): el codo nunca salta de lado.
    this.prevPerp = this.prevPerp || {};
    const pp = this.prevPerp[side];
    if (this.dt && pp) {
      const ang = pp.angleTo(perp), maxA = 300 * DEG * this.dt;
      if (ang > maxA) {
        const axis = pp.clone().cross(perp);
        if (axis.lengthSq() > 1e-10) perp = pp.clone().applyAxisAngle(axis.normalize(), maxA);
        perp.sub(dir.clone().multiplyScalar(perp.dot(dir))).normalize();
      }
    }
    if (commit) this.prevPerp[side] = perp.clone();
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

    // Muñeca: la pronación/supinación va al hueso de torsión del antebrazo (limitada); la mano sólo
    // conserva flexión/desviación dentro de rangos anatómicos y un giro residual mínimo.
    const restHand = this.rest[arm.hand.name].local;
    const desired = handDelta.clone().multiply(this.rest[arm.hand.name].world);
    arm.twist.quaternion.copy(this.rest[arm.twist.name].local);
    arm.twist.updateMatrixWorld(true);
    const twistW = arm.twist.getWorldQuaternion(new THREE.Quaternion());
    const X = twistW.clone().invert().multiply(desired).multiply(restHand.clone().invert());
    const twFull = new THREE.Quaternion(0, X.y, 0, X.w).normalize();
    const swing = X.clone().multiply(twFull.clone().invert());
    let theta = 2 * Math.atan2(twFull.y, twFull.w);
    theta = Math.atan2(Math.sin(theta), Math.cos(theta));
    // Continuidad: cerca de ±180° conserva el lado del fotograma anterior (sin volteos de la mano).
    this.prevTheta = this.prevTheta || {};
    const prevT = this.prevTheta[side];
    if (prevT !== undefined) {
      const un = theta + 2 * Math.PI * Math.round((prevT - theta) / (2 * Math.PI));
      if (Math.abs(un) < Math.PI * 1.15) theta = un;
    }
    if (!commit) return theta;
    this.prevTheta[side] = theta;
    const L = WRIST_LIMITS;
    let tBone = softLimit(theta, L.forearmTwist * DEG, L.forearmTwist * DEG);
    this.prevTB = this.prevTB || {};
    if (this.dt && this.prevTB[side] !== undefined) {
      tBone = this.prevTB[side] + clampAbs(tBone - this.prevTB[side], 360 * DEG * this.dt);
    }
    this.prevTB[side] = tBone;
    const tRes = clampAbs(theta - tBone, L.wristTwist * DEG);
    const yAx = V(0, 1, 0);
    const n = arm.palmLocal;
    const fAx = yAx.clone().cross(n).normalize();
    const r = axisAngle(swing);
    const flex = softLimit(r.dot(fAx), L.ext * DEG, L.flex * DEG);
    const dev = softLimit(r.dot(n), L.dev * DEG, L.dev * DEG);
    const swingC = fromRotVec(fAx.clone().multiplyScalar(flex).add(n.clone().multiplyScalar(dev)));
    const qBone = new THREE.Quaternion().setFromAxisAngle(yAx, tBone);
    const qRes = new THREE.Quaternion().setFromAxisAngle(yAx, tRes);
    arm.twist.quaternion.multiply(qBone);
    arm.twist.updateMatrixWorld(true);
    arm.hand.quaternion.copy(qBone.clone().invert().multiply(swingC).multiply(qBone).multiply(qRes).multiply(restHand));
    this.prevHandQ = this.prevHandQ || {};
    const ph = this.prevHandQ[side];
    if (this.dt && ph) {  // la muñeca gira como máximo ~540°/s
      const ang = ph.angleTo(arm.hand.quaternion);
      const maxA = 540 * DEG * this.dt;
      if (ang > maxA) arm.hand.quaternion.copy(ph.clone().slerp(arm.hand.quaternion, maxA / ang));
    }
    this.prevHandQ[side] = arm.hand.quaternion.clone();
    arm.hand.updateMatrixWorld(true);
    return theta;
  }

  // Corrige la posición de la muñeca para que palma y yemas no atraviesen el cuerpo.
  avoidBody(side, pose) {
    const arm = this.arms[side];
    for (let k = 0; k < 2; k++) {
      for (const off of [arm.palmOffset, arm.tipOffset, ...arm.extraOffsets]) {
        const pt = pose.pos.clone().add(off.clone().applyQuaternion(pose.rot));
        pose.pos.add(pushOut(pt, off === arm.palmOffset ? HAND_MARGIN : 0.008));
      }
    }
    return pose;
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

  // Igual que apply(), pero cada articulación sigue a su objetivo con un filtro críticamente
  // amortiguado (independiente de los FPS): sin saltos ni temblores entre fotogramas clave.
  applySmoothed(poses, dt, tau = 0.075) {
    if (!this.smoothBones) {
      this.smoothBones = Object.keys(this.bones).filter(n => /_(R|L)$/.test(n));
    }
    const prev = this.prevQ;
    this.dt = Math.min(dt, 0.1);
    this.apply(poses);
    this.dt = 0;
    if (prev) {
      const a = 1 - Math.exp(-dt / tau);
      for (const n of this.smoothBones) this.bones[n].quaternion.copy(prev[n].slerp(this.bones[n].quaternion, a));
    }
    this.prevQ = Object.fromEntries(this.smoothBones.map(n => [n, this.bones[n].quaternion.clone()]));
  }

  apply(poses) {
    for (const side of ['R', 'L']) {
      const p = poses[side];
      this.solveArmOrganic(side, p.pos, p.rot);
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
  const TRANS = 0.45; // transición entre señas
  const HOLD = 0.3;   // pausa en la pose final para que cada seña se lea con claridad
  const push = (label, frames, useTrained = false) => {
    // frames: [{t, R, L}] con t relativo
    const start = t + TRANS;
    for (const f of frames) {
      clips.push({ t: start + f.t, label, R: signer.poseFor('R', f.R, useTrained), L: f.L ? signer.poseFor('L', f.L) : null,
                   movR: f.R?.mov, movL: f.L?.mov });
    }
    t = start + frames[frames.length - 1].t + HOLD;
  };
  for (const item of sequence) {
    if (item.tipo === 'sena') {
      push(item.glosa, item.keyframes);
    } else {
      for (const L of item.letras) {
        const h = { shape: L.shape, loc: 'deletreo', palma: L.palma, dedos: L.dedos };
        const dur = L.mov ? 0.65 : 0.42;
        const frames = [{ t: 0, R: h }, { t: dur, R: { ...h, mov: L.mov } }];
        const before = t;
        push(`${item.glosa} · ${L.letra}`, frames, L.usar_orientacion_entrenada);
        t = before + 0.3 + dur + 0.12; // deletreo fluido pero legible
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
    // Trayectoria en arco (hacia el frente y ligeramente arriba), como el movimiento humano real.
    if (next !== prev && !next.mov) {
      const dist = prev.p.pos.distanceTo(next.p.pos);
      pos.add(V(0, 0.25, 1).normalize().multiplyScalar(Math.sin(Math.PI * e) * Math.min(0.06, 0.22 * dist)));
    }
    if (next.mov && next !== prev) pos.add(movementOffset(next.mov, u, side));
    const rot = prev.p.rot.clone().slerp(next.p.rot, e);
    out[side] = { pos, rot, shape: prev.p.shape, shapeB: next.p.shape, shapeT: Math.min(1, e * 1.4) };
    signer.avoidBody(side, out[side]);
    if (side === 'R') out.label = (u > 0.0 && next.label) ? next.label : prev.label;
  }
  // Separa las manos si llegaran a cruzarse una dentro de la otra.
  const pR = out.R.pos.clone().add(signer.arms.R.palmOffset.clone().applyQuaternion(out.R.rot));
  const pL = out.L.pos.clone().add(signer.arms.L.palmOffset.clone().applyQuaternion(out.L.rot));
  const gap = pR.clone().sub(pL);
  const MIN_GAP = 0.045;
  if (gap.length() < MIN_GAP) {
    const push = gap.lengthSq() > 1e-8 ? gap.normalize() : V(-1, 0, 0);
    const k = (MIN_GAP - pR.distanceTo(pL)) / 2;
    out.R.pos.add(push.clone().multiplyScalar(k));
    out.L.pos.sub(push.multiplyScalar(k));
  }
  return out;
}
