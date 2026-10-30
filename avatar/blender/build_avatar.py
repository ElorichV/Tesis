"""
Genera de forma procedural el avatar intérprete (torso, cabeza, brazos y manos) con su
esqueleto y lo exporta a GLB para el visor web.

Uso (Blender 4.2 LTS, sin interfaz):
    blender -b -P avatar/blender/build_avatar.py -- --out avatar/web/assets/avatar.glb \
        --skeleton avatar/data/rig_skeleton.json

Convenciones (coordenadas de Blender, metros, Z arriba, el avatar mira hacia -Y):
- Lado derecho del avatar = -X (sufijo "_R"), lado izquierdo = +X (sufijo "_L").
- Pose de reposo en T con palmas hacia abajo; el visor aplica la pose neutral.
- Cada mano tiene 19 huesos (metacarpos + 3 falanges por dedo + 3 del pulgar) cuyos
  extremos coinciden con los 21 landmarks de MediaPipe Hands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

SKIN_SRGB = (0.72, 0.53, 0.40)      # Tono de piel medio (moreno claro)
SHIRT_SRGB = (0.10, 0.12, 0.16)     # Ropa oscura lisa, como usan los intérpretes de LSM
HAIR_SRGB = (0.05, 0.035, 0.03)
NAIL_SRGB = (0.82, 0.64, 0.56)
LIP_SRGB = (0.55, 0.32, 0.27)

FINGERS = ("index", "middle", "ring", "pinky")

# Geometría de mano adulta de complexión media en el marco local (u: hacia los dedos,
# v: hacia el pulgar, n: hacia la palma). Valores en metros.
HAND_MCP = {
    "index": (0.094, 0.026, 0.000),
    "middle": (0.098, 0.004, 0.000),
    "ring": (0.092, -0.016, 0.000),
    "pinky": (0.082, -0.034, 0.002),
}
HAND_PHALANX = {
    "index": (0.042, 0.025, 0.021),
    "middle": (0.047, 0.029, 0.022),
    "ring": (0.044, 0.027, 0.021),
    "pinky": (0.034, 0.020, 0.019),
}
HAND_RADIUS = {
    "index": (0.0098, 0.0088, 0.0080, 0.0072),
    "middle": (0.0102, 0.0092, 0.0083, 0.0074),
    "ring": (0.0096, 0.0086, 0.0078, 0.0070),
    "pinky": (0.0086, 0.0077, 0.0070, 0.0063),
}
THUMB_CMC = (0.026, 0.020, 0.010)
THUMB_DIR = (0.62, 0.70, 0.36)
THUMB_LEN = (0.046, 0.033, 0.028)
THUMB_RADIUS = (0.0135, 0.0112, 0.0098, 0.0086)


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_material(name: str, srgb: tuple[float, float, float], roughness: float = 0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*[srgb_to_linear(c) for c in srgb], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


# ---------------------------------------------------------------------------
# Esqueleto
# ---------------------------------------------------------------------------

def hand_frame(side: str):
    s = -1.0 if side == "R" else 1.0
    u = Vector((s, 0.0, 0.0))
    v = Vector((0.0, -1.0, 0.0))
    n = Vector((0.0, 0.0, -1.0))
    return u, v, n


def build_bone_table() -> list[dict]:
    """Devuelve la lista de huesos (nombre, padre, cabeza, cola) en coordenadas Blender."""
    bones: list[dict] = []

    def add(name, parent, head, tail):
        bones.append({"name": name, "parent": parent, "head": Vector(head), "tail": Vector(tail)})

    add("hips", None, (0, 0, 0.95), (0, 0, 1.05))
    add("spine", "hips", (0, 0, 1.05), (0, 0.005, 1.18))
    add("spine1", "spine", (0, 0.005, 1.18), (0, 0.0, 1.31))
    add("chest", "spine1", (0, 0.0, 1.31), (0, -0.005, 1.43))
    add("neck", "chest", (0, -0.005, 1.43), (0, -0.012, 1.535))
    add("head", "neck", (0, -0.012, 1.535), (0, -0.012, 1.76))

    for side in ("R", "L"):
        s = -1.0 if side == "R" else 1.0
        shoulder = Vector((0.175 * s, 0.0, 1.405))
        elbow = Vector((0.455 * s, 0.0, 1.405))
        wrist = Vector((0.715 * s, 0.0, 1.405))
        mid = elbow.lerp(wrist, 0.5)
        add(f"clavicle_{side}", "chest", (0.02 * s, -0.025, 1.395), shoulder)
        add(f"upperarm_{side}", f"clavicle_{side}", shoulder, elbow)
        add(f"lowerarm_{side}", f"upperarm_{side}", elbow, mid)
        add(f"lowerarm_twist_{side}", f"lowerarm_{side}", mid, wrist)

        u, v, n = hand_frame(side)

        def hp(a, b, c):
            return wrist + u * a + v * b + n * c

        mcp_mid = hp(*HAND_MCP["middle"])
        add(f"hand_{side}", f"lowerarm_twist_{side}", wrist, mcp_mid)

        cmc = hp(*THUMB_CMC)
        tdir = (u * THUMB_DIR[0] + v * THUMB_DIR[1] + n * THUMB_DIR[2]).normalized()
        p_mcp = cmc + tdir * THUMB_LEN[0]
        tdir2 = (tdir + u * 0.35).normalized()
        p_ip = p_mcp + tdir2 * THUMB_LEN[1]
        tdir3 = (tdir2 + u * 0.15).normalized()
        p_tip = p_ip + tdir3 * THUMB_LEN[2]
        add(f"thumb_01_{side}", f"hand_{side}", cmc, p_mcp)
        add(f"thumb_02_{side}", f"thumb_01_{side}", p_mcp, p_ip)
        add(f"thumb_03_{side}", f"thumb_02_{side}", p_ip, p_tip)

        for f in FINGERS:
            mcp = hp(*HAND_MCP[f])
            spread = {"index": 0.06, "middle": 0.0, "ring": -0.05, "pinky": -0.11}[f]
            fdir = (u + v * spread).normalized()
            l1, l2, l3 = HAND_PHALANX[f]
            pip = mcp + fdir * l1
            dip = pip + fdir * l2
            tip = dip + fdir * l3
            meta_head = wrist + v * (HAND_MCP[f][1] * 0.35) + n * 0.002
            add(f"meta_{f}_{side}", f"hand_{side}", meta_head, mcp)
            add(f"{f}_01_{side}", f"meta_{f}_{side}", mcp, pip)
            add(f"{f}_02_{side}", f"{f}_01_{side}", pip, dip)
            add(f"{f}_03_{side}", f"{f}_02_{side}", dip, tip)
    return bones


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------

_UNIT_SPHERE: list[Vector] | None = None


def unit_sphere() -> list[Vector]:
    global _UNIT_SPHERE
    if _UNIT_SPHERE is None:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
        _UNIT_SPHERE = [v.co.copy() for v in bm.verts]
        bm.free()
    return _UNIT_SPHERE


def ellipsoid_pts(center, radii, rot=None):
    c = Vector(center)
    pts = []
    for p in unit_sphere():
        q = Vector((p.x * radii[0], p.y * radii[1], p.z * radii[2]))
        if rot is not None:
            q = rot @ q
        pts.append(c + q)
    return pts


def sphere_pts(center, r):
    return ellipsoid_pts(center, (r, r, r))


def hull_object(name: str, point_sets: list[list[Vector]]):
    bm = bmesh.new()
    for pts in point_sets:
        for p in pts:
            bm.verts.new(p)
    res = bmesh.ops.convex_hull(bm, input=bm.verts[:])
    interior = [g for g in res.get("geom_interior", []) if isinstance(g, bmesh.types.BMVert)]
    unused = [g for g in res.get("geom_unused", []) if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.delete(bm, geom=list(set(interior + unused)), context="VERTS")
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    return obj


def capsule(name, a, b, ra, rb):
    return hull_object(name, [sphere_pts(a, ra), sphere_pts(b, rb)])


def join_objects(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = name
    return obj


def apply_modifier(obj, mtype, **props):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new(mtype.lower(), mtype)
    for k, v in props.items():
        setattr(mod, k, v)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def remesh_smooth(obj, voxel, smooth_iters, smooth_factor=0.5, decimate=None):
    apply_modifier(obj, "REMESH", mode="VOXEL", voxel_size=voxel, adaptivity=0.0)
    if smooth_iters:
        apply_modifier(obj, "SMOOTH", factor=smooth_factor, iterations=smooth_iters)
    if decimate:
        apply_modifier(obj, "DECIMATE", ratio=decimate)
    for poly in obj.data.polygons:
        poly.use_smooth = True


def torso_parts(B, pad=0.0):
    """Envolventes del torso; `pad` infla las formas para generar la camisa."""
    P = pad
    pelvis = ellipsoid_pts((0, 0.0, 0.98), (0.165 + P, 0.11 + P, 0.09 + P))
    waist = ellipsoid_pts((0, 0.005, 1.13), (0.148 + P, 0.097 + P, 0.07 + P))
    chest = ellipsoid_pts((0, -0.008, 1.29), (0.168 + P, 0.112 + P, 0.09 + P))
    upper_chest = ellipsoid_pts((0, 0.0, 1.38), (0.172 + P, 0.095 + P, 0.065 + P))
    traps = [ellipsoid_pts((0.085 * s, 0.012, 1.435), (0.075 + P, 0.058 + P, 0.035 + P)) for s in (-1, 1)]
    delts = [sphere_pts(B[f"upperarm_{sd}"]["head"] + Vector((0, 0, -0.012)), 0.056 + P) for sd in ("R", "L")]
    return [hull_object("torso_low", [pelvis, waist]),
            hull_object("torso_up", [waist, chest, upper_chest, *traps, *delts])]


HEAD_C = Vector((0, -0.012, 1.628))


def build_body(B):
    parts = torso_parts(B)
    parts.append(capsule("neck", Vector((0, 0.0, 1.40)), Vector((0, -0.012, 1.55)), 0.058, 0.054))
    head_c = HEAD_C
    k = 1.06
    cranium = ellipsoid_pts(head_c + Vector((0, 0.008, 0.015)), (0.078 * k, 0.098 * k, 0.106 * k))
    jaw = ellipsoid_pts(head_c + Vector((0, -0.035, -0.058)), (0.058, 0.056, 0.052))
    chin = sphere_pts(head_c + Vector((0, -0.078, -0.088)), 0.022)
    parts.append(hull_object("head_shape", [cranium, jaw, chin]))
    nose = hull_object("nose", [sphere_pts(head_c + Vector((0, -0.102, 0.012)), 0.007),
                                sphere_pts(head_c + Vector((0, -0.113, -0.018)), 0.0115)])
    parts.append(nose)
    for s in (-1, 1):
        parts.append(hull_object(f"ear_{s}", [ellipsoid_pts(head_c + Vector((0.082 * s, 0.012, 0.0)),
                                                            (0.012, 0.022, 0.032))]))
    for side in ("R", "L"):
        sh = B[f"upperarm_{side}"]["head"]
        el = B[f"upperarm_{side}"]["tail"]
        wr = B[f"lowerarm_twist_{side}"]["tail"]
        parts.append(capsule(f"uarm_{side}", sh, el, 0.049, 0.041))
        fore_mid = el.lerp(wr, 0.3)
        parts.append(hull_object(f"farm_{side}", [sphere_pts(el, 0.041),
                                                   ellipsoid_pts(fore_mid, (0.041, 0.039, 0.036)),
                                                   ellipsoid_pts(wr, (0.013, 0.029, 0.02))]))
    body = join_objects(parts, "Body")
    remesh_smooth(body, voxel=0.0055, smooth_iters=8, smooth_factor=0.6, decimate=0.45)
    return body


def build_shirt(B):
    """Playera oscura de manga corta como malla independiente (bordes limpios)."""
    pad = 0.007
    parts = torso_parts(B, pad)
    for side in ("R", "L"):
        sh = B[f"upperarm_{side}"]["head"]
        el = B[f"upperarm_{side}"]["tail"]
        parts.append(capsule(f"sleeve_{side}", sh, sh.lerp(el, 0.55), 0.049 + pad + 0.004, 0.045 + pad + 0.004))
    shirt = join_objects(parts, "Shirt")
    apply_modifier(shirt, "REMESH", mode="VOXEL", voxel_size=0.005, adaptivity=0.0)
    bm = bmesh.new()
    bm.from_mesh(shirt.data)
    cut = []
    for vert in bm.verts:
        c = vert.co
        neck_hole = c.z > 1.40 and (c.x ** 2 + ((c.y + 0.012) / 1.15) ** 2) < 0.07 ** 2
        front_v = c.z > 1.385 and abs(c.x) < 0.045 and c.y < -0.03
        sleeve_end = abs(c.x) > 0.175 + 0.28 * 0.55
        if neck_hole or front_v or sleeve_end or c.z < 0.885:
            cut.append(vert)
    bmesh.ops.delete(bm, geom=cut, context="VERTS")
    bm.to_mesh(shirt.data)
    bm.free()
    apply_modifier(shirt, "SMOOTH", factor=0.6, iterations=6)
    apply_modifier(shirt, "DECIMATE", ratio=0.5)
    for poly in shirt.data.polygons:
        poly.use_smooth = True
    return shirt


def build_hand(B, side):
    u, v, n = hand_frame(side)
    wrist = B[f"hand_{side}"]["head"]
    parts = []
    # Palma: envolvente convexa entre muñeca y nudillos
    pts = [ellipsoid_pts(wrist + u * 0.006, (0.014, 0.026, 0.017))]
    for f in FINGERS:
        mcp = B[f"{f}_01_{side}"]["head"]
        r = HAND_RADIUS[f][0] * 1.05
        pts.append(ellipsoid_pts(mcp - u * 0.004 + n * 0.002, (r, r, r * 1.05)))
    pts.append(ellipsoid_pts(wrist + u * 0.05 + v * -0.03 + n * 0.006, (0.018, 0.014, 0.016)))
    parts.append(hull_object(f"palm_{side}", pts))
    # Eminencia tenar
    cmc = B[f"thumb_01_{side}"]["head"]
    t_mcp = B[f"thumb_01_{side}"]["tail"]
    parts.append(hull_object(f"thenar_{side}", [
        sphere_pts(cmc + n * 0.004, 0.016),
        sphere_pts(t_mcp.lerp(cmc, 0.35) + n * 0.003, THUMB_RADIUS[0]),
        sphere_pts(wrist + u * 0.045 + v * 0.012 + n * 0.008, 0.012),
    ]))
    chain = [f"thumb_01_{side}", f"thumb_02_{side}", f"thumb_03_{side}"]
    for i, bn in enumerate(chain):
        a, b = B[bn]["head"], B[bn]["tail"]
        parts.append(capsule(bn + "_m", a, b, THUMB_RADIUS[i], THUMB_RADIUS[i + 1]))
    for f in FINGERS:
        for i in range(3):
            bn = f"{f}_0{i + 1}_{side}"
            a, b = B[bn]["head"], B[bn]["tail"]
            ra, rb = HAND_RADIUS[f][i], HAND_RADIUS[f][i + 1]
            if i == 2:
                b = b - (b - a).normalized() * rb * 0.6
            parts.append(capsule(bn + "_m", a, b, ra, rb))
    hand = join_objects(parts, f"Hand_{side}")
    remesh_smooth(hand, voxel=0.0016, smooth_iters=4, smooth_factor=0.5, decimate=0.35)
    return hand


def build_nails(B, side):
    u, v, n = hand_frame(side)
    objs = []
    tips = [f"thumb_03_{side}"] + [f"{f}_03_{side}" for f in FINGERS]
    for bn in tips:
        a, b = B[bn]["head"], B[bn]["tail"]
        d = (b - a).normalized()
        r = THUMB_RADIUS[3] if bn.startswith("thumb") else HAND_RADIUS[bn.split("_")[0]][3]
        back = -n if not bn.startswith("thumb") else (-n * 0.6 - v * 0.8).normalized()
        c = b - d * (r * 1.9) + back * (r * 0.78)
        side_ax = d.cross(back).normalized()
        rot = np.array([side_ax, d, back]).T
        from mathutils import Matrix
        m = Matrix(rot.tolist())
        objs.append(hull_object(bn + "_nail", [ellipsoid_pts(c, (r * 0.72, r * 1.05, r * 0.22), m)]))
    nails = join_objects(objs, f"Nails_{side}")
    for poly in nails.data.polygons:
        poly.use_smooth = True
    return nails


def build_face_details(B):
    head_c = HEAD_C
    objs = []
    eye_mats = []
    for s in (-1, 1):
        c = head_c + Vector((0.032 * s, -0.082, 0.022))
        objs.append(("eye_white", hull_object(f"eye_{s}", [sphere_pts(c, 0.0125)])))
        objs.append(("iris", hull_object(f"iris_{s}", [ellipsoid_pts(c + Vector((0, -0.0105, 0)),
                                                                       (0.0065, 0.003, 0.0065))])))
        brow = [sphere_pts(head_c + Vector(((0.018 + 0.012 * k) * s, -0.093 + 0.004 * k, 0.045 + 0.003 * (1 - k))),
                           0.0042) for k in range(3)]
        objs.append(("hair", hull_object(f"brow_{s}", brow)))
    lips = hull_object("lips", [ellipsoid_pts(head_c + Vector((0, -0.096, -0.048)), (0.022, 0.008, 0.0065))])
    objs.append(("lip", lips))
    hair = hull_object("hair", [ellipsoid_pts(head_c + Vector((0, 0.014, 0.03)), (0.083, 0.104, 0.105)),
                                ellipsoid_pts(head_c + Vector((0, -0.03, 0.075)), (0.07, 0.06, 0.05))])
    # Recorta el cabello por debajo de la frente y detrás de la cara
    bm = bmesh.new()
    bm.from_mesh(hair.data)
    cut = [vert for vert in bm.verts
           if vert.co.z < head_c.z + 0.035 and vert.co.y < head_c.y + 0.02
           or vert.co.z < head_c.z - 0.045]
    bmesh.ops.delete(bm, geom=cut, context="VERTS")
    bm.to_mesh(hair.data)
    bm.free()
    apply_modifier(hair, "SOLIDIFY", thickness=0.006)
    objs.append(("hair", hair))
    for _, o in objs:
        for poly in o.data.polygons:
            poly.use_smooth = True
    return objs


# ---------------------------------------------------------------------------
# Pesos de deformación
# ---------------------------------------------------------------------------

def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def assign_weights(obj, bones, allowed, power=6.0, fixed=None):
    names = [b["name"] for b in bones if b["name"] in allowed]
    idx = {b["name"]: b for b in bones}
    children = {nm: [b["name"] for b in bones if b["parent"] == nm] for nm in names}
    co = np.array([v.co[:] for v in obj.data.vertices])
    for nm in names:
        if nm not in obj.vertex_groups:
            obj.vertex_groups.new(name=nm)
    if fixed:
        g = obj.vertex_groups[fixed]
        g.add(list(range(len(co))), 1.0, "REPLACE")
        return
    D = np.stack([seg_dist(co, np.array(idx[n]["head"][:]), np.array(idx[n]["tail"][:])) for n in names], 1)
    nearest = np.argmin(D, axis=1)
    weights = {n: [] for n in names}
    for vi in range(len(co)):
        b = names[nearest[vi]]
        cand = {b}
        par = idx[b]["parent"]
        if par in allowed:
            cand.add(par)
        cand.update(c for c in children[b] if c in allowed)
        cidx = [names.index(c) for c in cand]
        d = D[vi, cidx] + 1e-4
        w = (d.min() / d) ** power
        w = w / w.sum()
        for c, wi in zip(cidx, w):
            if wi > 0.02:
                weights[names[c]].append((vi, float(wi)))
    for n, lst in weights.items():
        g = obj.vertex_groups[n]
        for vi, w in lst:
            g.add([vi], w, "REPLACE")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def to_three(v: Vector) -> list[float]:
    return [round(v.x, 6), round(v.z, 6), round(-v.y, 6)]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="avatar/web/assets/avatar.glb")
    ap.add_argument("--skeleton", default="avatar/data/rig_skeleton.json")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bones = build_bone_table()
    B = {b["name"]: b for b in bones}

    skin = make_material("Skin", SKIN_SRGB, 0.55)
    shirt = make_material("Shirt", SHIRT_SRGB, 0.85)
    hair_m = make_material("Hair", HAIR_SRGB, 0.7)
    nail_m = make_material("Nail", NAIL_SRGB, 0.3)
    eye_w = make_material("EyeWhite", (0.93, 0.92, 0.9), 0.2)
    iris_m = make_material("Iris", (0.16, 0.09, 0.05), 0.2)
    lip_m = make_material("Lip", LIP_SRGB, 0.5)

    arm_data = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", arm_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for b in bones:
        eb = arm_data.edit_bones.new(b["name"])
        eb.head = b["head"]
        eb.tail = b["tail"]
    for b in bones:
        eb = arm_data.edit_bones[b["name"]]
        if b["parent"]:
            eb.parent = arm_data.edit_bones[b["parent"]]
        side = b["name"][-1]
        if b["name"].startswith(("thumb", "index", "middle", "ring", "pinky", "meta", "hand")):
            eb.align_roll(hand_frame(side)[2])
    bpy.ops.object.mode_set(mode="OBJECT")

    body_bones = {"hips", "spine", "spine1", "chest", "neck", "head"}
    for s in ("R", "L"):
        body_bones |= {f"clavicle_{s}", f"upperarm_{s}", f"lowerarm_{s}", f"lowerarm_twist_{s}", f"hand_{s}"}

    meshes = []
    body = build_body(B)
    body.data.materials.append(skin)
    assign_weights(body, bones, body_bones)
    meshes.append(body)
    shirt_obj = build_shirt(B)
    shirt_obj.data.materials.append(shirt)
    assign_weights(shirt_obj, bones, body_bones)
    meshes.append(shirt_obj)

    for s in ("R", "L"):
        hand = build_hand(B, s)
        hand.data.materials.append(skin)
        allowed = {b["name"] for b in bones if b["name"].endswith("_" + s) and
                   (b["name"].startswith(("hand", "thumb", "meta", *FINGERS)))}
        allowed.add(f"lowerarm_twist_{s}")
        assign_weights(hand, bones, allowed, power=8.0)
        meshes.append(hand)
        nails = build_nails(B, s)
        nails.data.materials.append(nail_m)
        nail_allowed = {f"thumb_03_{s}"} | {f"{f}_03_{s}" for f in FINGERS}
        assign_weights(nails, bones, nail_allowed, power=12.0)
        meshes.append(nails)

    mats = {"eye_white": eye_w, "iris": iris_m, "hair": hair_m, "lip": lip_m}
    for kind, o in build_face_details(B):
        o.data.materials.append(mats[kind])
        assign_weights(o, bones, {"head"}, fixed="head")
        meshes.append(o)

    for m in meshes:
        m.parent = rig
        mod = m.modifiers.new("Armature", "ARMATURE")
        mod.object = rig

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_skins=True,
                              export_animations=False, export_yup=True, export_apply=False)

    skel = {
        "units": "meters",
        "coordinate_system": "three.js / glTF (Y arriba, el avatar mira hacia +Z)",
        "bones": [{"name": b["name"], "parent": b["parent"],
                   "head": to_three(b["head"]), "tail": to_three(b["tail"])} for b in bones],
    }
    sk = Path(args.skeleton)
    sk.parent.mkdir(parents=True, exist_ok=True)
    sk.write_text(json.dumps(skel, indent=1), encoding="utf-8")
    tri = sum(len(m.data.polygons) for m in meshes)
    print(f"[avatar] GLB -> {out}  ({out.stat().st_size / 1e6:.2f} MB, {tri} polígonos, {len(bones)} huesos)")
    print(f"[avatar] Esqueleto -> {sk}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
