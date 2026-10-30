"""
Genera de forma procedural la avatar intérprete (mujer, complexión media, piel morena media,
cabello corto azul metálico): torso, cabeza, brazos y manos con su esqueleto, exportada a GLB.

Flujo de malla: formas base -> unión volumétrica (voxel remesh) -> retopología en quads
(QuadriFlow) -> subdivisión -> pesos por difusión de calor (bone heat) suavizados, máx. 4
influencias por vértice. La ropa copia los pesos de la piel y la piel cubierta se elimina, de modo
que nada atraviesa la tela al moverse.

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
from mathutils import Matrix, Quaternion, Vector

SKIN_SRGB = (0.60, 0.42, 0.33)      # Tono de piel medio (moreno)
SHIRT_SRGB = (0.075, 0.085, 0.115)  # Ropa oscura lisa, como usan los intérpretes de LSM
HAIR_SRGB = (0.12, 0.24, 0.56)      # Azul metálico profundo
NAIL_SRGB = (0.80, 0.58, 0.55)
LIP_SRGB = (0.56, 0.30, 0.30)
BROW_SRGB = (0.09, 0.11, 0.20)
METAL_SRGB = (0.86, 0.86, 0.88)

HAND_SCALE = 0.93
EYE_R = 0.0122  # ojos de tamaño natural, ligeramente estilizados                   # Mano femenina adulta (~7 % menor que la masculina media)

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

HAND_MCP = {k: tuple(c * HAND_SCALE for c in v) for k, v in HAND_MCP.items()}
HAND_PHALANX = {k: tuple(c * HAND_SCALE for c in v) for k, v in HAND_PHALANX.items()}
HAND_RADIUS = {k: tuple(c * HAND_SCALE * 0.97 for c in v) for k, v in HAND_RADIUS.items()}
THUMB_CMC = tuple(c * HAND_SCALE for c in THUMB_CMC)
THUMB_LEN = tuple(c * HAND_SCALE for c in THUMB_LEN)
THUMB_RADIUS = tuple(c * HAND_SCALE * 0.97 for c in THUMB_RADIUS)


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_material(name: str, srgb: tuple[float, float, float], roughness: float = 0.6,
                  metallic: float = 0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*[srgb_to_linear(c) for c in srgb], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
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


def build_bone_table(torso=None) -> list[dict]:
    """Devuelve la lista de huesos (nombre, padre, cabeza, cola) en coordenadas Blender."""
    bones: list[dict] = []

    def add(name, parent, head, tail):
        bones.append({"name": name, "parent": parent, "head": Vector(head), "tail": Vector(tail)})

    if torso:
        for nm, par in (("hips", None), ("spine", "hips"), ("spine1", "spine"), ("chest", "spine1"),
                        ("neck", "chest"), ("head", "neck")):
            add(nm, par, torso[nm][0], torso[nm][1])
    else:
        add("hips", None, (0, 0, 0.95), (0, 0, 1.05))
        add("spine", "hips", (0, 0, 1.05), (0, 0.005, 1.18))
        add("spine1", "spine", (0, 0.005, 1.18), (0, 0.0, 1.31))
        add("chest", "spine1", (0, 0.0, 1.31), (0, -0.005, 1.43))
        add("neck", "chest", (0, -0.005, 1.43), (0, -0.012, 1.535))
        add("head", "neck", (0, -0.012, 1.535), (0, -0.012, 1.76))

    for side in ("R", "L"):
        s = -1.0 if side == "R" else 1.0
        shoulder = Vector((0.163 * s, 0.0, 1.405))
        elbow = Vector((0.428 * s, 0.0, 1.405))
        wrist = Vector((0.672 * s, 0.0, 1.405))
        mid = elbow.lerp(wrist, 0.5)
        clav = torso[f"clavicle_head_{side}"] if torso else (0.02 * s, -0.025, 1.395)
        add(f"clavicle_{side}", "chest", clav, shoulder)
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


def set_smooth(obj):
    for poly in obj.data.polygons:
        poly.use_smooth = True


def retopo(obj, voxel, faces, smooth_iters=6, smooth_factor=0.6, subdiv=1):
    """Unión volumétrica + retopología en quads (QuadriFlow) + subdivisión Catmull-Clark."""
    apply_modifier(obj, "REMESH", mode="VOXEL", voxel_size=voxel, adaptivity=0.0)
    if smooth_iters:
        apply_modifier(obj, "SMOOTH", factor=smooth_factor, iterations=smooth_iters)
    before = len(obj.data.polygons)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.quadriflow_remesh("EXEC_DEFAULT", target_faces=faces, use_mesh_symmetry=False,
                                         use_preserve_sharp=False, use_preserve_boundary=False,
                                         smooth_normals=False, seed=7)
    except RuntimeError as exc:
        print(f"[avatar] QuadriFlow falló en {obj.name}: {exc}")
    polys = obj.data.polygons
    quads = sum(1 for p in polys if len(p.vertices) == 4)
    if len(polys) == before or quads < 0.9 * len(polys):
        print(f"[avatar] {obj.name}: sin retopología en quads, se usa decimado")
        apply_modifier(obj, "DECIMATE", ratio=min(1.0, faces / max(before, 1)))
    if subdiv:
        apply_modifier(obj, "SUBSURF", levels=subdiv, render_levels=subdiv)
    set_smooth(obj)
    print(f"[avatar] {obj.name}: {len(obj.data.polygons)} caras "
          f"({100.0 * sum(1 for p in obj.data.polygons if len(p.vertices) == 4) / len(obj.data.polygons):.0f} % quads)")
    return obj


def cut_and_relax(obj, predicate, relax=12):
    """Recorta vértices y relaja los bordes abiertos para que el corte quede limpio, sin escalones."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if predicate(v.co)], context="VERTS")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    for _ in range(3):  # quita astillas: vértices de borde que sólo tocan una cara
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.is_boundary and len(v.link_faces) <= 1], context="VERTS")
    seen, islands = set(), []
    for v0 in bm.verts:
        if v0 in seen:
            continue
        stack, isl = [v0], []
        seen.add(v0)
        while stack:
            v = stack.pop()
            isl.append(v)
            for e in v.link_edges:
                o = e.other_vert(v)
                if o not in seen:
                    seen.add(o)
                    stack.append(o)
        islands.append(isl)
    biggest = max((len(i) for i in islands), default=0)
    bmesh.ops.delete(bm, geom=[v for i in islands if len(i) < 0.02 * biggest for v in i], context="VERTS")
    for _ in range(relax):
        border = [v for v in bm.verts if v.is_boundary]
        bmesh.ops.smooth_vert(bm, verts=border, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(obj.data)
    bm.free()


def torso_parts(B, pad=0.0):
    """Envolventes del torso femenino; `pad` infla las formas para generar la blusa."""
    P = pad
    pelvis = ellipsoid_pts((0, 0.008, 0.950), (0.156 + P, 0.100 + P, 0.15 + P))
    waist = ellipsoid_pts((0, 0.010, 1.125), (0.112 + P, 0.082 + P, 0.075 + P))
    ribs = ellipsoid_pts((0, 0.0, 1.255), (0.132 + P, 0.090 + P, 0.09 + P))
    upper = ellipsoid_pts((0, 0.006, 1.365), (0.138 + P, 0.080 + P, 0.06 + P))
    traps = [ellipsoid_pts((0.068 * s, 0.014, 1.424), (0.056 + P, 0.044 + P, 0.026 + P)) for s in (-1, 1)]
    delts = [sphere_pts(B[f"upperarm_{sd}"]["head"] + Vector((0, 0, -0.01)), 0.042 + P) for sd in ("R", "L")]
    bust = [ellipsoid_pts((0.056 * s, -0.064, 1.282), (0.062 + P, 0.056 + P, 0.058 + P)) for s in (-1, 1)]
    return [hull_object("torso_low", [pelvis, waist]),
            hull_object("torso_mid", [waist, ribs]),
            hull_object("torso_up", [ribs, upper, *traps, *delts]),
            hull_object("bust", [*bust, ellipsoid_pts((0, -0.03, 1.275), (0.085 + P, 0.05 + P, 0.06 + P))])]


HEAD_C = Vector((0, -0.012, 1.614))


def build_body(B):
    """Torso, cuello y brazos como una sola piel continua."""
    parts = torso_parts(B)
    parts.append(capsule("neck", Vector((0, 0.004, 1.40)), Vector((0, -0.004, 1.575)), 0.056, 0.050))
    for side in ("R", "L"):
        sh = B[f"upperarm_{side}"]["head"]
        el = B[f"upperarm_{side}"]["tail"]
        wr = B[f"lowerarm_twist_{side}"]["tail"]
        parts.append(capsule(f"uarm_{side}", sh, el, 0.040, 0.032))
        parts.append(hull_object(f"farm_{side}", [sphere_pts(el, 0.032),
                                                   ellipsoid_pts(el.lerp(wr, 0.28), (0.036, 0.031, 0.029)),
                                                   ellipsoid_pts(wr, (0.012, 0.025, 0.017))]))
    for sx in (-1, 1):  # clavículas marcadas
        parts.append(capsule(f"clav_{sx}", Vector((0.022 * sx, -0.050, 1.405)), Vector((0.135 * sx, -0.022, 1.432)),
                             0.0075, 0.0085))
    body = join_objects(parts, "Body")
    return retopo(body, voxel=0.005, faces=12000, smooth_iters=8, smooth_factor=0.6)


def surface_point(bvh, x, z):
    hit = bvh.ray_cast(Vector((x, -0.5, z)), Vector((0, 1, 0)))
    return hit[0] if hit[0] is not None else Vector((x, HEAD_C.y - 0.08, z))


def build_head(B):
    """Cabeza femenina con rasgos suaves; ojos, párpados, cejas y labios se apoyan en la superficie."""
    from mathutils.bvhtree import BVHTree

    c = HEAD_C
    core = hull_object("head_core", [
        ellipsoid_pts(c + Vector((0, 0.010, 0.018)), (0.074, 0.093, 0.100)),
        ellipsoid_pts(c + Vector((0, -0.025, -0.042)), (0.056, 0.055, 0.049)),
        sphere_pts(c + Vector((0, -0.066, -0.080)), 0.016),
        *[ellipsoid_pts(c + Vector((0.044 * s, -0.054, -0.012)), (0.026, 0.030, 0.030)) for s in (-1, 1)],
    ])
    bvh = BVHTree.FromObject(core, bpy.context.evaluated_depsgraph_get())
    details = []

    for s in (-1, 1):
        surf = surface_point(bvh, 0.031 * s, c.z + 0.016)
        er = EYE_R
        ec = surf + Vector((0, er * 0.62, 0))
        details.append(("eye_white", hull_object(f"eye_{s}", [ellipsoid_pts(ec, (er * 1.05, er, er))])))
        details.append(("iris", hull_object(f"iris_{s}", [ellipsoid_pts(ec + Vector((0, -er * 0.9, -er * 0.03)),
                                                                          (er * 0.58, er * 0.2, er * 0.68))])))
        details.append(("pupil", hull_object(f"pupil_{s}", [ellipsoid_pts(ec + Vector((0, -er * 1.0, -er * 0.03)),
                                                                            (er * 0.26, er * 0.1, er * 0.32))])))
        details.append(("eye_white", hull_object(f"glint_{s}", [sphere_pts(ec + Vector((0.004 * s, -er * 1.02, er * 0.3)),
                                                                            er * 0.12)])))
        lid = hull_object(f"lid_{s}", [ellipsoid_pts(ec, (er * 1.12, er * 1.08, er * 1.08))])
        bm = bmesh.new()
        bm.from_mesh(lid.data)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < ec.z + er * 0.42 or v.co.y > ec.y + 0.003],
                         context="VERTS")
        bm.to_mesh(lid.data)
        bm.free()
        details.append(("skin", lid))
        details.append(("brow", hull_object(f"lash_{s}", [ellipsoid_pts(ec + Vector((0.002 * s, -er * 0.55, er * 0.78)),
                                                                          (er * 0.98, er * 0.30, er * 0.09))])))
        brow = []
        for k in range(5):
            x = (0.013 + 0.0085 * k) * s
            z = c.z + 0.036 + 0.0045 * math.sin(math.pi * (k + 0.8) / 5.6)
            p = surface_point(bvh, x, z) + Vector((0, -0.0012, 0))
            brow.append(sphere_pts(p, 0.0030 - 0.0003 * k))
        details.append(("brow", hull_object(f"brow_{s}", brow)))
    up = surface_point(bvh, 0.0, c.z - 0.041)
    lo = surface_point(bvh, 0.0, c.z - 0.049)
    details.append(("lip", hull_object("lip_up", [ellipsoid_pts(up + Vector((0, 0.0024, 0.0004)), (0.0185, 0.0058, 0.0046))])))
    details.append(("lip", hull_object("lip_lo", [ellipsoid_pts(lo + Vector((0, 0.0022, -0.0006)), (0.0168, 0.0068, 0.0058))])))

    nose_top = surface_point(bvh, 0.0, c.z + 0.018)
    nose_tip = surface_point(bvh, 0.0, c.z - 0.018)
    nose = hull_object("nose", [sphere_pts(nose_top + Vector((0, 0.003, 0)), 0.0045),
                                sphere_pts(nose_tip + Vector((0, -0.0080, 0.001)), 0.0068),
                                *[sphere_pts(nose_tip + Vector((0.0085 * s, -0.003, -0.003)), 0.0050) for s in (-1, 1)]])
    ears = [hull_object(f"ear_{s}", [ellipsoid_pts(c + Vector((0.077 * s, 0.014, 0.0)), (0.010, 0.018, 0.027))])
            for s in (-1, 1)]
    head = join_objects([core, nose, *ears], "Head")
    retopo(head, voxel=0.0022, faces=9000, smooth_iters=4, smooth_factor=0.5)
    for _, o in details:
        set_smooth(o)
    return head, details


def hair_strand(name, root, ctrl, tip, r0, r1, flat=0.38):
    """Mechón afilado: curva de Bézier cuadrática con sección elíptica que se adelgaza a la punta."""
    pts = []
    for k in range(9):
        t = k / 8.0
        p = root * (1 - t) ** 2 + ctrl * 2 * t * (1 - t) + tip * t * t
        r = r0 * (1 - t) + r1 * t
        pts.append(ellipsoid_pts(p, (r, r * flat + 0.0008, r)))
    return hull_object(name, pts)


def build_hair():
    """Bob corto despeinado en capas, raya al lado y mechones sueltos, azul metálico."""
    import random
    rnd = random.Random(11)
    j = lambda a: Vector((rnd.uniform(-a, a), rnd.uniform(-a, a), rnd.uniform(-a, a)))
    c = HEAD_C
    pieces = [hull_object("hair_cap", [ellipsoid_pts(c + Vector((0, 0.010, 0.022)), (0.084, 0.104, 0.108)),
                                       ellipsoid_pts(c + Vector((0, 0.018, 0.070)), (0.078, 0.090, 0.066))])]
    part = c + Vector((0.022, -0.020, 0.112))  # raya al lado
    # Mechones de la coronilla que caen desde la raya hacia la cara y los lados.
    for k in range(26):
        a = math.radians(-160 + 320 * k / 25) + rnd.uniform(-0.04, 0.04)
        dirv = Vector((math.sin(a), -math.cos(a) * 0.9, 0))
        root = part + dirv * 0.012
        tip = c + Vector((0.088 * math.sin(a), -0.092 * math.cos(a), 0.010)) + j(0.010)
        if abs(math.degrees(a)) < 45:  # delante: termina sobre la frente, sin tapar los ojos
            tip = c + Vector((0.050 * math.sin(a) + 0.020, -0.093, 0.046)) + j(0.006)
        ctrl = (root + tip) * 0.5 + (tip - c).normalized() * 0.03 + Vector((0, 0, 0.02))
        pieces.append(hair_strand(f"crown_{k}", root, ctrl, tip, rnd.uniform(0.010, 0.014), 0.0018))
    # Laterales y nuca hasta la mandíbula, con puntas abiertas.
    for k in range(46):
        a = math.radians(-118 + 236 * k / 45) + rnd.uniform(-0.03, 0.03)
        rx, ry = 0.092 * math.sin(a), 0.098 * math.cos(a)
        root = c + Vector((rx * 0.78, ry * 0.78 + 0.010, 0.066 + rnd.uniform(-0.012, 0.008)))
        low = -0.066 + rnd.uniform(-0.010, 0.012)
        tip = c + Vector((rx * 1.02, ry * 0.96 + 0.012, low)) + j(0.008) + Vector((rx, ry, 0)).normalized() * 0.010
        ctrl = c + Vector((rx * 1.16, ry * 1.10 + 0.010, 0.010)) + j(0.006)
        pieces.append(hair_strand(f"side_{k}", root, ctrl, tip, rnd.uniform(0.011, 0.016), 0.0016))
    # Mechones sueltos (flyaways) para romper la silueta.
    for k in range(22):
        a = math.radians(rnd.uniform(-120, 120))
        rx, ry = 0.095 * math.sin(a), 0.100 * math.cos(a)
        root = c + Vector((rx * 0.9, ry * 0.9, rnd.uniform(0.0, 0.05)))
        tip = root + Vector((rx, ry, 0)).normalized() * rnd.uniform(0.018, 0.03) + Vector((0, 0, rnd.uniform(-0.04, -0.01)))
        pieces.append(hair_strand(f"fly_{k}", root, (root + tip) * 0.5 + j(0.006), tip, 0.004, 0.0009))
    hair = join_objects(pieces, "Hair")
    retopo(hair, voxel=0.0011, faces=34000, smooth_iters=3, smooth_factor=0.4, subdiv=0)

    def cut(co):
        p = co - c
        face = p.y < -0.025 and p.z < 0.030 and abs(p.x) < 0.060
        sideburn = p.y < -0.045 and p.z < -0.010
        return face or sideburn or p.z < -0.088

    cut_and_relax(hair, cut, relax=6)
    apply_modifier(hair, "SUBSURF", levels=1, render_levels=1)
    apply_modifier(hair, "SMOOTH", factor=0.5, iterations=2)
    set_smooth(hair)
    return hair


def build_earrings():
    objs = []
    for s in (-1, 1):
        bpy.ops.mesh.primitive_torus_add(major_radius=0.0075, minor_radius=0.0012, major_segments=32,
                                         minor_segments=8, location=HEAD_C + Vector((0.081 * s, 0.010, -0.040)),
                                         rotation=(0.0, math.pi / 2, 0.0))
        o = bpy.context.active_object
        o.name = f"earring_{s}"
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        set_smooth(o)
        objs.append(o)
    return join_objects(objs, "Earrings")


def build_shirt(B, body):
    """Top negro de tirantes delgados, escote en U y largo a la cintura, con grosor de tela."""
    pad = 0.006
    shirt = join_objects(torso_parts(B, pad), "Shirt")
    retopo(shirt, voxel=0.0045, faces=7000, smooth_iters=6, smooth_factor=0.6, subdiv=0)

    def cut(p):
        neckline = 1.335 - (0.03 * max(0.0, 1 - abs(p.x) / 0.07) if p.y < 0 else 0.0)
        armhole = abs(p.x) > 0.122 and p.z > 1.245
        return p.z < 1.035 or p.z > neckline or armhole

    cut_and_relax(shirt, cut, relax=24)
    apply_modifier(shirt, "SUBSURF", levels=1, render_levels=1)
    apply_modifier(shirt, "SMOOTH", factor=0.4, iterations=2)
    apply_modifier(shirt, "SOLIDIFY", thickness=0.0022, offset=1.0, use_rim=True, thickness_clamp=1.0)
    # Tirantes delgados: recorren el hombro pegados a la superficie de la piel.
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    straps = []
    for sx in (-1, 1):
        path = []
        for k in range(17):
            t = math.pi * k / 16
            origin = Vector((0.086 * sx, -0.30 * math.cos(t), 1.34 + 0.30 * math.sin(t)))
            target = Vector((0.086 * sx, 0.0, 1.34))
            loc, nrm, _, _ = bvh.ray_cast(origin, (target - origin).normalized())
            if loc is not None:
                path.append(loc + nrm * 0.0045)
        straps += [capsule(f"strap_{sx}_{k}", a, b, 0.0032, 0.0032) for k, (a, b) in enumerate(zip(path, path[1:]))]
    shirt = join_objects([shirt, *straps], "Shirt")
    set_smooth(shirt)
    return shirt


def build_pants(B):
    """Pantalón deportivo negro de tiro medio con pretina."""
    pad = 0.006
    parts = torso_parts(B, pad)
    for extra in parts[1:]:
        bpy.data.objects.remove(extra, do_unlink=True)
    parts = parts[:1]
    parts.append(hull_object("band", [ellipsoid_pts((0, 0.008, 0.965), (0.156 + pad, 0.100 + pad, 0.020))]))
    pants = join_objects(parts, "Pants")
    retopo(pants, voxel=0.005, faces=4000, smooth_iters=6, smooth_factor=0.6, subdiv=0)
    cut_and_relax(pants, lambda p: p.z > 0.985, relax=8)
    apply_modifier(pants, "SUBSURF", levels=1, render_levels=1)
    apply_modifier(pants, "SOLIDIFY", thickness=0.003, offset=1.0, use_rim=True, thickness_clamp=1.0)
    set_smooth(pants)
    return pants


def build_hand(B, side):
    u, v, n = hand_frame(side)
    wrist = B[f"hand_{side}"]["head"]
    hs = HAND_SCALE
    parts = []
    pts = [ellipsoid_pts(wrist + u * 0.004, (0.016, 0.025 * hs, 0.017 * hs))]
    for f in FINGERS:
        mcp = B[f"{f}_01_{side}"]["head"]
        r = HAND_RADIUS[f][0] * 1.05
        pts.append(ellipsoid_pts(mcp - u * 0.004 + n * 0.002, (r, r, r * 1.05)))
    pts.append(ellipsoid_pts(wrist + (u * 0.05 + v * -0.03 + n * 0.006) * hs, (0.018 * hs, 0.014 * hs, 0.015 * hs)))
    parts.append(hull_object(f"palm_{side}", pts))
    cmc = B[f"thumb_01_{side}"]["head"]
    t_mcp = B[f"thumb_01_{side}"]["tail"]
    parts.append(hull_object(f"thenar_{side}", [
        sphere_pts(cmc + n * 0.004, 0.015 * hs),
        sphere_pts(t_mcp.lerp(cmc, 0.35) + n * 0.003, THUMB_RADIUS[0]),
        sphere_pts(wrist + (u * 0.045 + v * 0.012 + n * 0.008) * hs, 0.011 * hs),
    ]))
    chain = [f"thumb_01_{side}", f"thumb_02_{side}", f"thumb_03_{side}"]
    for i, bn in enumerate(chain):
        parts.append(capsule(bn + "_m", B[bn]["head"], B[bn]["tail"], THUMB_RADIUS[i], THUMB_RADIUS[i + 1]))
    for f in FINGERS:
        for i in range(3):
            bn = f"{f}_0{i + 1}_{side}"
            a, b = B[bn]["head"], B[bn]["tail"]
            ra, rb = HAND_RADIUS[f][i], HAND_RADIUS[f][i + 1]
            if i == 2:
                b = b - (b - a).normalized() * rb * 0.6
            parts.append(capsule(bn + "_m", a, b, ra, rb))
    hand = join_objects(parts, f"Hand_{side}")
    return retopo(hand, voxel=0.0013, faces=5000, smooth_iters=4, smooth_factor=0.5)


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
    set_smooth(nails)
    return nails


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




def heat_weights(rig, obj, allowed):
    """Pesos por difusión de calor de Blender restringidos a `allowed`."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    for g in list(obj.vertex_groups):
        if g.name not in allowed:
            obj.vertex_groups.remove(obj.vertex_groups["_zone"])


def finalize_weights(obj, bones, allowed, smooth_repeat=2, factor=0.5):
    """Suaviza (laplaciano sobre aristas), limpia, limita a 4 influencias, rellena y normaliza."""
    me = obj.data
    nv = len(me.vertices)
    groups = [g.name for g in obj.vertex_groups]
    gi = {g.index: k for k, g in enumerate(obj.vertex_groups)}
    W = np.zeros((nv, max(len(groups), 1)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, gi[g.group]] = g.weight
    if smooth_repeat and len(me.edges):
        e = np.array([ed.vertices[:] for ed in me.edges])
        deg = np.bincount(e.ravel(), minlength=nv).astype(float)[:, None]
        for _ in range(smooth_repeat):
            acc = np.zeros_like(W)
            np.add.at(acc, e[:, 0], W[e[:, 1]])
            np.add.at(acc, e[:, 1], W[e[:, 0]])
            W = np.where(deg > 0, (1 - factor) * W + factor * acc / np.maximum(deg, 1), W)
    W[W < 0.01] = 0.0
    if W.shape[1] > 4:
        drop = np.argsort(W, axis=1)[:, :-4]
        np.put_along_axis(W, drop, 0.0, axis=1)
    empty = np.where(W.sum(1) <= 1e-6)[0]
    if len(empty):
        idx = {b["name"]: b for b in bones}
        names = [n for n in allowed if n in idx]
        co = np.array([me.vertices[i].co[:] for i in empty])
        D = np.stack([seg_dist(co, np.array(idx[n]["head"][:]), np.array(idx[n]["tail"][:])) for n in names], 1)
        for vi, k in zip(empty, np.argmin(D, axis=1)):
            if names[k] not in groups:
                obj.vertex_groups.new(name=names[k])
                groups.append(names[k])
                W = np.hstack([W, np.zeros((nv, 1))])
            W[vi, groups.index(names[k])] = 1.0
    W /= W.sum(1, keepdims=True)
    for k, nm in enumerate(groups):
        g = obj.vertex_groups[nm]
        g.remove(list(range(nv)))
        nz = np.nonzero(W[:, k])[0]
        for vi in nz:
            g.add([int(vi)], float(W[vi, k]), "REPLACE")
    print(f"[avatar] {obj.name}: pesos listos ({len(empty)} vértices rellenados)")


def copy_weights(src, dst, factor=None, k=4):
    """Copia pesos del vértice más cercano de `src` (promedio IDW); `factor(co)` mezcla con los propios."""
    from mathutils.kdtree import KDTree

    kd = KDTree(len(src.data.vertices))
    for v in src.data.vertices:
        kd.insert(v.co, v.index)
    kd.balance()
    names = {g.index: g.name for g in src.vertex_groups}
    sw = [{names[g.group]: g.weight for g in v.groups} for v in src.data.vertices]
    for nm in names.values():
        if nm not in dst.vertex_groups:
            dst.vertex_groups.new(name=nm)
    dnames = {g.index: g.name for g in dst.vertex_groups}
    for v in dst.data.vertices:
        f = 1.0 if factor is None else factor(v.co)
        if f <= 0.0:
            continue
        acc, tot = {}, 0.0
        for _, i, d in kd.find_n(v.co, k):
            w = 1.0 / (d + 1e-4)
            tot += w
            for nm, x in sw[i].items():
                acc[nm] = acc.get(nm, 0.0) + w * x
        own = {dnames[g.group]: g.weight for g in v.groups}
        for nm in set(acc) | set(own):
            val = f * acc.get(nm, 0.0) / tot + (1.0 - f) * own.get(nm, 0.0)
            grp = dst.vertex_groups[nm]
            if val > 0.003:
                grp.add([v.index], val, "REPLACE")
            elif nm in own:
                grp.remove([v.index])


def hide_covered_skin(body, cloth, max_dist=0.03, keep_rings=2):
    """Elimina la piel tapada por la ropa (conservando un margen) para que nunca la atraviese."""
    from mathutils.bvhtree import BVHTree

    bvh = BVHTree.FromObject(cloth, bpy.context.evaluated_depsgraph_get())
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    covered = {v.index for v in bm.verts
               if bvh.ray_cast(v.co + v.normal * 1e-4, v.normal, max_dist)[0] is not None}
    for _ in range(keep_rings):
        border = {v.index for v in bm.verts if v.index in covered
                  and any(e.other_vert(v).index not in covered for e in v.link_edges)}
        covered -= border
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in covered], context="VERTS")
    bm.to_mesh(body.data)
    bm.free()
    print(f"[avatar] Piel oculta bajo la ropa eliminada: {len(covered)} vértices")


# ---------------------------------------------------------------------------
# Base realista MakeHuman (CC0): malla hm08 + targets de mujer joven, reposada al rig del avatar
# ---------------------------------------------------------------------------

MH_DIR = Path(__file__).resolve().parent / "makehuman"
MH_TARGETS = (
    ("universal-female-young-averagemuscle-averageweight", 1.0),
    ("african-female-young", 0.35),       # mezcla mestiza: rasgos y tono medio
    ("caucasian-female-young", 0.35),
    ("asian-female-young", 0.30),
    ("female-young-averagemuscle-averageweight-maxcup-averagefirmness", 0.5),
    ("stomach-tone-incr", 0.4),
)
MH_FINGER = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}


def load_makehuman():
    """Lee base.obj y aplica los targets; devuelve vértices (Blender, m), caras 'body' y articulaciones."""
    verts, faces, groups, g = [], [], [], None
    for line in open(MH_DIR / "base.obj", encoding="utf-8"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            g = line.split()[1]
        elif line.startswith("f "):
            faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
            groups.append(g)
    V = np.array(verts)
    for name, w in MH_TARGETS:
        for line in open(MH_DIR / f"{name}.target", encoding="utf-8"):
            p = line.split()
            if p and not p[0].startswith("#"):
                V[int(p[0])] += w * np.array([float(x) for x in p[1:4]])
    jv = {}
    for f, gr in zip(faces, groups):
        if gr.startswith("joint-"):
            jv.setdefault(gr[6:], set()).update(f)
    J = {k: V[list(ix)].mean(0) for k, ix in jv.items()}
    ground = J["ground"][1]
    scale = 1.405 / (J["r-shoulder"][1] - ground)  # hombros a la altura del rig

    def conv(p):
        return np.array([p[0] * scale, -p[2] * scale, (p[1] - ground) * scale])

    Vb = np.stack([V[:, 0] * scale, -V[:, 2] * scale, (V[:, 1] - ground) * scale], 1)
    body = [f for f, gr in zip(faces, groups) if gr == "body"]
    eye = {s: Vb[list({i for f, gr in zip(faces, groups) if gr == f"helper-{s}-eye" for i in f})] for s in "lr"}
    return Vb, body, {k: Vector(conv(p)) for k, p in J.items()}, eye


def mh_source_bones(J):
    """Huesos equivalentes del rig en la pose A de MakeHuman (cabeza, cola)."""
    src = {"hips": ("pelvis", "spine-3"), "spine": ("spine-3", "spine-2"), "spine1": ("spine-2", "spine-1"),
           "chest": ("spine-1", "neck"), "neck": ("neck", "head"), "head": ("head", "head-2")}
    out = {k: (J[a], J[b]) for k, (a, b) in src.items()}
    for side, m in (("R", "r"), ("L", "l")):
        el, wr = J[f"{m}-elbow"], J[f"{m}-hand"]
        out[f"clavicle_{side}"] = (J[f"{m}-clavicle"], J[f"{m}-shoulder"])
        out[f"upperarm_{side}"] = (J[f"{m}-shoulder"], el)
        out[f"lowerarm_{side}"] = (el, el.lerp(wr, 0.5))
        out[f"lowerarm_twist_{side}"] = (el.lerp(wr, 0.5), wr)
        out[f"hand_{side}"] = (wr, J[f"{m}-finger-3-1"])
        for f, k in MH_FINGER.items():
            j = [J[f"{m}-finger-{k}-{i}"] for i in range(1, 5)]
            if f == "thumb":
                for i in range(3):
                    out[f"thumb_0{i + 1}_{side}"] = (j[i], j[i + 1])
            else:
                out[f"meta_{f}_{side}"] = (wr, j[0])
                for i in range(3):
                    out[f"{f}_0{i + 1}_{side}"] = (j[i], j[i + 1])
    return out


def mh_torso_overrides(J):
    """Columna, cuello y cabeza del rig tomados de la anatomía MakeHuman (los brazos quedan igual)."""
    src = mh_source_bones(J)
    o = {k: src[k] for k in ("hips", "spine", "spine1", "chest", "neck", "head")}
    for side, m in (("R", "r"), ("L", "l")):
        o[f"clavicle_head_{side}"] = J[f"{m}-clavicle"]
    return o


def retarget_to_rig(obj, src, bones):
    """Reposa la malla de la pose A de MakeHuman a la pose T del rig: cada hueso temporal se alinea
    (posición, dirección y longitud) con el hueso del rig; la mano conserva palma abajo y pulgar al frente."""
    arm = bpy.data.armatures.new("MHRig")
    tmp = bpy.data.objects.new("MHRig", arm)
    bpy.context.collection.objects.link(tmp)
    bpy.context.view_layer.objects.active = tmp
    bpy.ops.object.mode_set(mode="EDIT")
    for b in bones:
        eb = arm.edit_bones.new(b["name"])
        eb.head, eb.tail = src[b["name"]]
        eb.inherit_scale = "NONE"
    for b in bones:
        if b["parent"]:
            arm.edit_bones[b["name"]].parent = arm.edit_bones[b["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    names = {b["name"] for b in bones}
    heat_weights(tmp, obj, names)
    finalize_weights(obj, [{"name": n, "head": src[n][0], "tail": src[n][1], "parent": None} for n in names],
                     names, smooth_repeat=2)
    dst = {b["name"]: b for b in bones}
    R = {}
    for b in bones:  # orden padre -> hijo
        n = b["name"]
        h0, t0 = src[n]
        d0 = (t0 - h0).normalized()
        d1 = (dst[n]["tail"] - dst[n]["head"]).normalized()
        par = R.get(b["parent"], Quaternion())
        if n.startswith("hand_"):
            s = n[-1]
            thumb_side = (src[f"meta_index_{s}"][1] - src[f"meta_pinky_{s}"][1])
            v0 = (thumb_side - d0 * thumb_side.dot(d0)).normalized()
            m0 = Matrix((d0, v0, d0.cross(v0)))
            u, v, nrm = hand_frame(s)
            m1 = Matrix((d1, (v - d1 * v.dot(d1)).normalized(), d1.cross(v).normalized()))
            q = (m1.transposed() @ m0).to_quaternion()
        else:
            q = (par @ d0).rotation_difference(d1) @ par
        R[n] = q
        pb = tmp.pose.bones[n]
        rest = tmp.data.bones[n].matrix_local
        ratio = (dst[n]["tail"] - dst[n]["head"]).length / max((t0 - h0).length, 1e-6)
        rot = q.to_matrix().to_4x4() @ rest.to_3x3().to_4x4()
        pb.matrix = Matrix.Translation(dst[n]["head"]) @ rot @ Matrix.Diagonal((1.0, ratio, 1.0, 1.0))
        bpy.context.view_layer.update()
    mod = obj.modifiers.new("retarget", "ARMATURE")
    mod.object = tmp
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    for md in list(obj.modifiers):
        obj.modifiers.remove(md)
    obj.parent = None
    bpy.data.objects.remove(tmp, do_unlink=True)  # los grupos (mismos nombres que el rig) se conservan


def build_mh_body(B, bones, mh):
    """Cuerpo realista (cabeza, torso, brazos y manos) a partir de MakeHuman, en la pose del rig."""
    V, faces, J, _ = mh
    me = bpy.data.meshes.new("Body")
    me.from_pydata([tuple(p) for p in V], [], faces)
    obj = bpy.data.objects.new("Body", me)
    bpy.context.collection.objects.link(obj)
    cut_and_relax(obj, lambda p: False, relax=0)  # quita vértices sueltos (ayudantes)
    retarget_to_rig(obj, mh_source_bones(J), bones)
    cut_and_relax(obj, lambda p: p.z < 0.74, relax=4)
    apply_modifier(obj, "SUBSURF", levels=1, render_levels=1)
    set_smooth(obj)
    return obj


def mh_face_details(body, mh):
    """Ojos, cejas y pestañas sobre la cara MakeHuman."""
    from mathutils.bvhtree import BVHTree

    _, _, J, eye = mh
    bvh = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    details = []
    for s, m in ((-1, "r"), (1, "l")):
        pts = eye[m]
        # La cabeza no se reposa (columna del rig = columna MakeHuman), así que el ojo queda donde está.
        ec = Vector(pts.mean(0))
        er = float(np.ptp(pts[:, 0])) * 0.5 * 0.97
        details.append(("eye_white", hull_object(f"eye_{s}", [sphere_pts(ec, er)])))
        details.append(("iris", hull_object(f"iris_{s}", [ellipsoid_pts(ec + Vector((0, -er * 0.86, 0)),
                                                                          (er * 0.50, er * 0.18, er * 0.50))])))
        details.append(("pupil", hull_object(f"pupil_{s}", [ellipsoid_pts(ec + Vector((0, -er * 0.98, 0)),
                                                                            (er * 0.22, er * 0.08, er * 0.22))])))
        details.append(("eye_white", hull_object(f"glint_{s}", [sphere_pts(ec + Vector((0.18 * er * s, -er * 1.0, er * 0.25)),
                                                                            er * 0.07)])))
        brow = []
        for k in range(6):
            x = ec.x + s * er * (-0.9 + 0.55 * k)
            z = ec.z + er * (1.25 + 0.30 * math.sin(math.pi * (k + 0.6) / 6.2))
            hit = bvh.ray_cast(Vector((x, -0.5, z)), Vector((0, 1, 0)))
            if hit[0] is not None:
                brow.append(sphere_pts(hit[0] + Vector((0, -0.0006, 0)), er * (0.12 - 0.012 * k)))
        details.append(("brow", hull_object(f"brow_{s}", brow)))
    for _, o in details:
        set_smooth(o)
    return details


def paint_lips(body, mh, lip_index):
    """Asigna el material de labios a las caras de la boca de la malla MakeHuman."""
    mid = [v.co for v in body.data.vertices if abs(v.co.x) < 0.004 and v.co.z > 1.45]
    tip = min(mid, key=lambda co: co.y)  # punta de la nariz
    mz = tip.z - 0.032
    front = min(co.y for co in mid if abs(co.z - mz) < 0.015)
    for p in body.data.polygons:
        c = p.center
        if c.y < front + 0.011 and (c.x / 0.023) ** 2 + ((c.z - mz) / 0.0125) ** 2 < 1.0:
            p.material_index = lip_index


def twist_gradient(obj, bones):
    """Reparte el antebrazo entre `lowerarm` y `lowerarm_twist` a lo largo de su longitud, para que la
    pronación/supinación se distribuya como en un antebrazo real (sin efecto de "envoltura de caramelo")."""
    B = {b["name"]: b for b in bones}
    G = {g.name: g.index for g in obj.vertex_groups}
    for side in "RL":
        lo, tw = f"lowerarm_{side}", f"lowerarm_twist_{side}"
        if lo not in G or tw not in G:
            continue
        e, w = Vector(B[lo]["head"]), Vector(B[f"hand_{side}"]["head"])
        ax = w - e
        gl, gt = obj.vertex_groups[lo], obj.vertex_groups[tw]
        for v in obj.data.vertices:
            ws = {g.group: g.weight for g in v.groups}
            tot = ws.get(G[lo], 0.0) + ws.get(G[tw], 0.0)
            if tot <= 0:
                continue
            t = min(1.0, max(0.0, (v.co - e).dot(ax) / ax.length_squared))
            u = min(1.0, max(0.0, (t - 0.1) / 0.85))
            k = u * u * (3 - 2 * u)
            gl.add([v.index], tot * (1 - k), "REPLACE")
            gt.add([v.index], tot * k, "REPLACE")


def strap_objects(body):
    """Tirantes delgados que recorren el hombro pegados a la piel."""
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    straps = []
    for sx in (-1, 1):
        path = []
        for k in range(17):
            t = math.pi * k / 16
            origin = Vector((0.086 * sx, -0.30 * math.cos(t), 1.34 + 0.30 * math.sin(t)))
            target = Vector((0.086 * sx, 0.0, 1.34))
            loc, nrm, _, _ = bvh.ray_cast(origin, (target - origin).normalized())
            if loc is not None:
                path.append(loc + nrm * 0.0045)
        straps += [capsule(f"strap_{sx}_{k}", a, b, 0.0032, 0.0032) for k, (a, b) in enumerate(zip(path, path[1:]))]
    return straps


def mh_cloth(body, name, keep, offset, thickness, straps=False, smooth=10, zone=None):
    """Prenda ajustada: copia de la piel (con sus pesos) recortada, separada `offset` y con grosor."""
    obj = body.copy()
    obj.data = body.data.copy()
    obj.name = obj.data.name = name
    bpy.context.collection.objects.link(obj)
    for md in list(obj.modifiers):
        obj.modifiers.remove(md)
    obj.data.materials.clear()
    for p in obj.data.polygons:
        p.material_index = 0
    cut_and_relax(obj, lambda p: not keep(p), relax=16)
    apply_modifier(obj, "SMOOTH", factor=0.5, iterations=smooth)  # la tela no copia relieves finos de la piel
    if zone:  # zona donde la prenda cae holgada (sin marcar el busto)
        g = obj.vertex_groups.new(name="_zone")
        for v in obj.data.vertices:
            w = zone(v.co)
            if w > 0:
                g.add([v.index], w, "REPLACE")
        mod = obj.modifiers.new("zone", "SMOOTH")
        mod.factor, mod.iterations, mod.vertex_group = 0.8, 60, "_zone"
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.vertex_groups.remove(obj.vertex_groups["_zone"])
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * offset
    bm.to_mesh(obj.data)
    bm.free()
    apply_modifier(obj, "SMOOTH", factor=0.3, iterations=2)
    apply_modifier(obj, "SOLIDIFY", thickness=thickness, offset=1.0, use_rim=True, thickness_clamp=1.0)
    if straps:
        st = strap_objects(body)
        bones_in = [g.name for g in obj.vertex_groups]
        for o in st:
            for nm in bones_in:
                o.vertex_groups.new(name=nm)
        merged = join_objects(st, "Straps")
        copy_weights(obj, merged)
        obj = join_objects([obj, merged], name)
    set_smooth(obj)
    return obj


def fit_to_head(obj, mh_head):
    """Escala y coloca piezas diseñadas para la cabeza procedural (cabello, aretes) sobre la cabeza MakeHuman."""
    k, c = mh_head
    for v in obj.data.vertices:
        v.co = c + (v.co - HEAD_C) * k


def mh_head_frame(body, J):
    co = np.array([v.co[:] for v in body.data.vertices])
    eye_z = J["l-eye"].z
    sk = co[(co[:, 2] > eye_z) & (np.abs(co[:, 0]) < 0.12)]
    half_w = float(np.abs(sk[:, 0]).max())
    yf, yb, zt = float(sk[:, 1].min()), float(sk[:, 1].max()), float(sk[:, 2].max())
    k = 0.5 * (half_w / 0.074 + (yb - yf) / (2 * 0.093))
    c = Vector((0.0, (yf + yb) / 2 - 0.010 * k, zt - 0.118 * k))
    print(f"[avatar] Cabeza MakeHuman: escala {k:.3f}, centro {tuple(round(x, 3) for x in c)}")
    return k, c


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def to_three(v: Vector) -> list[float]:
    return [round(v.x, 6), round(v.z, 6), round(-v.y, 6)]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="avatar/web/assets/avatar.glb")
    ap.add_argument("--skeleton", default="avatar/data/rig_skeleton.json")
    ap.add_argument("--base", choices=("makehuman", "procedural"), default="makehuman",
                    help="makehuman: malla realista CC0 reposada al rig; procedural: formas generadas por script")
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mh = load_makehuman() if args.base == "makehuman" else None
    bones = build_bone_table(mh_torso_overrides(mh[2]) if mh else None)
    B = {b["name"]: b for b in bones}

    mats = {
        "skin": make_material("Skin", SKIN_SRGB, 0.5),
        "shirt": make_material("Shirt", SHIRT_SRGB, 0.88),
        "hair": make_material("Hair", HAIR_SRGB, 0.42, metallic=0.35),
        "brow": make_material("Brow", BROW_SRGB, 0.6),
        "nail": make_material("Nail", NAIL_SRGB, 0.25),
        "eye_white": make_material("EyeWhite", (0.93, 0.92, 0.9), 0.15),
        "iris": make_material("Iris", (0.30, 0.17, 0.08), 0.25),  # café
        "pupil": make_material("Pupil", (0.01, 0.01, 0.01), 0.1),
        "lip": make_material("Lip", LIP_SRGB, 0.4),
        "metal": make_material("Metal", METAL_SRGB, 0.18, metallic=1.0),
    }

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
    if mh:
        all_bones = {b["name"] for b in bones}
        body = build_mh_body(B, bones, mh)
        body.data.materials.append(mats["skin"])
        body.data.materials.append(mats["lip"])
        paint_lips(body, mh, 1)
        # Pesos de calor calculados en la pose A (brazos separados del torso) y heredados por la subdivisión.
        finalize_weights(body, bones, all_bones, smooth_repeat=2)
        twist_gradient(body, bones)
        meshes.append(body)

        wrist_x = abs(Vector(next(b for b in bones if b["name"] == "hand_R")["head"]).x) - 0.012

        def sweater(p):
            ax = abs(p.x)
            if ax > 0.17:  # mangas largas hasta la muñeca
                return ax < wrist_x
            neck = 1.448 + 0.9 * max(0.0, ax - 0.05) - (0.012 if p.y < 0 else 0.0)  # cuello redondo
            return 0.90 < p.z < neck

        def loose(p):
            return max(0.0, 1 - abs(p.x) / 0.16) * max(0.0, 1 - abs(p.z - 1.29) / 0.12)

        shirt = mh_cloth(body, "Shirt", sweater, 0.007, 0.003, smooth=18, zone=loose)
        pants = mh_cloth(body, "Pants", lambda p: p.z < 0.985 and abs(p.x) < 0.3, 0.004, 0.003)
        def under_sweater(p):  # piel totalmente cubierta: se elimina para que nunca atraviese la tela
            ax = abs(p.x)
            if ax > 0.17:
                return ax < wrist_x - 0.03
            return 0.93 < p.z < 1.43 + 0.9 * max(0.0, ax - 0.05) - 0.03

        cut_and_relax(body, under_sweater, relax=0)
        for c in (shirt, pants):
            c.data.materials.append(mats["shirt"])
            finalize_weights(c, bones, all_bones, smooth_repeat=1)
            meshes.append(c)
            hide_covered_skin(body, c)
        details = mh_face_details(body, mh)
        frame = mh_head_frame(body, mh[2])
        hair, earrings = build_hair(), build_earrings()
        fit_to_head(hair, frame)
        fit_to_head(earrings, frame)
        hair.data.materials.append(mats["hair"])
        earrings.data.materials.append(mats["metal"])
        for kind, o in details:
            o.data.materials.append(mats[kind])
        for o in [*[o for _, o in details], hair, earrings]:
            assign_weights(o, bones, {"head"}, fixed="head")
            meshes.append(o)
    else:
        body = build_body(B)
        body.data.materials.append(mats["skin"])
        heat_weights(rig, body, body_bones)
        finalize_weights(body, bones, body_bones, smooth_repeat=3)
        meshes.append(body)

        shirt = build_shirt(B, body)
        shirt.data.materials.append(mats["shirt"])
        copy_weights(body, shirt)
        finalize_weights(shirt, bones, body_bones, smooth_repeat=1)
        meshes.append(shirt)

        for s in ("R", "L"):
            hand = build_hand(B, s)
            hand.data.materials.append(mats["skin"])
            allowed = {b["name"] for b in bones if b["name"].endswith("_" + s) and
                       b["name"].startswith(("hand", "thumb", "meta", *FINGERS))}
            allowed.add(f"lowerarm_twist_{s}")
            heat_weights(rig, hand, allowed)
            finalize_weights(hand, bones, allowed, smooth_repeat=1)
            wrist = B[f"hand_{s}"]["head"]
            u = hand_frame(s)[0]
            # Junto a la muñeca, la mano usa exactamente los pesos de la piel del antebrazo (sin costura).
            copy_weights(body, hand, factor=lambda co, w=wrist, u=u: float(np.clip(1.0 - ((co - w).dot(u) + 0.004) / 0.018, 0, 1)))
            finalize_weights(hand, bones, allowed | body_bones, smooth_repeat=0)
            meshes.append(hand)
            nails = build_nails(B, s)
            nails.data.materials.append(mats["nail"])
            assign_weights(nails, bones, {f"thumb_03_{s}"} | {f"{f}_03_{s}" for f in FINGERS}, power=12.0)
            meshes.append(nails)

        pants = build_pants(B)
        pants.data.materials.append(mats["shirt"])
        copy_weights(body, pants)
        finalize_weights(pants, bones, body_bones, smooth_repeat=1)
        meshes.append(pants)

        hide_covered_skin(body, shirt)
        hide_covered_skin(body, pants)

        head, details = build_head(B)
        head.data.materials.append(mats["skin"])
        rigid = [head, *[o for _, o in details], build_hair(), build_earrings()]
        for kind, o in details:
            o.data.materials.append(mats[kind])
        rigid[-2].data.materials.append(mats["hair"])
        rigid[-1].data.materials.append(mats["metal"])
        for o in rigid:
            assign_weights(o, bones, {"head"}, fixed="head")
            meshes.append(o)

    for m in meshes:
        m.parent = rig
        if not any(md.type == "ARMATURE" for md in m.modifiers):
            mod = m.modifiers.new("Armature", "ARMATURE")
            mod.object = rig

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_skins=True,
                              export_animations=False, export_yup=True, export_apply=False,
                              export_influence_nb=4, export_all_influences=False)
    skel = {
        "units": "meters",
        "coordinate_system": "three.js / glTF (Y arriba, el avatar mira hacia +Z)",
        "bones": [{"name": b["name"], "parent": b["parent"],
                   "head": to_three(b["head"]), "tail": to_three(b["tail"])} for b in bones],
    }
    sk = Path(args.skeleton)
    sk.parent.mkdir(parents=True, exist_ok=True)
    sk.write_text(json.dumps(skel, indent=1), encoding="utf-8")
    faces = sum(len(m.data.polygons) for m in meshes)
    print(f"[avatar] GLB -> {out}  ({out.stat().st_size / 1e6:.2f} MB, {faces} caras, {len(bones)} huesos)")
    print(f"[avatar] Esqueleto -> {sk}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
