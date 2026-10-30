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
from mathutils import Vector

SKIN_SRGB = (0.71, 0.52, 0.41)      # Tono de piel medio (moreno claro)
SHIRT_SRGB = (0.075, 0.085, 0.115)  # Ropa oscura lisa, como usan los intérpretes de LSM
HAIR_SRGB = (0.16, 0.36, 0.78)      # Azul metálico
NAIL_SRGB = (0.80, 0.58, 0.55)
LIP_SRGB = (0.62, 0.33, 0.33)
BROW_SRGB = (0.09, 0.11, 0.20)
METAL_SRGB = (0.86, 0.86, 0.88)

HAND_SCALE = 0.93
EYE_R = 0.0150  # ojos grandes estilo anime                   # Mano femenina adulta (~7 % menor que la masculina media)

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
        shoulder = Vector((0.163 * s, 0.0, 1.405))
        elbow = Vector((0.428 * s, 0.0, 1.405))
        wrist = Vector((0.672 * s, 0.0, 1.405))
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
        ellipsoid_pts(c + Vector((0, -0.024, -0.040)), (0.052, 0.054, 0.048)),
        sphere_pts(c + Vector((0, -0.064, -0.079)), 0.013),
        *[ellipsoid_pts(c + Vector((0.044 * s, -0.054, -0.012)), (0.026, 0.030, 0.030)) for s in (-1, 1)],
    ])
    bvh = BVHTree.FromObject(core, bpy.context.evaluated_depsgraph_get())
    details = []

    for s in (-1, 1):
        surf = surface_point(bvh, 0.032 * s, c.z + 0.012)
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
            z = c.z + 0.040 + 0.0055 * math.sin(math.pi * (k + 0.5) / 5.0)
            p = surface_point(bvh, x, z) + Vector((0, -0.0012, 0))
            brow.append(sphere_pts(p, 0.0021 - 0.00025 * k))
        details.append(("brow", hull_object(f"brow_{s}", brow)))
    up = surface_point(bvh, 0.0, c.z - 0.041)
    lo = surface_point(bvh, 0.0, c.z - 0.049)
    details.append(("lip", hull_object("lip_up", [ellipsoid_pts(up + Vector((0, 0.0026, 0.0004)), (0.0145, 0.0045, 0.0034))])))
    details.append(("lip", hull_object("lip_lo", [ellipsoid_pts(lo + Vector((0, 0.0024, 0)), (0.0125, 0.0052, 0.0042))])))

    nose_top = surface_point(bvh, 0.0, c.z + 0.018)
    nose_tip = surface_point(bvh, 0.0, c.z - 0.018)
    nose = hull_object("nose", [sphere_pts(nose_top + Vector((0, 0.003, 0)), 0.004),
                                sphere_pts(nose_tip + Vector((0, -0.0065, 0.001)), 0.0052),
                                *[sphere_pts(nose_tip + Vector((0.0062 * s, -0.002, -0.002)), 0.0036) for s in (-1, 1)]])
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
    """Corte corto en capas (bob asimétrico) con flequillo de lado en mechones, azul metálico."""
    c = HEAD_C
    pieces = [hull_object("hair_cap", [ellipsoid_pts(c + Vector((0, 0.010, 0.022)), (0.084, 0.104, 0.108)),
                                       ellipsoid_pts(c + Vector((0, 0.020, 0.074)), (0.077, 0.088, 0.064))])]
    # Flequillo barrido hacia un lado: mechones desde la coronilla que terminan sobre la frente.
    for k in range(9):
        u = k / 8.0
        x0 = -0.050 + 0.085 * u
        root = c + Vector((x0 * 0.6 - 0.012, -0.030, 0.105))
        tip = c + Vector((x0 + 0.030 + 0.012 * u, -0.094 + 0.010 * abs(u - 0.4), 0.050 + 0.018 * (1 - u) ** 2))
        ctrl = (root + tip) * 0.5 + Vector((0.006, -0.040, 0.012))
        pieces.append(hair_strand(f"fringe_{k}", root, ctrl, tip, 0.016, 0.0022))
    # Laterales hasta la mandíbula (más largo del lado del flequillo).
    for s in (-1, 1):
        for k in range(6):
            a = math.radians(-35 + 22 * k)
            yy = 0.090 * math.sin(a)
            root = c + Vector((0.070 * s, yy, 0.062))
            low = -0.072 if s > 0 else -0.058
            tip = c + Vector((0.080 * s, yy - 0.012 + 0.006 * k, low + 0.010 * (k % 2)))
            ctrl = c + Vector((0.098 * s, yy, 0.000))
            pieces.append(hair_strand(f"side_{s}_{k}", root, ctrl, tip, 0.021, 0.004))
    # Nuca en capas.
    for k in range(9):
        x = -0.064 + 0.016 * k
        root = c + Vector((x * 0.8, 0.070, 0.070))
        tip = c + Vector((x * 1.05, 0.084, -0.078 + 0.008 * (k % 2)))
        ctrl = c + Vector((x * 1.1, 0.112, 0.000))
        pieces.append(hair_strand(f"nape_{k}", root, ctrl, tip, 0.022, 0.004))
    hair = join_objects(pieces, "Hair")
    retopo(hair, voxel=0.0018, faces=16000, smooth_iters=6, smooth_factor=0.5, subdiv=0)

    def cut(co):
        p = co - c
        face = p.y < -0.025 and p.z < 0.030 and abs(p.x) < 0.060
        sideburn = p.y < -0.040 and p.z < -0.005
        return face or sideburn or p.z < -0.085

    cut_and_relax(hair, cut, relax=6)
    apply_modifier(hair, "SUBSURF", levels=1, render_levels=1)
    apply_modifier(hair, "SMOOTH", factor=0.5, iterations=3)
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


def build_shirt(B):
    """Blusa oscura entallada de manga corta y escote redondo, con grosor de tela."""
    pad = 0.008
    parts = torso_parts(B, pad)
    for side in ("R", "L"):
        sh = B[f"upperarm_{side}"]["head"]
        el = B[f"upperarm_{side}"]["tail"]
        parts.append(capsule(f"sleeve_{side}", sh, sh.lerp(el, 0.5), 0.040 + pad + 0.004, 0.036 + pad + 0.004))
    shirt = join_objects(parts, "Shirt")
    retopo(shirt, voxel=0.005, faces=6000, smooth_iters=6, smooth_factor=0.6, subdiv=0)
    sleeve_x = 0.163 + 0.265 * 0.5

    def cut(p):
        neck_hole = p.z > 1.385 and (p.x ** 2 + ((p.y + 0.008) / 1.2) ** 2) < 0.078 ** 2
        scoop = p.z > 1.345 and p.y < -0.02 and (p.x / 0.075) ** 2 + ((p.z - 1.43) / 0.085) ** 2 < 1.0
        return neck_hole or scoop or abs(p.x) > sleeve_x or p.z < 0.885

    cut_and_relax(shirt, cut)
    apply_modifier(shirt, "SUBSURF", levels=1, render_levels=1)
    apply_modifier(shirt, "SMOOTH", factor=0.4, iterations=2)
    apply_modifier(shirt, "SOLIDIFY", thickness=0.0028, offset=1.0, use_even_offset=True, use_rim=True)
    set_smooth(shirt)
    return shirt


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
            obj.vertex_groups.remove(g)


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

    mats = {
        "skin": make_material("Skin", SKIN_SRGB, 0.5),
        "shirt": make_material("Shirt", SHIRT_SRGB, 0.88),
        "hair": make_material("Hair", HAIR_SRGB, 0.3, metallic=0.75),
        "brow": make_material("Brow", BROW_SRGB, 0.6),
        "nail": make_material("Nail", NAIL_SRGB, 0.25),
        "eye_white": make_material("EyeWhite", (0.93, 0.92, 0.9), 0.15),
        "iris": make_material("Iris", (0.20, 0.30, 0.62), 0.25),
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
    body = build_body(B)
    body.data.materials.append(mats["skin"])
    heat_weights(rig, body, body_bones)
    finalize_weights(body, bones, body_bones, smooth_repeat=3)
    meshes.append(body)

    shirt = build_shirt(B)
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

    hide_covered_skin(body, shirt)

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
