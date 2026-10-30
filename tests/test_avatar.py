"""Pruebas deterministas del avatar LSM (cinemática, ajuste, léxico y glosa)."""

import json

import numpy as np
import pytest

from avatar.handshapes import AUTHORED, build_library, resolve_params
from avatar.kinematics import JOINT_LIMITS, PARAM_NAMES, HandModel, normalize_points, rotvec_matrix
from avatar.lexicon import Lexicon, parse_description
from avatar.train import HandshapeFitter, average_quaternions


@pytest.fixture(scope="module")
def lexicon():
    return Lexicon.load()


def test_hand_model_rest_and_flexion():
    h = HandModel("R")
    rest = h.forward({})
    assert rest.shape == (21, 3)
    assert np.allclose(rest[0], 0)
    flexed = h.forward({"index_mcp": 90, "index_pip": 90})
    assert np.linalg.norm(flexed[8] - flexed[0]) < np.linalg.norm(rest[8] - rest[0])
    assert np.allclose(flexed[12], rest[12])


def test_left_hand_mirrors_right():
    r = HandModel("R").forward({"index_mcp": 60, "thumb_cmc_abd": 30})
    lft = HandModel("L").forward({"index_mcp": 60, "thumb_cmc_abd": 30})
    assert np.allclose(r * [-1, 1, 1], lft, atol=1e-6)


def test_hyper_limits_extend_natural_range():
    for k in PARAM_NAMES:
        nat, hyp = JOINT_LIMITS["natural"][k], JOINT_LIMITS["hyper"][k]
        assert hyp[0] <= nat[0] and hyp[1] >= nat[1]


def test_fitter_recovers_synthetic_pose():
    fitter = HandshapeFitter()
    true = {"index_mcp": 70, "index_pip": 80, "middle_mcp": 75, "middle_pip": 85, "thumb_cmc_flex": 30}
    rot = rotvec_matrix(np.array([0.3, -0.2, 0.1]))
    target = normalize_points(fitter.model.forward(true, rot))
    x, rms = fitter.fit(target)
    assert rms < 0.08
    params, _ = fitter.unpack(x)
    assert params["index_mcp"] > 35 and params["middle_mcp"] > 35


def test_average_quaternions_handles_sign_flip():
    q = np.array([0.0, 0.0, np.sin(0.2), np.cos(0.2)])
    avg = average_quaternions(np.array([q, -q, q]))
    assert np.allclose(np.abs(avg), np.abs(q), atol=1e-6)


def test_library_contains_alphabet_shapes(tmp_path):
    lib = build_library(tmp_path / "hs.json")
    shapes = lib["handshapes"]
    for name in AUTHORED:
        assert name in shapes
    alphabet = json.loads((tmp_path / "hs.json").read_text(encoding="utf-8"))
    assert set(alphabet["handshapes"]) == set(shapes)
    for q in shapes["puño"]["rotations"]["R"].values():
        assert abs(np.linalg.norm(q) - 1) < 1e-3
    assert resolve_params()  # combina entrenadas + diseñadas


def test_every_alphabet_shape_is_resolvable(lexicon):
    shapes = set(resolve_params())
    for letter, spec in lexicon.alphabet.items():
        assert spec["shape"] in shapes, letter


def test_every_lexicon_shape_is_resolvable(lexicon):
    shapes = set(resolve_params())
    for entry in lexicon.entries.values():
        for kf in entry["keyframes"]:
            for side in ("R", "L"):
                if kf.get(side):
                    assert kf[side]["shape"] in shapes, (entry["termino"], kf[side]["shape"])


def test_gloss_reorders_time_and_question(lexicon):
    assert lexicon.gloss("¿Dónde está el baño?") == ["BAÑO", "DONDE"]
    assert lexicon.gloss("Yo estudio mañana")[0] == "MAÑANA"


def test_lemmatization_and_enclitics(lexicon):
    assert lexicon.lookup("ayúdame")["termino"] == "ayudar"
    assert lexicon.lookup("juegan")["termino"] == "jugar"
    assert lexicon.lookup("libros")["termino"] == "libro"


def test_unknown_word_is_fingerspelled(lexicon):
    seq = lexicon.translate("Hola Ñandú")["secuencia"]
    assert seq[0]["tipo"] == "sena"
    assert seq[1]["tipo"] == "deletreo"
    assert [x["letra"] for x in seq[1]["letras"]] == ["Ñ", "A", "N", "D", "U"]


def test_parse_description_extracts_parameters():
    frames = parse_description("Ambas manos en forma de C frente a la boca, palma hacia abajo, se mueven hacia adelante dos veces.")
    assert frames[0]["R"]["shape"] == "LSM_C"
    assert frames[0]["R"]["loc"] == "boca"
    assert frames[0]["R"]["palma"] == "abajo"
    assert "L" in frames[0]
    assert any(f["R"].get("offset", [0, 0, 0])[2] > 0 for f in frames)
