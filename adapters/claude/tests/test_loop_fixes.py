"""Contract tests for the loop-fixes backlog — run WITHOUT SolidWorks on live recordings.

Recorded 2026-10-16 (SOLIDWORKS 2026 SP4, rebuilt execution server) from
Z:\\cadcritic\\live-run\\bracket_dimensioned.SLDPRT — the SPEC 11.1 bracket at nominal parameters,
fully dimensioned (19 driving dimensions) — and from the undimensioned proof bracket:
  fixtures/loop_fixes/bracket_faces.json / _feature_map.json / _features.json / _sketches.json
      the raw ExecutionResponses dimension_for_face consumes (sketches keyed by name)
  fixtures/loop_fixes/faces_pass.json       dimension_for_face output for every face, LIVE
  fixtures/loop_fixes/frame_circle.json     item 2: analyze_model(sketch) -> add_dimension with those
                                             exact numbers -> analyze_model(sketch) read-back
  fixtures/loop_fixes/nominal_check_contract_result.json + nominal_near_calls.json
      check_contract on the nominal bracket (equal pad-top areas -> the near tie-break decides)
"""
import copy
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_DIR = os.path.dirname(_HERE)
_FIX = os.path.join(_HERE, "fixtures", "loop_fixes")
sys.path.insert(0, _ADAPTER_DIR)

import contract_check   # noqa: E402
import face_dimension   # noqa: E402
import server           # noqa: E402


def _load(name):
    with open(os.path.join(_FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


def _payload(resp):
    return json.loads(resp["cadState"]["features"][0])


def _faces():
    return _payload(_load("bracket_faces.json"))["faces"]


def _fmap():
    return _payload(_load("bracket_feature_map.json"))


def _recipe():
    return _payload(_load("bracket_features.json"))


def _sketches():
    return {k: _payload(v) for k, v in _load("bracket_sketches.json").items()}


def _resolve(i, faces=None, fmap=None, sketches=None):
    faces = faces or _faces()
    sk = sketches or _sketches()
    face = next(f for f in faces if f["i"] == i)
    return face_dimension.resolve(face, fmap or _fmap(), _recipe(), lambda n: sk[n], faces)


# ---------------------------------------------------------------------------
# Item 1 — dimension_for_face
# ---------------------------------------------------------------------------
def test_offline_resolution_equals_live_for_every_face():
    live = _load("faces_pass.json")
    for f in _faces():
        off, on = _resolve(f["i"]), live[str(f["i"])]
        assert off["status"] == on["status"], f["i"]
        assert off.get("feature") == on.get("feature"), f["i"]
        assert (off.get("dimension") or {}).get("name") == (on.get("dimension") or {}).get("name"), f["i"]


def test_holes_with_equal_diameters_resolve_by_attachment_not_value():
    d12, d13 = _resolve(12)["dimension"], _resolve(13)["dimension"]
    assert d12["value_si"] == d13["value_si"] == 0.00845          # same value...
    assert d12["name"].startswith("D4@Sketch4") and d12["segments"] == [1]
    assert d13["name"].startswith("D1@Sketch4") and d13["segments"] == [0]   # ...different segments
    r = _resolve(12)
    assert r["feature"] == "Cut-Extrude1" and r["role"] == "cylinder_wall"
    assert [p["center_of"] for d in r["position_dimensions"] for p in d["points"]] == [1, 1]


def test_coplanar_pad_tops_belong_to_their_own_extrudes():
    assert _resolve(9)["feature"] == "Boss-Extrude2"
    assert _resolve(11)["feature"] == "Boss-Extrude3"
    assert _resolve(11)["dimension"]["name"].startswith("D1@Boss-Extrude3")


def test_hidden_creator_is_unattributed_never_the_first_coplanar_feature():
    # The pre-fix feature_map hid Boss-Extrude3's pad top (coplanar with Boss-Extrude2's).
    fmap = _fmap()
    for e in fmap["map"]:
        if e["feature"] == "Boss-Extrude3":
            e["created_faces"] = [c for c in e["created_faces"] if not c["planar"]]
    r = _resolve(11, fmap=fmap)
    assert r["status"] == "unattributed" and "feature" not in r


def test_line_length_dimensions_attach_through_endpoint_ids():
    r = _resolve(1)                                   # bottom wall <- line 0 (W)
    assert (r["role"], r["segment"]["index"]) == ("side_wall", 0)
    assert r["dimension"]["name"].startswith("D1@Sketch1") and r["dimension"]["value_si"] == 0.04
    sk = _sketches()
    for d in sk["Sketch1"]["dimensions"]:
        for p in d.get("points", []):
            p.pop("endpoint_of", None)                # without IDs nothing is attributed by value
    assert _resolve(1, sketches=sk)["status"] == "undimensioned"


def test_derived_inner_walls_are_undimensioned_with_add_parameters():
    r = _resolve(3)
    assert r["status"] == "undimensioned" and r["segment"]["index"] == 2
    seg = r["segment"]
    args = r["add_dimension"]
    assert args["px"] == pytest.approx((seg["x1"] + seg["x2"]) / 2)
    assert args["py"] == pytest.approx((seg["y1"] + seg["y2"]) / 2)
    assert args["value"] == pytest.approx(abs(seg["x2"] - seg["x1"]))   # current length: 34 mm


def test_caps_parallel_to_the_sketch_map_to_the_extrude_depth():
    r = _resolve(6)
    assert (r["role"], r["status"]) == ("extrude_cap", "dimensioned")
    assert r["dimension"]["name"].startswith("D1@Boss-Extrude1") and r["dimension"]["type"] == "depth"


@pytest.fixture
def offline_reads(monkeypatch):
    calls = []
    sk = _load("bracket_sketches.json")
    by_type = {"faces": _load("bracket_faces.json"), "feature_map": _load("bracket_feature_map.json"),
               "features": _load("bracket_features.json")}

    def fake_call_raw(tool, params):
        calls.append((tool, dict(params)))
        if tool == "analyze_model":
            return sk[params["name"]] if params["analysis_type"] == "sketch" else by_type[params["analysis_type"]]
        if tool == "add_dimension":
            return {"status": "COMPLETED", "result_geometry": {
                "dimension": "D5@Sketch1@bracket_dimensioned.Part", "short_name": "D5@Sketch1",
                "sketch": "Sketch1", "type": "linear", "requested": params["value"],
                "effective": params["value"]}}
        raise AssertionError(tool)

    monkeypatch.setattr(server, "_call_raw", fake_call_raw)
    monkeypatch.setattr(server, "_log", lambda msg: None)
    return calls


def test_tool_dimensioned_face(offline_reads):
    out = json.loads(server.dimension_for_face(13))
    assert out["status"] == "dimensioned" and out["dimension"]["name"].startswith("D1@Sketch4")
    assert not any(t == "add_dimension" for t, _p in offline_reads)


def test_tool_add_if_missing_dimensions_the_segment_at_its_size(offline_reads):
    out = json.loads(server.dimension_for_face(3, add_if_missing=True))
    assert out["status"] == "added" and out["dimension"]["name"] == "D5@Sketch1@bracket_dimensioned.Part"
    (_tool, params), = [c for c in offline_reads if c[0] == "add_dimension"]
    assert params["sketch_name"] == "Sketch1"
    assert params["value"] == pytest.approx(0.034)
    assert (params["px"], params["py"]) == pytest.approx((0.023, 0.006))   # middle of line 2


def test_tool_reports_missing_face(offline_reads):
    assert server.dimension_for_face(99).startswith("FAILED | FACE_NOT_FOUND")


# ---------------------------------------------------------------------------
# Item 2 — one sketch frame: analyze_model(sketch) numbers go straight into add_dimension
# ---------------------------------------------------------------------------
def test_live_circle_read_then_dimensioned_with_the_same_numbers():
    rec = _load("frame_circle.json")
    idx = rec["segment_index"]
    before, after = _payload(rec["sketch_before"]), _payload(rec["sketch_after"])
    frame = before["plane"]["frame"]
    assert frame["xdir"] != [1.0, 0.0, 0.0]        # a frame that is NOT model XY (the old bug's blind spot)
    c = before["sketch"]["segments"][idx]
    diam, pos_u, pos_v = rec["add_dimension"]
    assert diam["params"]["px"] == c["cx"] + c["radius"] and diam["params"]["py"] == c["cy"]
    assert (pos_u["params"]["px"], pos_u["params"]["py"]) == (c["cx"], c["cy"])
    assert before["dimensions"] == []
    dims = {d["name"]: d for d in after["dimensions"]}
    # item 5: add_dimension names what it created; the read-back finds exactly those
    created = [a["response"]["result_geometry"]["dimension"] for a in rec["add_dimension"]]
    assert set(created) == set(dims)
    assert dims[created[0]]["type"] == "diameter" and dims[created[0]]["segments"] == [idx]
    assert dims[created[0]]["value_si"] == pytest.approx(2 * c["radius"])
    for name in created[1:]:
        assert [p.get("center_of") for p in dims[name]["points"]] == [idx]
    assert after["sketch"]["segments"][idx] == c       # dimensioned at its size: geometry unchanged


def test_add_dimension_passes_sketch_frame_numbers_through(monkeypatch):
    sent = []
    monkeypatch.setattr(server, "_call", lambda tool, params: sent.append((tool, params)) or "OK")
    server.add_dimension(px=0.025, py=-0.025, value=0.009)
    server.add_dimension(px=0.025, py=-0.0205, value=0.025, sketch_name="Sketch3",
                         entity_type="point", to_origin=True, orientation="vertical")
    assert sent[0] == ("add_dimension", {"px": 0.025, "py": -0.025, "value": 0.009,
                                         "label_offset_x": 0.0, "label_offset_y": -0.015})
    assert sent[1][1] == {"px": 0.025, "py": -0.0205, "value": 0.025, "label_offset_x": 0.0,
                          "label_offset_y": -0.015, "sketch_name": "Sketch3", "entity_type": "point",
                          "to_origin": True, "orientation": "vertical"}


# ---------------------------------------------------------------------------
# Item 3 — check_contract near tie-break
# ---------------------------------------------------------------------------
def _nominal_report():
    """The raw cadcritic report: the live annotated result with the adapter's additions removed."""
    out = _load("nominal_check_contract_result.json")
    out.pop("score")
    out.pop("face_mapping")
    for req in out["requirements"]:
        for d in req["features"] + ((req.get("error") or {}).get("matched") or []):
            d.pop("face_index")
            d.pop("face_match")
    return out


def _recorded_near():
    table = {}
    for c in _load("nominal_near_calls.json"):
        assert c["params"]["k"] == 1
        table[tuple(json.loads(c["params"]["near"]))] = _payload(c["response"])["faces"][0]
    return lambda pt: table.get(tuple(pt))


def test_equal_area_pad_tops_resolved_by_near_as_live():
    out = contract_check.annotate_report(_nominal_report(), _faces(), _recorded_near())
    live = _load("nominal_check_contract_result.json")
    r10 = next(r for r in out["requirements"] if r["id"] == "R10-pads-coplanar")
    assert [(d["face_index"], d["face_match"]["resolved_by"]) for d in r10["features"]] == [(9, "near"), (11, "near")]
    assert r10["features"][0]["face_match"]["near"] == {"query_m": [-0.025, 0.025, 0.008], "face": 9, "dist_m": 0.0}
    # The live recording predates face_indices; every single-face feature has face_indices == [face_index].
    for req in out["requirements"]:
        for d in req["features"]:
            assert d.pop("face_indices") == ([d["face_index"]] if d["face_index"] is not None else [])
    assert out == live
    assert out["face_mapping"]["multiple"] == []


def test_without_near_the_equal_areas_stay_multiple():
    out = contract_check.annotate_report(_nominal_report(), _faces())
    assert {e["id"] for e in out["face_mapping"]["multiple"]} == {"plane:-25.000,25.000,8.000",
                                                                  "plane:25.000,25.000,8.000"}


def test_near_tie_or_foreign_answer_keeps_multiple():
    faces = _faces()
    p9 = next(f for f in faces if f["i"] == 9)["point"]
    tied = copy.deepcopy(faces)
    next(f for f in tied if f["i"] == 11)["point"] = list(p9)      # equally near the left centroid
    out = contract_check.annotate_report(_nominal_report(), tied, lambda pt: {"i": 9, "point": p9})
    m = {e["id"]: e for e in out["face_mapping"]["multiple"]}
    assert "equally near" in m["plane:-25.000,25.000,8.000"]["reason"]
    out = contract_check.annotate_report(_nominal_report(), faces, lambda pt: {"i": 4})   # not a candidate
    assert len(out["face_mapping"]["multiple"]) == 2
    out = contract_check.annotate_report(_nominal_report(), faces, lambda pt: None)       # lookup failed
    assert len(out["face_mapping"]["multiple"]) == 2


def test_check_contract_tool_queries_near_with_k1(monkeypatch):
    sent = []

    def fake_call_raw(tool, params):
        sent.append((tool, dict(params)))
        return next(c["response"] for c in _load("nominal_near_calls.json")
                    if c["params"]["near"] == params["near"])
    monkeypatch.setattr(server, "_call_raw", fake_call_raw)
    face = server._nearest_face([0.025, 0.025, 0.008])
    assert face["i"] == 11
    assert sent == [("analyze_model", {"analysis_type": "faces", "name": "",
                                       "near": "[0.025, 0.025, 0.008]", "k": 1})]
