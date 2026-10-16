"""Contract tests for check_contract (adapter-only; cadcritic behind a process boundary).

Runs WITHOUT SolidWorks and WITHOUT cadcritic: every input is a recording from one live run
(2026-10-16, SOLIDWORKS 2026 SP4, cadcritic 1.0.0) of the SPEC 11.1 bracket at nominal
parameters with ONE deliberate error — the right hole drilled at D=9.00 mm instead of 8.45:
  fixtures/check_contract/analyze_model_faces.json  raw ExecutionResponse of analyze_model(faces)
  fixtures/check_contract/cadcritic_check.json      argv / exit_code / stdout / stderr of
                                                    `cadcritic check active_part.step contract.json`
  fixtures/check_contract/contract.json             the demo contract (SPEC 11.2)
Expected live outcome: R04 fails with residual +0.45 mm on face 13 (the r=4.5 mm cylinder).
The two pad tops (both z=8 mm, normal +Z) each pass the geometric test on faces 9 AND 11; the
area tie-break separates them because the wrong hole makes the right pad's annulus smaller
(137.4 vs 145.0 mm2). With equal areas they would stay `multiple` — reported, not guessed.
"""
import ast
import copy
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_DIR = os.path.dirname(_HERE)
_FIX = os.path.join(_HERE, "fixtures", "check_contract")
sys.path.insert(0, _ADAPTER_DIR)

import contract_check  # noqa: E402
import server          # noqa: E402


def _load(name):
    with open(os.path.join(_FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


def _faces_response():
    return _load("analyze_model_faces.json")


def _faces():
    return json.loads(_faces_response()["cadState"]["features"][0])["faces"]


def _recorded_run():
    return _load("cadcritic_check.json")


def _report():
    return json.loads(_recorded_run()["stdout"])


def _req(report, rid):
    return next(r for r in report["requirements"] if r["id"] == rid)


def _strip_additions(report):
    out = copy.deepcopy(report)
    out.pop("score", None)
    out.pop("face_mapping", None)
    for req in out["requirements"]:
        for desc in req["features"] + ((req.get("error") or {}).get("matched") or []):
            desc.pop("face_index", None)
            desc.pop("face_indices", None)
            desc.pop("face_match", None)
    return out


# ---------------------------------------------------------------------------
# Mapping on the live recording
# ---------------------------------------------------------------------------
def test_wrong_hole_residual_points_at_its_face():
    out = contract_check.annotate_report(_report(), _faces())
    r04 = _req(out, "R04-hole-right-diameter")
    assert r04["status"] == "fail"
    assert r04["residual"] == pytest.approx(0.45)
    (feat,) = r04["features"]
    assert feat["id"] == "hole:25.000,25.000,4.000"
    assert feat["face_index"] == 13
    assert feat["face_match"] == {"status": "matched", "candidates": [13]}
    face13 = next(f for f in _faces() if f["i"] == 13)
    assert face13["kind"] == "cylinder" and face13["radius"] == pytest.approx(0.0045)


def test_every_feature_is_resolved_or_reported():
    out = contract_check.annotate_report(_report(), _faces())
    expected = {
        "hole:-25.000,25.000,4.000": 12,
        "hole:25.000,25.000,4.000": 13,
        "plane:0.000,22.653,6.000": 3,     # base top
        "plane:0.000,6.000,28.000": 4,     # flange front
        "plane:0.000,3.000,50.000": 5,     # flange top
        "plane:-0.061,19.806,0.000": 1,    # bottom
        "plane:-25.000,25.000,8.000": 11,  # left pad top  (area tie-break)
        "plane:25.000,25.000,8.000": 9,    # right pad top (area tie-break)
    }
    seen = {}
    for req in out["requirements"]:
        for d in req["features"]:
            seen[d["id"]] = d["face_index"]
    assert seen == expected
    fm = out["face_mapping"]
    assert (fm["features"], fm["matched"], fm["unmatched"], fm["multiple"]) == (8, 8, [], [])
    for d in _req(out, "R10-pads-coplanar")["features"]:
        assert d["face_match"]["resolved_by"] == "area"
        assert d["face_match"]["geometric_candidates"] == [9, 11]
    assert "resolved_by" not in _req(out, "R04-hole-right-diameter")["features"][0]["face_match"]


def _pad_top_desc():
    return copy.deepcopy(_req(_report(), "R10-pads-coplanar")["features"][0])  # left, 145.0 mm2


def test_equal_areas_stay_multiple():
    faces = _faces()
    for f in faces:
        if f["i"] == 9:
            f["area"] = 0.000145          # right pad top now indistinguishable from the left
    m = contract_check.match_descriptor(_pad_top_desc(), faces, "mm")
    assert m["status"] == "multiple" and m["candidates"] == [9, 11]
    assert "share the feature's area" in m["reason"]


def test_area_tie_break_tolerance():
    d = _pad_top_desc()
    d["area"] += 0.9                       # +0.9 mm2 = 0.9e-6 m2: inside AREA_TOL_M2
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [11]
    d["area"] += 1.0                       # +1.9 mm2: no face has this area any more
    m = contract_check.match_descriptor(d, _faces(), "mm")
    assert m["status"] == "multiple" and m["candidates"] == [9, 11]
    assert "none has the feature's area" in m["reason"]


def test_area_tie_break_is_unit_aware():
    d = _pad_top_desc()
    d["area"] /= 25.4 ** 2
    d["centroid"] = [x / 25.4 for x in d["centroid"]]
    d["units"] = {"area": "in2", "centroid": "in", "normal": "1"}
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [11]


def test_cylinder_without_area_stays_multiple():
    faces = _faces()
    twin = copy.deepcopy(next(f for f in faces if f["i"] == 13))
    twin.update(i=99, origin=[0.025, 0.025, -0.02])   # same axis line and radius, another face
    faces.append(twin)
    m = contract_check.match_descriptor(_hole_desc(), faces, "mm")
    assert m["status"] == "multiple" and m["candidates"] == [13, 99]
    assert "no area" in m["reason"]


def test_score_pair_and_cadcritic_report_otherwise_unchanged():
    original = _report()
    out = contract_check.annotate_report(original, _faces())
    assert out["score"] == [0, pytest.approx(2.255048)]
    assert out["score"] == [original["error_count"], original["violation_score"]]
    assert _strip_additions(out) == original
    assert "face_index" not in json.dumps(original)   # input not mutated


# ---------------------------------------------------------------------------
# Units and tolerances
# ---------------------------------------------------------------------------
_LENGTH_MEMBERS = ("centroid", "center", "radius", "length")


def _to_unit(report, unit, per_mm):
    out = copy.deepcopy(report)
    out["units"]["length"] = unit
    for req in out["requirements"]:
        for d in req["features"]:
            for key in _LENGTH_MEMBERS:
                if key in d:
                    d[key] = ([x * per_mm for x in d[key]] if isinstance(d[key], list)
                              else d[key] * per_mm)
            if "area" in d:
                d["area"] *= per_mm ** 2
            d["units"] = {k: (unit if v == "mm" else unit + "2" if v == "mm2" else v)
                          for k, v in d["units"].items()}
    return out


@pytest.mark.parametrize("unit,per_mm", [("in", 1 / 25.4), ("cm", 0.1), ("m", 1e-3)])
def test_contract_unit_is_converted_to_metres(unit, per_mm):
    base = contract_check.annotate_report(_report(), _faces())
    conv = contract_check.annotate_report(_to_unit(_report(), unit, per_mm), _faces())
    pick = lambda rep: [(d["id"], d["face_index"], d["face_match"]["status"])  # noqa: E731
                        for r in rep["requirements"] for d in r["features"]]
    assert pick(conv) == pick(base)


def _hole_desc():
    return copy.deepcopy(_req(_report(), "R04-hole-right-diameter")["features"][0])


def test_tolerances_are_ten_microns():
    assert (contract_check.DIR_TOL, contract_check.LINEAR_TOL_M) == (1e-5, 1e-5)


def test_radius_tolerance():
    d = _hole_desc()
    d["radius"] = 4.5 + 0.009             # +9 µm
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [13]
    d["radius"] = 4.5 + 0.015             # +15 µm
    assert contract_check.match_descriptor(d, _faces(), "mm")["status"] == "unmatched"


def test_axis_line_offset_and_tilt_are_checked():
    d = _hole_desc()
    d["center"] = [25.009, 25.0, 4.0]     # axis line moved 9 µm sideways: inside
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [13]
    d["center"] = [25.015, 25.0, 4.0]     # 15 µm: outside
    assert contract_check.match_descriptor(d, _faces(), "mm")["status"] == "unmatched"
    d = _hole_desc()
    d["center"] = [25.0, 25.0, 30.0]      # slid ALONG the axis: same line, still matches
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [13]
    d = _hole_desc()
    d["axis"] = [9e-6, 0.0, 1.0]          # sin(tilt) = 9e-6 <= 1e-5
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [13]
    d["axis"] = [2e-5, 0.0, 1.0]          # sin(tilt) = 2e-5 > 1e-5
    assert contract_check.match_descriptor(d, _faces(), "mm")["status"] == "unmatched"


def test_plane_offset_tolerance_and_sign_free_normal():
    d = copy.deepcopy(_req(_report(), "R09-flange-top-parallel")["features"][0])  # z = 50 mm
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [5]
    d["centroid"][2] = 50.009             # 9 µm: inside
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [5]
    d["centroid"][2] = 50.015             # 15 µm: outside
    assert contract_check.match_descriptor(d, _faces(), "mm")["status"] == "unmatched"
    d["centroid"][2] = 50.0
    d["normal"] = [0.0, 0.0, -1.0]        # parallel, as specified — sign is not a criterion
    assert contract_check.match_descriptor(d, _faces(), "mm")["candidates"] == [5]


def test_unknown_type_and_unit_are_unmatched_with_reason():
    d = _hole_desc()
    d["type"] = "cone"
    m = contract_check.match_descriptor(d, _faces(), "mm")
    assert m["status"] == "unmatched" and "type" in m["reason"]
    d = _hole_desc()
    d["units"]["center"] = "furlong"
    m = contract_check.match_descriptor(d, _faces(), "mm")
    assert m["status"] == "unmatched" and "unit" in m["reason"]


def test_error_matched_descriptors_get_face_index():
    rep = _report()
    r03 = _req(rep, "R03-hole-left-diameter")
    hole_l, hole_r = (_req(rep, "R05-hole-pitch")["features"])
    r03.update(status="error", measured=None, residual=None, violation=None, features=[],
               details={}, error={"kind": "ambiguous", "selector": "select",
                                  "matched": [hole_l, hole_r]})
    out = contract_check.annotate_report(rep, _faces())
    assert [d["face_index"] for d in _req(out, "R03-hole-left-diameter")["error"]["matched"]] == [12, 13]


# ---------------------------------------------------------------------------
# Process boundary
# ---------------------------------------------------------------------------
def test_parse_report_accepts_exit_0_and_1_only():
    rec = _recorded_run()
    report, failure = contract_check.parse_report(rec["exit_code"], rec["stdout"], rec["stderr"])
    assert failure is None and report["command"] == "check"
    usage = json.dumps({"error": {"code": 2, "kind": "usage", "message": "bad contract"}})
    report, failure = contract_check.parse_report(2, usage, "cadcritic: error: bad contract")
    assert report is None and failure.startswith("FAILED | CADCRITIC_USAGE | exit=2")
    report, failure = contract_check.parse_report(4, "", "cadcritic: error: boolean failed\n")
    assert report is None and failure == "FAILED | CADCRITIC_FAILED | exit=4 | cadcritic: error: boolean failed"


def test_cadcritic_is_never_imported():
    for name in ("server.py", "contract_check.py"):
        with open(os.path.join(_ADAPTER_DIR, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "cadcritic" for a in node.names), name
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "cadcritic", name
    assert "cadcritic" not in sys.modules


@pytest.fixture
def offline_tool(monkeypatch, tmp_path):
    """check_contract wired to the recordings: export writes a placeholder STEP, analyze_model
    returns the recorded faces, and subprocess.run returns the recorded cadcritic stdout."""
    calls = {"rest": [], "argv": None, "kwargs": None, "step_dir": None}
    rec = _recorded_run()

    def fake_call_raw(tool, params):
        calls["rest"].append((tool, dict(params)))
        if tool == "export_document":
            calls["step_dir"] = os.path.dirname(params["file_path"])
            with open(params["file_path"], "w") as fh:
                fh.write("ISO-10303-21;\n")
            return {"status": "COMPLETED", "stateVersion": 7, "cadState": {}}
        if tool == "analyze_model":
            return _faces_response()
        raise AssertionError(f"unexpected execution call {tool}")

    def fake_run(argv, **kwargs):
        calls["argv"], calls["kwargs"] = argv, kwargs
        return subprocess.CompletedProcess(argv, rec["exit_code"], rec["stdout"].encode("utf-8"),
                                           rec["stderr"].encode("utf-8"))

    exe = tmp_path / "cadcritic.exe"
    exe.write_text("")
    monkeypatch.setattr(server, "_call_raw", fake_call_raw)
    monkeypatch.setattr(server, "CADCRITIC_EXE", str(exe))
    monkeypatch.setattr(contract_check.subprocess, "run", fake_run)
    monkeypatch.setattr(server, "_log", lambda msg: None)
    calls["exe"] = str(exe)
    return calls


def test_tool_end_to_end_offline(offline_tool):
    contract = os.path.join(_FIX, "contract.json")
    out = json.loads(server.check_contract(contract))

    rest = offline_tool["rest"]
    assert rest[0][0] == "export_document" and rest[0][1]["format"] == "STEP"
    assert rest[1] == ("analyze_model", {"analysis_type": "faces", "name": ""})
    step = rest[0][1]["file_path"]
    assert offline_tool["argv"] == [offline_tool["exe"], "check", step, os.path.abspath(contract)]
    assert offline_tool["kwargs"]["stdin"] is subprocess.DEVNULL
    assert not os.path.exists(offline_tool["step_dir"])          # temp STEP cleaned up

    assert out["score"] == [0, pytest.approx(2.255048)]
    assert _req(out, "R04-hole-right-diameter")["features"][0]["face_index"] == 13


def test_tool_reports_missing_contract_and_failed_export(offline_tool, monkeypatch, tmp_path):
    assert server.check_contract(str(tmp_path / "nope.json")).startswith("FAILED | CONTRACT_NOT_FOUND")
    monkeypatch.setattr(server, "_call_raw", lambda tool, params: {
        "status": "FAILED", "error": {"code": "NO_ACTIVE_DOCUMENT", "message": "none"}})
    out = server.check_contract(os.path.join(_FIX, "contract.json"))
    assert out == "FAILED | EXPORT_FAILED | NO_ACTIVE_DOCUMENT: none"
    assert offline_tool["argv"] is None                            # cadcritic never ran


def test_tool_surfaces_cadcritic_error_document(offline_tool, monkeypatch):
    doc = json.dumps({"error": {"code": 3, "kind": "input", "message": "part has 2 solids"}})
    monkeypatch.setattr(contract_check.subprocess, "run", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 3, doc.encode(), b"cadcritic: error: x"))
    out = server.check_contract(os.path.join(_FIX, "contract.json"))
    assert out == "FAILED | CADCRITIC_INPUT | exit=3 | part has 2 solids"
