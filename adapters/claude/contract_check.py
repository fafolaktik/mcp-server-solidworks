"""check_contract support: run the cadcritic CLI and map its features onto SolidWorks faces.

cadcritic is reached ONLY as a subprocess (`cadcritic check PART CONTRACT`) — never imported. It
pulls build123d/OpenCascade, and keeping it behind a process boundary means a missing or broken
install degrades to one clean tool error instead of touching the MCP server process.

Face mapping is purely geometric. A cadcritic feature descriptor (SPEC.md section 9.1) is paired
with an analyze_model('faces') entry when ALL of these hold, in METRES after converting
cadcritic's contract length unit:
  plane       <-> planar face: normals parallel (|n1 x n2| <= DIR_TOL) AND the face's point lies
                  within LINEAR_TOL_M of the feature's plane (through its centroid);
  hole / boss <-> cylinder:    axes parallel (|a1 x a2| <= DIR_TOL) AND |r1 - r2| <= LINEAR_TOL_M
                  AND the face's axis origin lies within LINEAR_TOL_M of the feature's axis line.
analyze_model reports no inside/outside sense for a cylinder, so hole and boss both map to
`cylinder`.

Tie-break by area, when several faces pass the geometric test (cadcritic >= 1.1 reports `area` on
every descriptor; 1.0 only on planes — without it this step is skipped):
  plane       keep the single face whose area agrees within AREA_TOL_M2.
  hole / boss the candidates share one axis line and radius, so they are either pieces of ONE
              wall (a cylinder split into several faces) or walls of DIFFERENT coaxial features
              (an interrupted bore). Find the set of candidate faces whose SUMMED area agrees
              with the feature's area (tolerance AREA_TOL_M2 per face summed). Exactly one such
              set -> the feature is those faces — one face for an interrupted bore, all of them
              for a split wall. Several sets -> still ambiguous.
If area cannot decide, ask SolidWorks for the face nearest cadcritic's centroid
(analyze_model(faces, near=centroid_m, k=1)); it decides only when it is a remaining candidate and
no other candidate is equally near. NOTE: `near` measures distance to each face's representative
(UV-mid) point — a proximity ranking, not a point-on-face test (analyze_model exposes no face
boundary). A resolved feature gets face_indices (every face of it) and face_index (the lowest of
them), tagged `resolved_by: area|near` when a tie-break decided. Otherwise face_index null with
the reason and the candidate indices. Never a guess.

Precision note: analyze_model rounds to 6 decimals in SI units — <= 5e-7 m per coordinate and
<= 5e-7 m2 per area (i.e. areas are only known to ~1 mm2).
"""
import json
import math
import os
import shutil
import subprocess
import sys

DIR_TOL = 1e-5        # |sin(angle)| between unit vectors
LINEAR_TOL_M = 1e-5   # radius, point-on-axis, plane offset — metres
AREA_TOL_M2 = 1e-6    # tie-break only: two rounding half-steps of analyze_model's m2 area, per face
MAX_SUBSET_FACES = 12 # hole/boss summed-area search is exhaustive (2^n sets) up to this many faces

_UNIT_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}

# cadcritic exit codes (SPEC.md section 2): 0/1 = completed report; 2/3/4 = error document.
_REPORT_EXITS = (0, 1)


# ---------------------------------------------------------------------------
# Process boundary
# ---------------------------------------------------------------------------
def find_cadcritic(configured: str = "") -> str:
    """Resolve the cadcritic executable: explicit config, else the console script installed
    beside THIS interpreter (the adapter's own environment), else PATH. '' when not found."""
    if configured:
        return configured if os.path.isfile(configured) else ""
    scripts = os.path.dirname(sys.executable)
    for name in ("cadcritic.exe", "cadcritic"):
        for folder in (scripts, os.path.join(scripts, "Scripts")):
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate):
                return candidate
    return shutil.which("cadcritic") or ""


def run_cadcritic(exe: str, step_path: str, contract_path: str, timeout: float):
    """Run `cadcritic check STEP CONTRACT`; return (exit_code, stdout_text, stderr_text).

    stdin is DEVNULL: the MCP server speaks JSON-RPC over its own stdin/stdout, and an inherited
    handle would let the child read (or write) the protocol stream. CREATE_NO_WINDOW stops a
    console flash on Windows. Raises subprocess.TimeoutExpired / OSError to the caller."""
    creationflags = 0x08000000 if os.name == "nt" else 0
    proc = subprocess.run(
        [exe, "check", step_path, contract_path],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        creationflags=creationflags,
    )
    return (proc.returncode,
            proc.stdout.decode("utf-8", errors="replace"),
            proc.stderr.decode("utf-8", errors="replace"))


def parse_report(exit_code: int, stdout: str, stderr: str):
    """Return (report, None) for a completed check, or (None, 'FAILED | ...') otherwise."""
    try:
        doc = json.loads(stdout) if stdout.strip() else None
    except ValueError:
        doc = None
    if exit_code in _REPORT_EXITS and isinstance(doc, dict) and doc.get("command") == "check":
        return doc, None
    err = (doc or {}).get("error") if isinstance(doc, dict) else None
    if isinstance(err, dict):
        kind = str(err.get("kind") or "unknown").upper()
        return None, f"FAILED | CADCRITIC_{kind} | exit={exit_code} | {err.get('message')}"
    tail = (stderr.strip().splitlines() or ["(no stderr)"])[-1]
    return None, f"FAILED | CADCRITIC_FAILED | exit={exit_code} | {tail}"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _vec(v):
    if not (isinstance(v, (list, tuple)) and len(v) == 3):
        return None
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else None


def _cross_norm(a, b):
    return math.sqrt((a[1] * b[2] - a[2] * b[1]) ** 2
                     + (a[2] * b[0] - a[0] * b[2]) ** 2
                     + (a[0] * b[1] - a[1] * b[0]) ** 2)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _scale(v, s):
    return [x * s for x in v]


_MATCHED_LENGTHS = {"plane": ("centroid",), "hole": ("center", "radius"), "boss": ("center", "radius")}


def _length_unit(descriptor: dict, report_unit: str):
    """The unit of the length members the match actually compares (the descriptor's own `units`
    map wins over the report's). Members that disagree yield a unit name that never converts."""
    units = descriptor.get("units") or {}
    found = {units.get(k) or report_unit for k in _MATCHED_LENGTHS.get(descriptor.get("type"), ())}
    if len(found) > 1:
        return "/".join(sorted(str(u) for u in found))
    return found.pop() if found else report_unit


def _plane_matches(desc: dict, to_m: float, face: dict) -> bool:
    if not face.get("planar"):
        return False
    n_cc, c_cc = _vec(desc.get("normal")), _vec(desc.get("centroid"))
    n_sw, p_sw = _vec(face.get("normal")), _vec(face.get("point"))
    if None in (n_cc, c_cc, n_sw, p_sw):
        return False
    n_cc, n_sw = _unit(n_cc), _unit(n_sw)
    if n_cc is None or n_sw is None or _cross_norm(n_cc, n_sw) > DIR_TOL:
        return False
    return abs(_dot(n_cc, _sub(p_sw, _scale(c_cc, to_m)))) <= LINEAR_TOL_M


def _cylinder_matches(desc: dict, to_m: float, face: dict) -> bool:
    if face.get("kind") != "cylinder":
        return False
    a_cc, c_cc = _vec(desc.get("axis")), _vec(desc.get("center"))
    a_sw, o_sw = _vec(face.get("axis")), _vec(face.get("origin"))
    if None in (a_cc, c_cc, a_sw, o_sw):
        return False
    try:
        r_cc, r_sw = float(desc["radius"]) * to_m, float(face["radius"])
    except (KeyError, TypeError, ValueError):
        return False
    a_cc, a_sw = _unit(a_cc), _unit(a_sw)
    if a_cc is None or a_sw is None or _cross_norm(a_cc, a_sw) > DIR_TOL:
        return False
    if abs(r_cc - r_sw) > LINEAR_TOL_M:
        return False
    # Distance from SolidWorks' axis origin to cadcritic's axis line.
    return _cross_norm(_sub(o_sw, _scale(c_cc, to_m)), a_cc) <= LINEAR_TOL_M


def match_descriptor(desc: dict, faces: list, report_unit: str, near_lookup=None) -> dict:
    """{'status': matched|unmatched|multiple, 'candidates': [...], 'reason'?: str}.

    near_lookup(point_m) -> the analyze_model(faces, near=point_m, k=1) face dict (or None):
    the last tie-break, used only when the geometric test and area leave several faces."""
    ftype = desc.get("type")
    unit = _length_unit(desc, report_unit)
    if unit not in _UNIT_TO_M:
        return {"status": "unmatched", "candidates": [], "reason": f"unknown length unit {unit!r}"}
    if ftype == "plane":
        test = _plane_matches
    elif ftype in ("hole", "boss"):
        test = _cylinder_matches
    else:
        return {"status": "unmatched", "candidates": [], "reason": f"unknown feature type {ftype!r}"}
    to_m = _UNIT_TO_M[unit]
    hits = sorted((f for f in faces
                   if isinstance(f, dict) and isinstance(f.get("i"), int) and test(desc, to_m, f)),
                  key=lambda f: f["i"])
    candidates = [f["i"] for f in hits]
    if len(candidates) == 1:
        return {"status": "matched", "candidates": candidates}
    if not candidates:
        return {"status": "unmatched", "candidates": [],
                "reason": "no analyze_model face within tolerance"}
    m = _break_tie_by_area(desc, to_m, hits)
    if m["status"] == "multiple" and near_lookup is not None:
        m = _break_tie_by_near(desc, to_m, hits, m, near_lookup)
    return m


def _break_tie_by_near(desc: dict, to_m: float, hits: list, area_result: dict, near_lookup) -> dict:
    """Area left several faces: ask SolidWorks for the face nearest cadcritic's centroid
    (analyze_model(faces, near=..., k=1) — distance to each face's representative point).
    Resolved only when that face is one of the remaining candidates AND no other remaining
    candidate is within LINEAR_TOL_M of the same distance (a tie k=1 alone would hide)."""
    remaining = area_result["candidates"]
    centroid = _vec(desc.get("centroid"))
    if centroid is None:
        return dict(area_result, reason=area_result["reason"] + "; no centroid for a near query")
    query = [round(c * to_m, 9) for c in centroid]
    nearest = near_lookup(query)
    if not isinstance(nearest, dict) or nearest.get("i") not in remaining:
        got = nearest.get("i") if isinstance(nearest, dict) else None
        return dict(area_result, reason=area_result["reason"]
                    + f"; nearest face to the centroid is {got}, not a candidate")
    best = nearest["i"]
    pts = {f["i"]: _vec(f.get("point")) for f in hits}
    dist = {i: math.dist(p, query) for i, p in pts.items() if p is not None and i in remaining}
    if best not in dist:
        return dict(area_result, reason=area_result["reason"] + "; nearest face has no point")
    rivals = [i for i in remaining if i != best and i in dist
              and abs(dist[i] - dist[best]) <= LINEAR_TOL_M]
    near_info = {"query_m": query, "face": best, "dist_m": round(dist[best], 9)}
    if rivals:
        return dict(area_result, reason=area_result["reason"]
                    + f"; faces {sorted([best] + rivals)} are equally near the centroid",
                    near=near_info)
    out = {"status": "matched", "candidates": [best], "resolved_by": "near",
           "geometric_candidates": [f["i"] for f in hits], "near": near_info}
    if desc.get("area") is not None:
        out["area_result"] = area_result["reason"]
    return out


def _break_tie_by_area(desc: dict, to_m: float, hits: list) -> dict:
    """Several geometric candidates: decide by cadcritic's area, if reported (see module doc)."""
    candidates = [f["i"] for f in hits]
    area = desc.get("area")
    if not isinstance(area, (int, float)) or isinstance(area, bool):
        return {"status": "multiple", "candidates": candidates,
                "reason": f"{len(candidates)} faces within tolerance; cadcritic reports no area "
                          f"for a {desc.get('type')} to separate them"}
    area_m2 = float(area) * to_m * to_m
    if desc.get("type") in ("hole", "boss"):
        return _break_cylinder_tie_by_summed_area(area_m2, hits)
    same = [f["i"] for f in hits if isinstance(f.get("area"), (int, float))
            and abs(float(f["area"]) - area_m2) <= AREA_TOL_M2]
    if len(same) == 1:
        return {"status": "matched", "candidates": same, "resolved_by": "area",
                "geometric_candidates": candidates}
    if not same:
        return {"status": "multiple", "candidates": candidates,
                "reason": f"{len(candidates)} faces within tolerance; none has the feature's area "
                          f"(the feature may span several faces)"}
    return {"status": "multiple", "candidates": same,
            "reason": f"{len(same)} faces within tolerance share the feature's area",
            "geometric_candidates": candidates}


def _break_cylinder_tie_by_summed_area(area_m2: float, hits: list) -> dict:
    """Hole/boss: the candidates share the feature's axis line and radius. The feature is the ONE
    set of them whose summed area equals its area — a single face (another coaxial feature owns
    the rest) or several (one wall split into faces). Exhaustive over the candidate sets."""
    candidates = [f["i"] for f in hits]
    areas = [(f["i"], float(f["area"])) for f in hits if isinstance(f.get("area"), (int, float))]
    if len(areas) != len(hits):
        return {"status": "multiple", "candidates": candidates,
                "reason": "a candidate face reports no area — summed-area test not possible"}
    if len(areas) > MAX_SUBSET_FACES:
        return {"status": "multiple", "candidates": candidates,
                "reason": f"{len(areas)} coaxial candidate faces — beyond the summed-area search limit"}
    matches = []
    for mask in range(1, 1 << len(areas)):
        subset = [areas[k] for k in range(len(areas)) if mask >> k & 1]
        if abs(sum(a for _i, a in subset) - area_m2) <= AREA_TOL_M2 * len(subset):
            matches.append(sorted(i for i, _a in subset))
    if len(matches) == 1:
        return {"status": "matched", "candidates": matches[0], "resolved_by": "area",
                "geometric_candidates": candidates}
    if not matches:
        return {"status": "multiple", "candidates": candidates,
                "reason": f"{len(candidates)} coaxial faces within tolerance; no set of them sums "
                          f"to the feature's area"}
    return {"status": "multiple", "candidates": sorted({i for m in matches for i in m}),
            "reason": f"{len(matches)} different sets of coaxial faces sum to the feature's area",
            "geometric_candidates": candidates, "area_sets": matches}


def _descriptor_lists(requirement: dict):
    yield requirement.get("features") or []
    err = requirement.get("error")
    if isinstance(err, dict):
        yield err.get("matched") or []


def annotate_report(report: dict, faces: list, near_lookup=None) -> dict:
    """Return a copy of cadcritic's check report with `face_index`, `face_indices` and
    `face_match` on every feature descriptor, a top-level `score` = [error_count,
    violation_score] (SPEC 9.2's comparison key) and a `face_mapping` summary listing every
    unresolved feature by id. face_indices = every face of a resolved feature (several for a
    split cylinder wall), face_index = the lowest of them; [] / null when unresolved.
    near_lookup: see match_descriptor (None = no near tie-break)."""
    out = json.loads(json.dumps(report))
    report_unit = (out.get("units") or {}).get("length", "mm")
    resolved: dict = {}      # feature id -> match result (identical for every occurrence)
    used_by: dict = {}       # feature id -> requirement ids, in contract order
    for req in out.get("requirements") or []:
        for descriptors in _descriptor_lists(req):
            for desc in descriptors:
                fid = desc.get("id")
                if fid not in resolved:
                    resolved[fid] = match_descriptor(desc, faces, report_unit, near_lookup)
                m = resolved[fid]
                matched = m["status"] == "matched"
                desc["face_index"] = m["candidates"][0] if matched else None
                desc["face_indices"] = list(m["candidates"]) if matched else []
                desc["face_match"] = dict(m)
                ids = used_by.setdefault(fid, [])
                if req.get("id") not in ids:
                    ids.append(req.get("id"))

    def _entries(status):
        return [{"id": fid, "candidates": m["candidates"], "reason": m.get("reason"),
                 "requirements": used_by[fid]}
                for fid, m in resolved.items() if m["status"] == status]

    out["score"] = [out.get("error_count"), out.get("violation_score")]
    out["face_mapping"] = {
        "source": "analyze_model(faces)",
        "tolerance": {"direction_sin": DIR_TOL, "linear_m": LINEAR_TOL_M,
                      "area_tie_break_m2": AREA_TOL_M2},
        "features": len(resolved),
        "matched": sum(1 for m in resolved.values() if m["status"] == "matched"),
        "unmatched": _entries("unmatched"),
        "multiple": _entries("multiple"),
    }
    return out
