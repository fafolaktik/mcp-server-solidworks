"""dimension_for_face support: face index -> creating feature -> profile sketch -> segment -> dimension.

Pure functions over the payloads of existing reads (no SolidWorks access here):
  analyze_model('faces')         the target face (planar: normal + point; cylinder: origin/axis/radius)
  analyze_model('feature_map')   which feature CREATED a face on the target's surface
  analyze_model('features')      the feature's `profile_sketch` and its own dimensions (depth)
  analyze_model('sketch', name)  segments, the sketch frame, and each dimension's ATTACHED segments

Every step must resolve to exactly one thing or the result says why it stopped — status values:
  dimensioned     exactly one driving dimension is attached to the matched segment (or, for a cap
                  face, the extrude's single depth dimension)
  undimensioned   the segment was found but no dimension drives it (the segment is returned)
  added           undimensioned, and add_if_missing added one (filled in by the tool)
  not_dimension_driven  the face lies IN the sketch plane — positioned by the plane, no dimension
  multiple_dimensions / ambiguous_feature / ambiguous_segment / unattributed / no_segment /
  no_profile_sketch / unsupported   — nothing chosen; candidates are listed
Never a guess: value coincidences are not used to attribute anything.
"""
import math

TOL = 1e-5          # metres / |sin| — the same scale as check_contract's matching
_DRIVING_TYPES = {"circle": ("diameter", "radial"), "line": ("linear", "horizontal", "vertical")}


def _v(x):
    try:
        return [float(c) for c in x] if isinstance(x, (list, tuple)) and len(x) == 3 else None
    except (TypeError, ValueError):
        return None


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return [c / n for c in a] if n > 0 else None


def _line_dist(p, origin, axis):
    return _norm(_cross(_sub(p, origin), axis))


def face_kind(face: dict):
    if face.get("planar") and _v(face.get("normal")) and _v(face.get("point")):
        return "plane"
    if face.get("kind") == "cylinder" and _v(face.get("axis")) and _v(face.get("origin")):
        return "cylinder"
    return None


# ---------------------------------------------------------------------------
# face -> feature
# ---------------------------------------------------------------------------
def _created_on_surface(created: dict, face: dict, kind: str) -> bool:
    p = _v(created.get("point"))
    if p is None:
        return False
    if kind == "plane":
        n_f, n_c = _unit(_v(face["normal"])), _unit(_v(created.get("normal")) or [0, 0, 0])
        if not created.get("planar") or n_f is None or n_c is None:
            return False
        if _norm(_cross(n_f, n_c)) > TOL or _dot(n_f, n_c) <= 0:
            return False
        return abs(_dot(n_f, _sub(p, _v(face["point"])))) <= TOL
    # cylinder: a non-planar created face whose representative point is on the same cylinder
    if created.get("planar"):
        return False
    axis = _unit(_v(face["axis"]))
    return abs(_line_dist(p, _v(face["origin"]), axis) - float(face["radius"])) <= TOL


def _same_surface_faces(face: dict, kind: str, faces: list) -> int:
    """How many CURRENT faces lie on the target's surface (itself included)."""
    probe = {"planar": kind == "plane"}
    n = 0
    for f in faces:
        if face_kind(f) != kind or not _v(f.get("point")):
            continue
        if kind == "plane":
            if _created_on_surface(dict(probe, point=f["point"], normal=f["normal"]), face, kind):
                n += 1
        elif (abs(float(f["radius"]) - float(face["radius"])) <= TOL
              and _norm(_cross(_unit(_v(f["axis"])), _unit(_v(face["axis"])))) <= TOL
              and _line_dist(_v(f["origin"]), _v(face["origin"]), _unit(_v(face["axis"]))) <= TOL):
            n += 1
    return n


def creating_features(face: dict, feature_map: dict, faces: list = ()) -> list:
    """Features whose created_faces lie on the target face's surface. When the surface is
    shared — several owners, or several CURRENT faces on it (disconnected coplanar pad tops,
    interrupted coaxial bores) — an owner must also have created a face at the target's
    representative point; surface equality alone would attribute every such face to the
    first creator."""
    kind = face_kind(face)
    owners = []
    for entry in feature_map.get("map") or []:
        hits = [c for c in entry.get("created_faces") or [] if _created_on_surface(c, face, kind)]
        if hits:
            owners.append((entry.get("feature"), hits))
    if len(owners) > 1 or _same_surface_faces(face, kind, list(faces)) > 1:
        p = _v(face.get("point"))
        owners = [(name, hits) for name, hits in owners
                  if p and any(_norm(_sub(_v(c["point"]), p)) <= TOL for c in hits)]
    return [name for name, _hits in owners]


# ---------------------------------------------------------------------------
# sketch geometry in model space
# ---------------------------------------------------------------------------
class Frame:
    def __init__(self, frame: dict):
        self.o = _v(frame.get("origin"))
        self.x = _v(frame.get("xdir"))
        self.y = _v(frame.get("ydir"))
        self.n = _unit(_cross(self.x, self.y)) if self.x and self.y else None

    def ok(self):
        return None not in (self.o, self.x, self.y, self.n)

    def to_model(self, u, v):
        return [self.o[i] + u * self.x[i] + v * self.y[i] for i in range(3)]


def _segment_kind(seg: dict):
    if seg.get("kind") == "line":
        return "line"
    if seg.get("kind") == "arc/circle":
        return "circle"
    return None


def match_segments(face: dict, kind: str, segments: list, frame: Frame) -> list:
    """Indices of the (non-construction) sketch segments that sweep into this face."""
    hits = []
    for idx, seg in enumerate(segments):
        if seg.get("construction"):
            continue
        sk = _segment_kind(seg)
        try:
            if kind == "plane" and sk == "line":
                n_f, p_f = _unit(_v(face["normal"])), _v(face["point"])
                a = frame.to_model(float(seg["x1"]), float(seg["y1"]))
                b = frame.to_model(float(seg["x2"]), float(seg["y2"]))
                if abs(_dot(n_f, _sub(a, p_f))) <= TOL and abs(_dot(n_f, _sub(b, p_f))) <= TOL:
                    hits.append(idx)
            elif kind == "cylinder" and sk == "circle":
                c = frame.to_model(float(seg["cx"]), float(seg["cy"]))
                axis = _unit(_v(face["axis"]))
                if (abs(float(seg["radius"]) - float(face["radius"])) <= TOL
                        and _line_dist(c, _v(face["origin"]), axis) <= TOL):
                    hits.append(idx)
        except (KeyError, TypeError, ValueError):
            continue
    return hits


def add_dimension_args(seg: dict) -> dict:
    """add_dimension parameters that dimension this segment AT ITS CURRENT SIZE (sketch frame)."""
    if _segment_kind(seg) == "circle":
        r = float(seg["radius"])
        return {"px": float(seg["cx"]) + r, "py": float(seg["cy"]), "value": 2 * r,
                "label_offset_x": 0.004, "label_offset_y": 0.004}
    x1, y1, x2, y2 = (float(seg[k]) for k in ("x1", "y1", "x2", "y2"))
    return {"px": (x1 + x2) / 2, "py": (y1 + y2) / 2, "value": math.hypot(x2 - x1, y2 - y1),
            "label_offset_x": 0.0, "label_offset_y": -0.008}


# ---------------------------------------------------------------------------
# the whole chain
# ---------------------------------------------------------------------------
def _attached_to(dim: dict, idx: int, seg_kind: str) -> bool:
    """A dimension drives segment idx when SolidWorks attached it to the segment itself, or —
    for a line LENGTH, which SolidWorks attaches to the end points — when it is attached to
    exactly two points that are both end points of idx (endpoint_of, resolved by entity ID)."""
    if idx in (dim.get("segments") or []):
        return True
    pts = dim.get("points") or []
    return (seg_kind == "line" and not dim.get("segments") and len(pts) == 2
            and all(idx in (p.get("endpoint_of") or []) for p in pts))


def resolve(face: dict, feature_map: dict, recipe: dict, read_sketch, faces: list = ()) -> dict:
    """read_sketch(name) -> the analyze_model(sketch, name) payload (dict); faces = the full
    analyze_model(faces) list (needed to detect surfaces shared by several faces)."""
    out = {"face_index": face.get("i")}
    kind = face_kind(face)
    out["face_kind"] = kind
    if kind is None:
        return dict(out, status="unsupported",
                    reason=f"face {face.get('i')} is neither planar nor a cylinder with axis data")

    owners = creating_features(face, feature_map, faces)
    if len(owners) != 1:
        return dict(out, status="ambiguous_feature" if owners else "unattributed",
                    candidates=owners,
                    reason=("several features created a face at this face's position" if owners else
                            "no feature_map entry created a face on this surface (or, on a "
                            "surface shared by several faces, at this face's position)"))
    feature_name = owners[0]
    out["feature"] = feature_name
    feat = next((f for f in recipe.get("features") or [] if f.get("name") == feature_name), None)
    if feat is None:
        return dict(out, status="unattributed", reason=f"{feature_name} is not in the recipe")
    sketch_name = feat.get("profile_sketch")
    if not sketch_name:
        return dict(out, status="no_profile_sketch", feature_dimensions=feat.get("dimensions", []),
                    reason=f"{feature_name} ({feat.get('type')}) has no profile sketch")
    out["sketch"] = sketch_name

    sketch = read_sketch(sketch_name)
    frame = Frame(((sketch.get("plane") or {}).get("frame")) or {})
    if not frame.ok():
        return dict(out, status="unsupported", reason=f"{sketch_name} reported no sketch frame")
    segments = (sketch.get("sketch") or {}).get("segments") or []
    dims = sketch.get("dimensions") or []

    if kind == "plane":
        n_f = _unit(_v(face["normal"]))
        if _norm(_cross(n_f, frame.n)) <= TOL:                    # cap face, parallel to sketch
            if abs(_dot(frame.n, _sub(_v(face["point"]), frame.o))) <= TOL:
                return dict(out, status="not_dimension_driven", role="sketch_plane",
                            reason="the face lies in the sketch plane — positioned by the plane")
            depth = feat.get("dimensions") or []
            out["role"] = "extrude_cap"
            if len(depth) == 1:
                return dict(out, status="dimensioned", dimension=dict(depth[0], type="depth"))
            return dict(out, status="multiple_dimensions" if depth else "undimensioned",
                        candidates=depth, reason=f"{feature_name} carries {len(depth)} dimensions")
        if abs(_dot(n_f, frame.n)) > TOL:
            return dict(out, status="unsupported",
                        reason="face is neither parallel nor perpendicular to the sketch plane")
        out["role"] = "side_wall"
    else:
        if _norm(_cross(_unit(_v(face["axis"])), frame.n)) > TOL:
            return dict(out, status="unsupported", reason="cylinder axis is not normal to the sketch")
        out["role"] = "cylinder_wall"

    seg_hits = match_segments(face, kind, segments, frame)
    if len(seg_hits) != 1:
        return dict(out, status="ambiguous_segment" if seg_hits else "no_segment",
                    candidates=seg_hits,
                    reason=f"{len(seg_hits)} segments of {sketch_name} sweep into this face")
    idx = seg_hits[0]
    seg = segments[idx]
    seg_kind = _segment_kind(seg)
    out["segment"] = dict(seg, index=idx)

    attached = [d for d in dims if _attached_to(d, idx, seg_kind)]
    driving = [d for d in attached if d.get("type") in _DRIVING_TYPES[seg_kind]]
    position = [d for d in dims if any(p.get("center_of") == idx for p in d.get("points") or [])]
    if position:
        out["position_dimensions"] = position
    other = [d for d in attached if d not in driving]
    if other:
        out["other_dimensions"] = other
    if len(driving) == 1:
        return dict(out, status="dimensioned", dimension=driving[0])
    if driving:
        return dict(out, status="multiple_dimensions", candidates=driving,
                    reason=f"{len(driving)} dimensions are attached to segment {idx}")
    return dict(out, status="undimensioned", add_dimension=add_dimension_args(seg),
                reason=f"no driving dimension is attached to segment {idx} of {sketch_name}")
