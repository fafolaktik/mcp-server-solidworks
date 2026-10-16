"""Build the cadcritic SPEC 11.1 demo bracket FULLY DIMENSIONED in SolidWorks, through the adapter.

Every contract parameter is ONE driving dimension, constructed so modify_dimension cannot tear a
profile: the base+flange L profile is lines with H/V relations and its corner fixed to the origin
(dimensions W, T, Tf, H), Lx is the mid-plane extrude depth, each pad/hole circle has a diameter
plus two origin-position dimensions, and each pad height is its own extrude depth. Model points
are converted to sketch (u, v) through the frame create_sketch echoes, and every dimension name
comes from add_dimension's result. Writes OUT.dims.json (parameter -> dimension name).

    python build_dimensioned_bracket.py OUT.SLDPRT [--hole-right-mm 8.45]
                                        [--yz-plane "Front Plane"] [--xy-plane "Top Plane"]

The plane options name the default planes whose normals are +X (YZ) and +Z (XY) in YOUR part
template — on the template this was written against those are "Front Plane" and "Top Plane"; on a
standard SolidWorks template they are "Right Plane" and "Front Plane". The script checks each
sketch's normal and stops instead of building a mis-oriented part.

Known hazard: on SOLIDWORKS 2026 SP4 (Makers) creating sketch entities through the API crashed
SolidWorks intermittently, and corner rectangles did so repeatedly — this script uses lines and
circles only. Save your work in other documents before running it.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402

mm = 1e-3
P = dict(Lx=80, W=40, T=6, Tf=6, H=50, D=8.45, S=50, Yh=25, Dp=16, Pl=2, Pr=2)   # SPEC 11.1 nominal


def rg(text):
    return json.loads(text.split("result_geometry=", 1)[1]) if "result_geometry=" in text else {}


class Sketch:
    """An open sketch plus its frame: model mm -> sketch (u, v) metres."""

    def __init__(self, plane, normal, dims):
        f = rg(server.create_sketch(plane=plane))["frame"]
        self.o, self.x, self.y = f["origin"], f["xdir"], f["ydir"]
        n = [self.x[1] * self.y[2] - self.x[2] * self.y[1], self.x[2] * self.y[0] - self.x[0] * self.y[2],
             self.x[0] * self.y[1] - self.x[1] * self.y[0]]
        if any(abs(a - b) > 1e-6 for a, b in zip(n, normal)):
            raise SystemExit(f"'{plane}' has normal {n} in this template, expected {normal}; "
                             f"pass the right plane name with --yz-plane / --xy-plane")
        self.dims = dims

    def uv(self, X, Y, Z):
        d = [X * mm - self.o[0], Y * mm - self.o[1], Z * mm - self.o[2]]
        return (sum(a * b for a, b in zip(d, self.x)), sum(a * b for a, b in zip(d, self.y)))

    def dim(self, key, model_pt, value_mm, **kw):
        u, v = self.uv(*model_pt)
        r = rg(server.add_dimension(px=u, py=v, value=value_mm * mm, **kw))
        self.dims[key] = r["dimension"]
        print(f"  {key:<14} = {value_mm:>6} mm -> {r['dimension']}  ({r['short_name']}, {r['type']})")

    def circle(self, X, Y, d):
        u, v = self.uv(X, Y, 0)
        server.add_sketch_entity("circle", cx=u, cy=v, radius=d / 2 * mm)

    def circle_dims(self, key, X, Y, d):
        # diameter: pick the circumference; position: the center point to the origin along each
        # sketch axis (which model axis that is comes from the frame).
        self.dim(f"{key}_d", (X + d / 2, Y, 0), d, label_offset_x=0.004, label_offset_y=0.004)
        u, v = self.uv(X, Y, 0)
        self.dim(f"{key}_pos_u", (X, Y, 0), abs(u) / mm, entity_type="point", to_origin=True,
                 orientation="horizontal", label_offset_y=0.01)
        self.dim(f"{key}_pos_v", (X, Y, 0), abs(v) / mm, entity_type="point", to_origin=True,
                 orientation="vertical", label_offset_x=0.01)


def feature_name(extrude_result):
    return extrude_result.split("features=[")[1].split("'")[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="output .SLDPRT path")
    ap.add_argument("--hole-right-mm", type=float, default=P["D"], help="right hole diameter (error injection)")
    ap.add_argument("--yz-plane", default="Front Plane", help="default plane whose normal is +X")
    ap.add_argument("--xy-plane", default="Top Plane", help="default plane whose normal is +Z")
    a = ap.parse_args()
    dims = {}

    for _ in range(8):
        try:
            print(server.ensure_ready())
            break
        except Exception as ex:  # noqa: BLE001 - a cold SolidWorks launch can outlast one budget
            print("ensure_ready retry:", type(ex).__name__, ex)
            time.sleep(10)
    print(server.open_new_part())

    print("L profile (base + flange) in model YZ, mid-plane extruded over X")
    sk = Sketch(a.yz_plane, [1, 0, 0], dims)
    L = [(0, 0), (P["W"], 0), (P["W"], P["T"]), (P["Tf"], P["T"]), (P["Tf"], P["H"]), (0, P["H"])]
    segs = []
    for (y1, z1), (y2, z2) in zip(L, L[1:] + L[:1]):
        (u1, v1), (u2, v2) = sk.uv(0, y1, z1), sk.uv(0, y2, z2)
        segs.append({"entity_type": "line", "x1": u1, "y1": v1, "x2": u2, "y2": v2})
    server.add_sketch_entities(json.dumps(segs))
    for s in segs:
        kind = "horizontal" if abs(s["y1"] - s["y2"]) < 1e-12 else "vertical"
        server.add_sketch_constraint(kind, px1=(s["x1"] + s["x2"]) / 2, py1=(s["y1"] + s["y2"]) / 2)
    server.add_sketch_constraint("coincident", px1=0.0, py1=0.0, entity_type1="SKETCHPOINT",
                                 entity_type2="ORIGIN")
    sk.dim("W", (0, P["W"] / 2, 0), P["W"], label_offset_y=-0.01)
    sk.dim("T", (0, P["W"], P["T"] / 2), P["T"], label_offset_x=0.01)
    sk.dim("Tf", (0, P["Tf"] / 2, P["H"]), P["Tf"], label_offset_y=0.01)
    sk.dim("H", (0, 0, P["H"] / 2), P["H"], label_offset_x=-0.01)
    dims["Lx"] = "D1@" + feature_name(server.extrude_feature(depth=P["Lx"] * mm, mid_plane=True))

    for side, sx, height in (("padL", -1, P["Pl"]), ("padR", +1, P["Pr"])):
        print(f"{side}: circle on the XY plane, extruded from z=0")
        sk = Sketch(a.xy_plane, [0, 0, 1], dims)
        X = sx * P["S"] / 2
        sk.circle(X, P["Yh"], P["Dp"])
        sk.circle_dims(side, X, P["Yh"], P["Dp"])
        dims[f"{side}_height"] = "D1@" + feature_name(server.extrude_feature(depth=(P["T"] + height) * mm))

    print(f"holes: left D={P['D']}, right D={a.hole_right_mm}")
    sk = Sketch(a.xy_plane, [0, 0, 1], dims)
    sk.circle(-P["S"] / 2, P["Yh"], P["D"])
    sk.circle(P["S"] / 2, P["Yh"], a.hole_right_mm)
    sk.circle_dims("holeL", -P["S"] / 2, P["Yh"], P["D"])
    sk.circle_dims("holeR", P["S"] / 2, P["Yh"], a.hole_right_mm)
    print(server.extrude_feature(depth=0.1, feature_type="cut", mid_plane=True)[:120])

    print(server.save_document(os.path.abspath(a.out))[:160])
    with open(os.path.splitext(a.out)[0] + ".dims.json", "w", encoding="utf-8") as fh:
        json.dump(dims, fh, indent=1)
    print(json.dumps(dims, indent=1))


if __name__ == "__main__":
    main()
