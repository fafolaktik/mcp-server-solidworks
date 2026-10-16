"""The closed design-contract loop, live, through the adapter:
check_contract -> failing residual's face_index -> dimension_for_face -> modify_dimension -> repeat.

Starts from a part built by build_dimensioned_bracket.py, injects one error (the right hole
diameter, via the dimension named in its .dims.json), saves that as a COPY, and loops until the
contract passes. Actuation policy is deliberately narrow — never a guess: act only when the
requirement measures the same quantity the dimension drives (hole_diameter on a diameter
dimension), setting it to the middle of the contract range. Everything else is reported.

    python contract_loop.py PART.SLDPRT CONTRACT.json COPY.SLDPRT [--wrong-hole-mm 9.0]

Reads and dimension edits only — no sketch entity is created.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402

UNIT_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}
ACTUATORS = {"hole_diameter": ("diameter",)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", help="dimensioned bracket .SLDPRT (with its .dims.json beside it)")
    ap.add_argument("contract", help="cadcritic contract JSON")
    ap.add_argument("copy", help="where to save the wrong-hole copy (.SLDPRT)")
    ap.add_argument("--wrong-hole-mm", type=float, default=9.0)
    ap.add_argument("--max-iterations", type=int, default=6)
    a = ap.parse_args()

    with open(os.path.splitext(a.part)[0] + ".dims.json", encoding="utf-8") as fh:
        built = json.load(fh)
    # names were captured before the first save; re-qualify for this document
    dim, sketch = built["holeR_d"].split("@")[:2]
    doc = os.path.splitext(os.path.basename(a.part))[0]
    hole_r = f"{dim}@{sketch}@{doc}.Part"

    print(server.ensure_ready())
    print(server.open_document(os.path.abspath(a.part))[:90])
    print("introduce the error:", server.modify_dimension(hole_r, a.wrong_hole_mm / 1000).split("result_geometry=")[1])
    print(server.save_document(os.path.abspath(a.copy))[:120])

    for it in range(a.max_iterations):
        rep = json.loads(server.check_contract(os.path.abspath(a.contract)))
        s = rep["summary"]
        fm = rep["face_mapping"]
        print(f"\n=== iteration {it}: {s['pass']}/{s['total']} pass   score [error_count, violation_score] = {rep['score']}")
        print(f"  face_mapping: matched {fm['matched']}/{fm['features']}  unmatched={fm['unmatched']}  multiple={fm['multiple']}")
        if rep["passed"]:
            print("CONTRACT PASSES — loop converged")
            break
        acted = False
        to_m = UNIT_M[rep["units"]["length"]]
        for req in rep["requirements"]:
            if req["status"] == "pass":
                continue
            print(f"  FAIL {req['id']} ({req['type']}): measured={req['measured']} {req['unit']} "
                  f"range={req['range']} residual={req['residual']} {req['unit']}")
            if not req["features"]:
                print("    global requirement (no feature) — not actuated")
                continue
            for desc in req["features"]:
                fi = desc["face_index"]
                print(f"    feature {desc['id']} -> face_index={fi} ({desc['face_match']['status']})")
                if fi is None:
                    continue
                dff = json.loads(server.dimension_for_face(fi))
                d = dff.get("dimension") or {}
                print(f"    dimension_for_face({fi}) -> {dff['status']}: {dff.get('feature')} / {dff.get('sketch')} "
                      f"segment {(dff.get('segment') or {}).get('index')} -> {d.get('name')} = {d.get('value_si')} ({d.get('type')})")
                if dff["status"] != "dimensioned" or d.get("type") not in ACTUATORS.get(req["type"], ()):
                    print("    no matching actuator for this requirement/dimension pair — not actuated")
                    continue
                lo, hi = req["range"]
                m = server.modify_dimension(d["name"], (lo + hi) / 2 * to_m)
                print(f"    modify_dimension -> {m.split('result_geometry=')[1]}")
                acted = True
        if not acted:
            print("nothing actuatable left — stopping")
            break
    print(server.save_document()[:120])


if __name__ == "__main__":
    main()
