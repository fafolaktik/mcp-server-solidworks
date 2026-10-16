"""Two STEP test parts for check_contract's cadcritic-1.1 area tie-breaks (built with build123d).

A  interrupted bore: block 60x20x20 mm, D8 bore along X through y=10,z=10, cut by a slot over
   x in [-5, 15] — two COAXIAL, equal-radius hole features of different lengths (25 / 15 mm).
B  split wall: block 40x20x10 mm with ONE vertical D8 hole at (0, 10), cut as two half-cylinders
   without cleaning, so its wall is TWO cylindrical faces (one hole feature, summed area).
"""
import os
import sys

from build123d import Box, Cylinder, Pos, Rot, Align, export_step, GeomType

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
C = (Align.MIN, Align.MIN, Align.MIN)


def cyl_faces(part):
    return [f for f in part.faces() if f.geom_type == GeomType.CYLINDER]


# --- A: interrupted bore --------------------------------------------------------------------
block = Pos(-30, 0, 0) * Box(60, 20, 20, align=C)
bore = Pos(0, 10, 10) * Rot(0, 90, 0) * Cylinder(4, 80)
slot = Pos(-5, -1, 5) * Box(20, 22, 16, align=C)
part_a = block - bore - slot
print("A cylindrical faces:", [round(f.area, 4) for f in cyl_faces(part_a)])
export_step(part_a, os.path.join(OUT, "bore_interrupted.step"))

# --- B: one hole whose wall is two half-cylinder faces -------------------------------------
block = Pos(-20, 0, 0) * Box(40, 20, 10, align=C)
holed = block - Pos(0, 10, 5) * Cylinder(4, 20)
# split the finished part at x=0 and fuse the halves back WITHOUT cleaning: the internal face
# goes, the seam on the hole wall (and on the top/bottom/side faces) stays
neg = holed & (Pos(-30, -5, -5) * Box(30, 30, 20, align=C))
pos = holed & (Pos(0, -5, -5) * Box(30, 30, 20, align=C))
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse   # raw OCC fuse: no SimplifyResult / unify
from build123d import Solid
fuse = BRepAlgoAPI_Fuse(neg.solids()[0].wrapped, pos.solids()[0].wrapped)
part_b = Solid(fuse.Shape()) if fuse.Shape().ShapeType().name == "TopAbs_SOLID" else Solid.cast(fuse.Shape())
part_b = Solid(part_b.solids()[0].wrapped)
print("B cylindrical faces:", [round(f.area, 4) for f in cyl_faces(part_b)])
export_step(part_b, os.path.join(OUT, "hole_split.step"))
