import os
from branching_wire_builder import BranchingCableBuilder

tcl_path = r"../TestData/A20016147_6150-37-520-5295/cable.tcl"

builder = BranchingCableBuilder(tcl_path)
mesh = builder.build()

out_dir = "test_output/splices"
os.makedirs(out_dir, exist_ok=True)
mesh.export(os.path.join(out_dir, "A20016147.glb"))
print("Done!")
