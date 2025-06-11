#!/usr/bin/env python3
import pathlib

FILES = ["dtype.py", "helpers.py", "gradient.py", "device.py",
         "__init__.py",
         # "tensor.py",
         "renderer/__init__.py", "runtime/ops_cpu.py",
         "nn/__init__.py", "nn/optim.py", "nn/state.py", "nn/datasets.py",
         "uop/__init__.py", "uop/ops.py", "uop/mathtraits.py", "uop/upat.py", "uop/spec.py",
         # "shape/shapetracker.py",
         "shape/view.py"]
src = pathlib.Path("../tinygrad/tinygrad")
dest = pathlib.Path("testgrad")

for f in FILES:
  print("importing", f)
  rd = open(src/f).read()
  rd = rd.replace("from tinygrad.", "from testgrad.")
  rd = rd.replace("import tinygrad.", "import testgrad.")
  (dest/f).parent.mkdir(parents=True, exist_ok=True)
  with open(dest/f, "w") as f:
    f.write(rd)
