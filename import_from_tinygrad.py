#!/usr/bin/env python3
import pathlib

FILES = ["dtype.py", "helpers.py", "gradient.py",
         # tinygrad hardcoded
         #"device.py",
         "__init__.py", "tensor.py",
         # remove CPU graph
         #"runtime/ops_cpu.py",
         "runtime/ops_python.py",
         # split libc import line
         # "runtime/support/elf.py"
         "renderer/__init__.py", "renderer/cstyle.py", "runtime/autogen/libc.py",
         "nn/__init__.py", "nn/optim.py", "nn/state.py", "nn/datasets.py",
         "uop/__init__.py",  "uop/mathtraits.py", "uop/upat.py",
         # changing reduce here
         #"uop/ops.py",
         "uop/spec.py", "uop/symbolic.py", "uop/transcendental.py",
         # changing reduce function here
         #"shape/shapetracker.py",
         "shape/view.py",
         "viz/serve.py", "viz/index.html", "viz/js/index.js", "viz/js/worker.js",
]
src = pathlib.Path("../tinygrad/tinygrad")
dest = pathlib.Path("testgrad")

for f in FILES:
  print("importing", f)
  rd = open(src/f).read()
  rd = rd.replace("from tinygrad ", "from testgrad ")
  rd = rd.replace("from tinygrad.", "from testgrad.")
  rd = rd.replace("import tinygrad.", "import testgrad.")
  (dest/f).parent.mkdir(parents=True, exist_ok=True)
  with open(dest/f, "w") as f:
    f.write(rd)
