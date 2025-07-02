#!/usr/bin/env python3
import sys
import pathlib

if len(sys.argv) > 1:
  FILES = sys.argv[1:]
  TEST_FILES = []
else:
  FILES = ["dtype.py", "helpers.py",
          # fix reduce gradient
          "gradient.py",
          "device.py",
          "__init__.py",
          # tensor replaces assign with store
          #"tensor.py",
          "kernelize/grouper.py",
          "runtime/ops_cpu.py",
          "runtime/ops_python.py",
          "runtime/ops_npy.py",
          "runtime/ops_disk.py",
          "runtime/ops_metal.py",
          "runtime/support/elf.py",
          "renderer/__init__.py",
          "renderer/cstyle.py",
          "runtime/autogen/libc.py",
          "nn/__init__.py", "nn/optim.py", "nn/state.py", "nn/datasets.py",
          "uop/__init__.py",  "uop/mathtraits.py", "uop/upat.py",
          # changing reduce here
          #"uop/ops.py",
          # add one rule here
          "uop/symbolic.py",
          "uop/spec.py",
          "uop/transcendental.py",
          # changing reduce function here
          #"shape/shapetracker.py",
          "shape/view.py",
          "viz/serve.py", "viz/index.html", "viz/js/index.js", "viz/js/worker.js",
          # okay parts of codegen
          # lowerer changed
          #"codegen/__init__.py",
          "codegen/devectorizer.py",
          "codegen/expander.py",
          "codegen/linearize.py",
          "engine/realize.py",
  ]

  TEST_FILES = [
    "test_ops.py",
    "test_tiny.py",
    "test_arange.py",
    "test_schedule.py",
    "test_outerworld_range.py",
    # enabled flash attention
    #"test_softmax_fusion.py",
    "unit/test_disk_tensor.py",
    "unit/test_simple_schedule.py",
  ]

def move_files(FILES, src, dest):
  for f in FILES:
    print("importing", f)
    rd = open(src/f).read()
    rd = rd.replace("from tinygrad ", "from testgrad ")
    rd = rd.replace("from tinygrad.", "from testgrad.")
    rd = rd.replace("import tinygrad.", "import testgrad.")
    (dest/f).parent.mkdir(parents=True, exist_ok=True)
    with open(dest/f, "w") as f:
      f.write(rd)

move_files(FILES, pathlib.Path("../tinygrad/tinygrad"), pathlib.Path("testgrad"))
move_files(TEST_FILES, pathlib.Path("../tinygrad/test"), pathlib.Path("test"))