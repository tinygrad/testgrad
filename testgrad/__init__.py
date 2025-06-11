import os
if int(os.getenv("TYPED", "0")):
  from typeguard import install_import_hook
  install_import_hook(__name__)
from testgrad.tensor import Tensor                                    # noqa: F401
from testgrad.engine.jit import TinyJit                               # noqa: F401
from testgrad.uop.ops import UOp
Variable = UOp.variable
from testgrad.dtype import dtypes                                     # noqa: F401
from testgrad.helpers import GlobalCounters, fetch, Context, getenv   # noqa: F401
from testgrad.device import Device                                    # noqa: F401
