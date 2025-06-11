from testgrad.uop.ops import UOp, graph_rewrite, PatternMatcher, track_rewrites, UPat, Ops, GroupOp, resolve
from testgrad.helpers import prod, unwrap
from testgrad.shape.shapetracker import ShapeTracker, strides_for_shape
from testgrad.shape.view import View
from dataclasses import dataclass

@dataclass(frozen=True)
class Kernel:
  ast: UOp
  def __repr__(self):
    ast_rep = f"SINK{tuple(s.op for s in self.ast.src)}" if self.ast.op is Ops.SINK else repr(self.ast.op)
    return f"<Kernel {len(list(self.ast.toposort()))} {ast_rep}>"

merge_views = PatternMatcher([
  # merge adjacent views
  (UPat(Ops.VIEW, src=(UPat(Ops.VIEW, name="v1"),), name="v2"), lambda v1,v2: v1.replace(arg=v1.arg+v2.arg)),
  # replace MovementOps with VIEW
  (UPat(GroupOp.Movement, src=(UPat.var("x"),), name="mop"), lambda mop,x: x.base.view(mop.st)),
  # remove NOOP views
  (UPat.var("x").view(name="view"), lambda x,view: x if x.st is not None and view.st.contiguous and view.shape == x.shape else None),
])

# change reduceop axes and input ShapeTrackers, view gets replaced with a reshape.
# src->r->view  -->   src->view->r
def swizzle_reduceop(src:UOp, r:UOp, view:UOp):
  # don't push expands
  if view.st.size > r.st.size: return None

  # confirm the input is in order
  # TODO: replace this with a UOp that allows for nothing else then remove this
  input_st = ShapeTracker.from_shape(src.shape)
  permute = tuple(i for i in range(len(input_st.shape)) if i not in r.axis_arg)+r.axis_arg
  assert permute == tuple(range(len(permute))), f"reduce axis must already be in order, {permute} isn't"

  # append the reduce shape to each of the views
  prshape = prod(rshape:=input_st.shape[-len(r.axis_arg):])
  rstrides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+rstrides, v.offset*prshape,
                    v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in unwrap(view.st).views]

  # no reshape required with shrinking REDUCE_AXIS
  return UOp(Ops.REDUCE_AXIS, r.dtype, (src.view(input_st + ShapeTracker(tuple(nv))),),
             (r.arg[0], tuple(range(len(view.shape), len(view.shape) + len(r.axis_arg)))))

view_left = merge_views+PatternMatcher([
  # view before elementwise and buffer ops
  (UPat(Ops.VIEW, src=(UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.BIND, Ops.LOAD, Ops.STORE, Ops.VALID}, name="e"),), name="view"),
   lambda e,view: e.replace(src=tuple(s.view(view.st) for s in e.src)) if view.st.size <= e.st.size else None),
  # push a non contiguous ShapeTracker through reduceop
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), swizzle_reduceop),
])

def to_contiguous_buffer(x:UOp):
  # remove contiguous
  buf = UOp.new_buffer(x.device, prod(x.shape), x.dtype)
  return buf.store(x.src[0]).reshape(x.shape).load()

def to_view_buffer(x:UOp, v:UOp):
  # view before load
  buf = UOp.new_buffer(x.device, prod(x.shape), x.dtype)
  return buf.store(x).view(v.arg).load()

to_buffers = merge_views+PatternMatcher([
  # replace CONTIGUOUS with a store to a buffer
  (UPat(Ops.CONTIGUOUS, name="x"), to_contiguous_buffer),
  # VIEW on LOAD moves before LOAD
  (UPat(Ops.VIEW, src=(UPat(Ops.LOAD, name="l"),), name="v"), lambda v,l: l.replace(src=(l.src[0].view(v.arg),)+l.src[1:])),
  # VIEW not on BUFFER or CONST needs to be a buffer
  # TODO: why is DEVICE here?
  (UPat(Ops.VIEW, src=(UPat(GroupOp.All - {Ops.BUFFER, Ops.CONST, Ops.VIEW, Ops.DEVICE, Ops.STORE} - GroupOp.Movement, name="x"),),
        name="v"), to_view_buffer),
])

def do_kernelize(x:UOp):
  srcs = []
  def gate(y:UOp):
    if y.op is Ops.STORE:
      srcs.append(y)
      return False
    return True
  srcs.append(x.src[0])
  x.src[1].toposort(gate)

  # get bufs_replace
  bufs_replace = {}
  for ty in srcs:
    # TODO: lil helper in UOp for this
    while ty.op is Ops.STORE: ty = ty.src[0]
    assert ty.op is Ops.BUFFER
    bufs_replace[ty] = UOp(Ops.DEFINE_GLOBAL, ty.dtype, arg=len(bufs_replace))

  return x.src[0].store(UOp(Ops.KERNEL, src=tuple(srcs), arg=Kernel(x.substitute(bufs_replace))))

kernelize = PatternMatcher([
  # kernels come from STORE
  (UPat(Ops.STORE, src=(UPat(), UPat(GroupOp.All - {Ops.KERNEL})), name="x"), do_kernelize),
])

@track_rewrites()
def get_kernelize_map(sink:UOp) -> dict[UOp, UOp]:
  # NOTE: might need to insert some contiguous if there's reduces that would fork
  sink = graph_rewrite(sink, view_left, name="move views")
  sink = graph_rewrite(sink, to_buffers, name="add buffers")
  sink = graph_rewrite(sink, kernelize, name="create kernels")
  graph_rewrite(sink, PatternMatcher([]), name="output")
  return {}
