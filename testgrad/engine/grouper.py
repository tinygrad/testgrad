from testgrad.uop.ops import UOp, graph_rewrite, PatternMatcher, track_rewrites, UPat, Ops, GroupOp, graph_rewrite_map, _substitute, KernelInfo
from testgrad.uop.ops import resolve, identity_element
from testgrad.helpers import prod, unwrap, pluralize, merge_dicts, dedup, colored
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
  # view after COPY unless it's a shrink
  (UPat(Ops.COPY, src=(UPat(Ops.VIEW, name="v"), UPat(name="d")), name="c"),
   lambda v,c,d: v.src[0].copy_to_device(d).view(v.arg) if v.src[0].size <= v.size else None),
  # remove NOOP views
  (UPat.var("x").view(name="view"), lambda x,view: x if x.st is not None and view.st.contiguous and view.shape == x.shape else None),
])

# change reduceop axes and input ShapeTrackers, view gets replaced with a reshape.
# src->r->view  -->   src->view->r
def swizzle_reduceop(src:UOp, r:UOp, view:UOp):
  # don't push expands unless this is a const reduce
  if view.st.size > r.st.size and any([x.op not in {Ops.VIEW, Ops.CONST, Ops.DEVICE} for x in src.toposort()]): return None

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
  (UPat(Ops.VIEW, src=(UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.BIND, Ops.VALID}, name="e"),), name="view"),
   lambda e,view: e.replace(src=tuple(s.view(view.st) for s in e.src)) if
    (view.st.size <= e.st.size or not any([x.op is Ops.BUFFER for x in view.toposort()])) and \
    (e.op not in GroupOp.UnsafePad or all([x.mask is None for x in view.arg.views])) else None),
  # push a non contiguous ShapeTracker through reduceop
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), swizzle_reduceop),
])

early_rules = PatternMatcher([
  # remove STOREs that don't target a BUFFER or another STORE
  (UPat(Ops.STORE, src=(UPat(GroupOp.All-{Ops.BUFFER, Ops.STORE}), UPat.var('x'))), lambda x: x),
  # remove DETACH
  (UPat(Ops.DETACH, name="x"), lambda x: x.src[0]),
  # UOp with size 0 is zero
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
    and not (root.base.op is Ops.CONST and root.base.arg == 0) else None),
])

kernel_fixup = PatternMatcher([
  # always put view before load
  (UPat(Ops.VIEW, src=(UPat.var("x").load(),), name="v"), lambda x,v: x.view(v.arg).load()),
  # store doesn't need a load
  (UPat(Ops.STORE, src=(UPat(Ops.LOAD, src=(UPat.var("buf"),)), UPat.var("val"))), lambda val, buf: buf.store(val)),
])

def do_kernelize(x:UOp):
  const_replace = {}
  view_replace = {}
  unbound_dicts = []
  srcs = []
  def gate(y:UOp):
    if y.op in {Ops.STORE, Ops.BUFFER, Ops.BUFFER_VIEW}:
      srcs.append(y)
      return False
    if y.op is Ops.VIEW:
      unbound_view, unbound_dict = y.arg.unbind()
      if unbound_view != y.arg:
        unbound_dicts.append(unbound_dict)
        view_replace[y] = y.replace(arg=unbound_view)
    # TODO: should this be CONST(VIEW(DEVICE)) and not just CONST(DEVICE)?
    # yea, CONST should just have shape () and this should be removed
    if y.op is Ops.CONST and len(y.src): const_replace[y] = y.replace(src=()).view(ShapeTracker.from_shape(y.shape))
    return True
  srcs.append(x.src[0])
  x.src[1].toposort(gate)
  srcs = dedup(srcs)  # 0 should always stay at 0

  bufs_replace = {}
  for i,y in enumerate(srcs):
    dg = UOp(Ops.DEFINE_GLOBAL, y.dtype.ptr(y.buffer.size), arg=len(bufs_replace))
    assert y not in bufs_replace
    bufs_replace[y] = dg.view(ShapeTracker.from_shape((y.buffer.size,))).load() if i != 0 else dg.load()
  bufs_replace.update(const_replace)
  bufs_replace.update(view_replace)
  if len(unbound_dicts):
    for k,v in merge_dicts(unbound_dicts).items(): srcs.append(k.bind(v))

  # this is a normal kernel
  if len(srcs) == 2 and srcs[0].device != srcs[1].device:
    kast = UOp(Ops.COPY)
  else:
    kast = graph_rewrite(x, _substitute+merge_views+kernel_fixup, ctx=bufs_replace, name="fixup kernel", bottom_up=True)
    rr = sorted(dedup([x.shape for x in kast.toposort() if x.st is not None]))
    dims = [colored(x, 'blue') for x in rr[0] if resolve(x != 1)]
    dims += [colored(x, 'red') for x in rr[-1][len(dims):] if resolve(x != 1)]
    info = KernelInfo(name='k_'+colored('_', 'BLACK').join(dims))
    kast = kast.sink(arg=info)
  return x.src[0].store(UOp(Ops.KERNEL, src=tuple(srcs), arg=Kernel(kast)))

kernelize = PatternMatcher([
  # kernels come from STORE
  (UPat(Ops.STORE, src=(UPat(), UPat(GroupOp.All - {Ops.KERNEL})), name="x"), do_kernelize),
])

def to_buffer_view(v:UOp):
  if len(v.arg.views) > 1 or not ShapeTracker.from_shape(v.shape, v.arg.views[0].strides).contiguous: return None
  bv = UOp(Ops.BUFFER_VIEW, v.dtype, (v.src[0],), (v.arg.size, v.arg.views[0].offset)) \
    if v.arg.size != v.src[0].size or v.arg.views[0].offset != 0 else v.src[0]
  return bv.reshape(v.shape)

add_gbarrier = merge_views+PatternMatcher([
  # add GBARRIER on all SINK bases (with a tag)
  #(UPat(Ops.SINK, name="s"), lambda s:
  #  ns if (ns:=s.replace(src=tuple([x.base.gbarrier().view(x.st) if x.base.op is not Ops.GBARRIER else x for x in s.src]))) is not s else None),
  # replace CONTIGUOUS with GBARRIER (should be done in tensor.py?)
  (UPat(Ops.CONTIGUOUS, name="x"), lambda x: x.src[0].gbarrier()),
  # add GBARRIER before VIEW
  (UPat(Ops.VIEW, src=(UPat(GroupOp.All - {Ops.BUFFER, Ops.CONST, Ops.VIEW, Ops.DEVICE, Ops.STORE, Ops.GBARRIER, Ops.BUFFER_VIEW, Ops.LOAD} - GroupOp.Movement,
                            name="x"),), name="v"), lambda x,v: v.replace(src=(x.gbarrier(),))),
  # add GBARRIER before COPY, unless it's a BUFFER or GBARRIER
  (UPat(Ops.COPY, src=(UPat(GroupOp.All - {Ops.BUFFER, Ops.GBARRIER, Ops.BUFFER_VIEW}, name="x"), UPat(Ops.DEVICE, name="d")), name="v"),
    lambda x,v,d: v.replace(src=(x.gbarrier(), d))),
  # add GBARRIER after COPY (and tag the COPY to not repeat)
  (UPat(Ops.COPY, name="x"), lambda x: x.replace(tag=1).gbarrier() if x.tag is None else None),
  # delete GBARRIERs on GBARRIERs or BUFFERs
  (UPat(Ops.GBARRIER, src=(UPat((Ops.GBARRIER, Ops.BUFFER, Ops.LOAD), name="x"),)), lambda x: x),
  # some GBARRIERs can be BUFFER_VIEW or just RESHAPE
  (UPat(Ops.GBARRIER, src=(UPat(Ops.VIEW, src=(UPat((Ops.BUFFER, Ops.GBARRIER)),), name="v"),)), to_buffer_view),
  # force realize anything in the context
  (UPat(GroupOp.All, name="x"), lambda ctx,x: x.replace(tag=1).gbarrier() if x in ctx and x.tag is None else None),
])

gbarrier_to_buffer = merge_views+PatternMatcher([
  (UPat(Ops.GBARRIER, name="x"), lambda x: UOp.new_buffer(x.device, prod(x.shape), x.dtype).store(x.src[0]).reshape(x.shape)),
  # remove tags from COPY
  #(UPat(GroupOp.All, name="x"), lambda x: x.replace(tag=None) if x.tag is not None else None),
  # NOTE: if tags are removed, the replace loops forever
])

@track_rewrites(name_fxn=lambda big_sink,ret: f"Schedule {pluralize('Kernel',len([u for u in ret[big_sink].toposort() if u.op is Ops.KERNEL]))}")
def get_kernelize_map(sink:UOp) -> dict[UOp, UOp]:
  # NOTE: might need to insert some contiguous if there's reduces that would fork
  tensor_map = graph_rewrite_map(sink, merge_views+early_rules, name="merge views")
  tensor_map = graph_rewrite_map(tensor_map[sink], view_left, input_map=tensor_map, name="views left")
  # force realize bases
  force_realize = set([x.base for x in tensor_map[sink].src if x.base.op is not Ops.CONST])
  tensor_map = graph_rewrite_map(tensor_map[sink], add_gbarrier, ctx=force_realize, input_map=tensor_map, bottom_up=True, name="add gbarriers")
  tensor_map = graph_rewrite_map(tensor_map[sink], gbarrier_to_buffer, input_map=tensor_map, name="gbarrier to buffers")
  tensor_map = graph_rewrite_map(tensor_map[sink], kernelize, input_map=tensor_map, name="create kernels")

  graph_rewrite(tensor_map[sink], PatternMatcher([]), name="output")
  return tensor_map
