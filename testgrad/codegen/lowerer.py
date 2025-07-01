from dataclasses import dataclass, field
from testgrad.dtype import dtypes
from testgrad.uop.ops import PatternMatcher, UPat, Ops, UOp, graph_rewrite

@dataclass
class LowererContext:
  current_range: list[UOp]=field(default_factory=list)
  range_number: int = 0

def add_store_indexing(ctx:LowererContext, store:UOp, buf:UOp, view:UOp):
  assert store.src[1].shape == view.shape, f"shape mismatch on store {store.src[1].shape} != {view.shape}"
  # create the output range
  ctx.current_range = [UOp.range(dtypes.int, s, i) for i,s in enumerate(view.st.shape)]
  ctx.range_number = len(ctx.current_range)
  idx, valid = view.st.to_indexed_uops(ctx.current_range)
  return store.replace(src=(buf.index(idx, valid),)+store.src[1:])

def add_reduce_indexing(ctx:LowererContext, red:UOp):
  more_shape = red.src[0].shape[len(ctx.current_range):]
  reduce_range = [UOp.range(dtypes.int, s, ctx.range_number+i) for i,s in enumerate(more_shape)]
  lc = LowererContext(ctx.current_range+reduce_range, ctx.range_number+len(more_shape))
  from testgrad.codegen.lowerer import pm_lowerer  # TODO: better way to do this?
  ret = graph_rewrite(red.src[0], pm_lowerer, lc, name="subreduce", bottom_up=True)
  ctx.range_number = lc.range_number
  return ret.reduce(*reduce_range, arg=red.arg[0])

def view_const(ctx:LowererContext, view:UOp, c:UOp):
  if all(x.mask is None for x in view.arg.views): return c
  _, valid = view.arg.to_indexed_uops(ctx.current_range)
  return valid.where(c, c.const_like(0))

def view_buffer(ctx:LowererContext, view:UOp, buf:UOp):
  idx, valid = view.arg.to_indexed_uops(ctx.current_range)
  return buf.index(idx, valid).load()

pm_lowerer = PatternMatcher([
  (UPat(Ops.STORE, src=(UPat(Ops.DEFINE_GLOBAL, name="buf").view(name="view"), UPat()), name="store"), add_store_indexing),
  (UPat(Ops.REDUCE_AXIS, name="red"), add_reduce_indexing),
  (UPat(Ops.VIEW, src=(UPat.cvar("c"),), name="view"), view_const),
  (UPat(Ops.VIEW, src=(UPat(Ops.DEFINE_GLOBAL, name="buf"),), name="view").load(), view_buffer),
])
