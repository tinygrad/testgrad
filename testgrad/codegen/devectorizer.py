from testgrad.uop.ops import UOp, Ops

# ugh, these should be generic helpers maybe

def no_vectorized_alu(alu:UOp):
  if alu.dtype.vcount == 1: return None
  alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg) for i in range(alu.dtype.vcount))
  return UOp(Ops.VECTORIZE, alu.dtype, alus)
