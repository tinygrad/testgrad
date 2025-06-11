from testgrad.uop.ops import UOp, graph_rewrite, PatternMatcher, track_rewrites

@track_rewrites()
def get_kernelize_map(big_sink:UOp) -> dict[UOp, UOp]:
  graph_rewrite(big_sink, PatternMatcher([]))
  return {}
