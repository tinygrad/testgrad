from typing import cast
from collections import deque, defaultdict
from dataclasses import dataclass, field
from testgrad.helpers import Metadata, merge_dicts
from testgrad.device import Buffer
from testgrad.uop.ops import UOp, Variable, Ops

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[Variable, int] = field(default_factory=dict)

def create_schedule_with_vars(sched_sink:UOp):
  # construct the KERNEL children graph based on assigns
  children: defaultdict[UOp, list[UOp]] = defaultdict(list)
  in_degree: dict[UOp, int] = {}
  for u in sched_sink.toposort():
    if u.op is not Ops.STORE: continue  # anything that's not a STORE doesn't write a kernel, so we can skip
    k = u.src[1]
    in_degree.setdefault(k, 0)
    for s in k.src:
      if s.op is Ops.STORE:
        children[s.src[1]].append(k)
        in_degree[k] += 1
      elif s.op in {Ops.BUFFER, Ops.BIND, Ops.BUFFER_VIEW}:
        pass  # a BUFFER is already realized, nothing to do here
      else:
        raise RuntimeError(f"input to kernel must be STORE, BUFFER, or BIND, not {s.op}")

  # linearize KERNEL UOps into ScheduleItems in BFS order
  queue = deque(k for k,v in in_degree.items() if v == 0)
  schedule: list[ScheduleItem] = []

  bound_vars_dicts = []
  while queue:
    k = queue.popleft()
    ubufs = tuple(s.buf_uop.buffer for s in k.src if s.op is not Ops.BIND)
    bound_vars = dict([s.unbind() for s in k.src if s.op is Ops.BIND])
    if len(bound_vars): bound_vars_dicts.append(bound_vars)
    schedule.append(ScheduleItem(k.arg.ast, cast(tuple[Buffer, ...], ubufs)))
    for x in children[k]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  return schedule, merge_dicts(bound_vars_dicts)