from typing import Optional, cast, Generator
from dataclasses import dataclass, replace, field
import time, pprint
from testgrad.uop.ops import Variable, UPat, Ops, PatternMatcher, sym_infer, UOp, track_rewrites
from testgrad.renderer import Estimates, ProgramSpec, Renderer
from testgrad.engine.schedule import ScheduleItem
from testgrad.helpers import colored, GlobalCounters, DEBUG, time_to_str, TRACEMETA, ansilen, all_same, Metadata, BEAM, NOOPT, DEVECTORIZE
from testgrad.device import Buffer, Device
from testgrad.codegen import full_rewrite

prg_cnt = 0
@track_rewrites(named=True)
def get_program(renderer:Renderer, ast:UOp) -> ProgramSpec:
  global prg_cnt
  linearized = full_rewrite(ast, renderer)
  prg_cnt += 1
  return ProgramSpec(ast.arg.name, renderer.render(linearized), renderer.device, ast, linearized)

# **************** Runners ****************

class Runner:
  def __init__(self, display_name:str, device:str, estimates=Estimates()):
    self.first_run, self.display_name, self.device, self.estimates = True, display_name, device, estimates
  @property
  def dev(self): return Device[self.device]
  def exec(self, rawbufs:list[Buffer], var_vals:Optional[dict[Variable, int]]=None) -> Optional[float]:
    return self(rawbufs, {} if var_vals is None else var_vals)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[Variable, int], wait=False) -> Optional[float]:
    raise NotImplementedError("override this")

class CompiledRunner(Runner):
  def __init__(self, p:ProgramSpec, precompiled:Optional[bytes]=None, prg=None):
    if DEBUG >= 4: print(p.src)
    self.p:ProgramSpec = p
    self.lib:bytes = precompiled if precompiled is not None else Device[p.device].compiler.compile_cached(p.src)
    if DEBUG >= 7: Device[p.device].compiler.disassemble(self.lib)
    self._prg = Device[p.device].runtime(p.function_name, self.lib) if prg is None else prg
    super().__init__(p.name, p.device, p.estimates)

  def __reduce__(self): return self.__class__, (self.p, self.lib)

  def __call__(self, rawbufs:list[Buffer], var_vals:dict[Variable, int], wait=False) -> Optional[float]:
    global_size, local_size = self.p.launch_dims(var_vals)
    lra = {}
    if global_size:
      lra['global_size'] = tuple(global_size)
      assert len(global_size) == 3, "global size must have len 3"
    if local_size:
      lra['local_size'] = tuple(local_size)
      assert len(local_size) == 3, "local size must have len 3"
    return self._prg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k] for k in self.p.vars), wait=wait)

class BufferCopy(Runner):
  def __init__(self, total_sz, dest_device, src_device):
    if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    super().__init__(colored(name, "yellow"), dest_device, Estimates(lds=total_sz, mem=total_sz))
  def copy(self, dest, src):
    disk_supports_fast_copyout = src.device.startswith("DISK") and hasattr(src.allocator.dev, 'io_uring') and \
      getattr(src.allocator.dev, 'fd', None) is not None
    if src.device.startswith("DISK") and hasattr(dest.allocator, 'copy_from_disk') and disk_supports_fast_copyout and src.nbytes >= 4096:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif src.device.startswith("DISK") and hasattr(dest.allocator, '_as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
    else:
      dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[Variable, int], wait=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    if wait:
      Device[dest.device].synchronize()
      return time.perf_counter() - st

class BufferXfer(BufferCopy):
  def copy(self, dest, src): dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev)

# **************** method cache ****************

method_cache: dict[tuple[str, bytes, tuple[int, ...], bool], CompiledRunner] = {}
def get_runner(device:str, ast:UOp) -> CompiledRunner:
  # TODO: this should be all context relevant to rendering
  context = (BEAM.value, NOOPT.value, DEVECTORIZE.value)
  ckey = (device, ast.key, context, False)
  if cret:=method_cache.get(ckey): return cret
  bkey = (device.split(":")[0], ast.key, context, True)
  if bret:=method_cache.get(bkey):
    method_cache[ckey] = ret = CompiledRunner(replace(bret.p, device=device), bret.lib)
  else:
    prg: ProgramSpec = get_program(Device[device].renderer, ast)
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
  return ret

# **************** lowering functions ****************

@dataclass(frozen=True)
class ExecItem:
  prg: Runner
  bufs: list[Optional[Buffer]]
  metadata: Optional[tuple[Metadata, ...]] = None
  fixedvars: dict[Variable, int] = field(default_factory=dict)
  def run(self, _var_vals:Optional[dict[Variable, int]]=None, wait=False, jit=False, do_update_stats=True) -> Optional[float]:
    var_vals = self.fixedvars if _var_vals is None else (_var_vals|self.fixedvars)
    bufs = [cast(Buffer, x) for x in self.bufs] if jit else [cast(Buffer, x).ensure_allocated() for x in self.bufs]
    et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_est:=sym_infer(self.prg.estimates.ops, var_vals))
      GlobalCounters.global_mem += (mem_est:=sym_infer(self.prg.estimates.mem, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        lds_est = sym_infer(self.prg.estimates.lds, var_vals)
        mem_est = min(mem_est, lds_est)   # there can't be more memory accessed than loads/stores. remove this when symbolic is fixed
        ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
        print(f"{colored(f'*** {self.prg.device[:7]:7s} {GlobalCounters.kernel_count:4d}', 'magenta' if jit else ('green' if self.prg.first_run else None))} {self.prg.display_name+' '*(41-ansilen(self.prg.display_name))} arg {len(bufs):2d} mem {GlobalCounters.mem_used/1e9:5.2f} GB " +  # noqa: E501
              (str() if et is None else f"tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_est/((et or 1e-20)*1e9):9.2f} GFLOPS {mem_est/((et or 1e-20)*1e9):6.1f}|{lds_est/((et or 1e-20)*1e9):<7.1f} GB/s)" +  # noqa: E501
               f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in self.metadata] if self.metadata else ''}"))
      self.prg.first_run = False
    return et

si_lowerer = PatternMatcher([
  (UPat(Ops.SINK, name="sink"), lambda ctx,sink: (runner:=get_runner(ctx[0].device, sink), [ctx[x] for x in runner.p.globals])),
  (UPat(Ops.COPY, name="copy"), lambda ctx,copy: ((BufferXfer(ctx[0].nbytes, ctx[0].device, ctx[1].device) \
      if hasattr(Device[ctx[0].device].allocator, '_transfer') and all_same([x.device.split(":")[0] for x in ctx]) \
      else BufferCopy(ctx[0].nbytes, ctx[0].device, ctx[1].device)), list(ctx))),
])
def lower_schedule_item(si:ScheduleItem) -> ExecItem:
  return ExecItem(*cast(tuple[Runner,list], si_lowerer.rewrite(si.ast, si.bufs)), si.metadata, si.fixedvars)

def lower_schedule(schedule:list[ScheduleItem]) -> Generator[tuple[ScheduleItem, ExecItem], None, None]:
  while len(schedule):
    si = schedule.pop(0)
    try: yield (si, lower_schedule_item(si))
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {si.ast.op}")
        print("tensor operations:")
        pprint.pprint(si.metadata, indent=2)
      raise e

def run_schedule(schedule:list[ScheduleItem], var_vals:dict[Variable, int]|None=None, do_update_stats=True):
  for si, ei in lower_schedule(schedule):
    ei.run(var_vals, do_update_stats=do_update_stats)
