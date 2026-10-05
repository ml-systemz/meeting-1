# tools

Profiling and tracing for the meeting. Use them in this order.

## 1. Before anything else

```sh
python3 tools/env_check.py
```

Reports driver, GPU, compute capability, toolchain, profilers and torch. It is
free to run and it catches the failures that otherwise cost GPU time — a
missing profiler, restricted counters, a torch built against the wrong CUDA.

Anything marked `BAD` will waste your session. Fix it first.

## 2. Capture a timeline

```sh
tools/profile.sh --label baseline -- python baseline/decode.py
```

Runs the command under Nsight Systems, writes `traces/baseline.nsys-rep`,
appends the run to `traces/runs.tsv`, and prints the gap summary automatically.

If `nsys` is missing — common on rented and containerised boxes, which often do
not grant it the permissions it wants — the command still runs, untraced.
Instrument it from inside Python instead:

```python
from tools.torchprof import trace

for _ in range(5):
    step()                       # warm up outside the trace

with trace("traces/baseline.json"):
    for _ in range(20):
        step()
```

The torch profiler needs no special permissions and `trace_stats.py` reads its
output just as happily.

## 3. Read the timeline

```sh
python3 tools/trace_stats.py traces/baseline.nsys-rep
python3 tools/trace_stats.py traces/baseline.json      # torch trace
```

Prints launch count, GPU busy time, and **gap time** — wall clock on the GPU
timeline that no kernel occupied. It also lists every kernel under 10 µs, which
is the set worth fusing.

The bubble fraction is the number to watch. Fusing kernels is an attack on
exactly that figure, and a megakernel is the limit case of the attack.

## 4. Deep dive one kernel

```sh
tools/ncu_kernel.sh --kernel 'fused_mlp' --count 3 -- python bench.py
```

Nsight Compute, on the kernel `nsys` told you to care about. It replays each
launch under instrumentation, so it is far slower than a real run — always
bound it with `--count`.

If it fails with `ERR_NVGPUCTRPERM`, the driver is restricting performance
counters. Run as root, or set `NVreg_RestrictProfilingToAdminUsers=0` and
reload the module.

## Which profiler answers which question

| Question | Tool |
| --- | --- |
| Where does the time go between kernels? | `nsys` — `profile.sh` |
| How many launches, how much of it is bubble? | `trace_stats.py` |
| Why is *this* kernel slow? | `ncu` — `ncu_kernel.sh` |

`nsys` is the timeline, `ncu` is the microscope. Reach for the microscope only
after the timeline has told you where to point it.
