# meeting-1

A harness for the first ML systems reading group session, so the time goes into
kernels rather than plumbing.


## Our Refrences 

1. [Jax's Offical Pallas Docs](https://docs.jax.dev/en/latest/pallas/gpu/index.html)
2. 

## Quick start

```sh
python3 tools/env_check.py                                   # is this box usable?
tools/profile.sh --label baseline -- python baseline/decode.py
python3 tools/trace_stats.py traces/baseline.nsys-rep        # where did the time go?
```

## Layout

```
tools/      profiling and tracing — see tools/README.md
traces/     captured traces and runs.tsv (gitignored)
```

## The number that matters

`trace_stats.py` reports a **bubble fraction**: the share of the GPU timeline
that no kernel occupied. At batch size 1 a decode step issues dozens of small
kernels, each paying launch overhead, with idle gaps between them. That gap is
the thing a megakernel exists to remove.

Measure it before you optimise anything.

## Notes on the hardware

WGMMA and TMA are SM90 and newer. They do not exist on consumer Ampere or Ada,
so a laptop GPU cannot run the interesting paths — `env_check.py` will say so.

Nsight Systems and Nsight Compute both want permissions that rented boxes
frequently withhold. `tools/torchprof.py` is the fallback that always works.
