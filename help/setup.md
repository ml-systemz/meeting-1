# Setup

One command. Nobody installs anything.

```sh
autor node create --chip h100-1 --name <your-name>-attn \
  --image jax-0.11-cuda12.9 --clock-lock --wait
```

`--clock-lock` pins the GPU clocks so your timings are reproducible. It is
free and there is no reason not to use it when you are benchmarking.

**Rate: $0.066/min.** A 90-minute session is about $6.

## Stop your node when you are done

```sh
autor node stop <your-name>-attn -y
```

Billing runs while the node is up, whether or not anything is executing. The
disk is kept, so you can come back to it.

## What is on the image

Verified on 2026-10-07, not quoted from docs:

| | |
| --- | --- |
| GPU | H100 80GB HBM3, compute capability **9.0** |
| Driver | 595.91.07 |
| CUDA | 12.9 (nvcc 12.9.86) |
| Python | 3.12.3 |
| JAX / jaxlib | **0.11.1** |
| Pallas | `jax.experimental.pallas` and `.mosaic_gpu` both import clean |
| `ncu` | present at `/usr/local/cuda/bin/ncu` |
| `nsys` | **not installed** |

Your home is `/home/dev` and you are user `dev`, not root. Writing to `/root`
fails with permission denied.

Reference points measured on this exact node: a 4096³ bf16 matmul runs at
**678 TFLOP/s**, about 69% of the 989 TFLOP/s dense peak.

## Profiling

`ncu` is installed but its counters are locked unless the node was created
with `--profiling`, and **that flag only works on 8x shapes**. If you want
Nsight Compute, you need `--chip h100-8 --profiling`, which costs
proportionally more.

For this session you do not need it. `bench.py` reports the only numbers that
matter.

## Getting files on and off

```sh
autor cp ./kernel.py <node>:/home/dev/meeting-1/kernel.py
autor cp <node>:/home/dev/meeting-1/kernel.py ./
autor run <node> -- bash -lc 'cd /home/dev/meeting-1 && python3 bench.py'
```
