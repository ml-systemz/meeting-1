# meeting-1 — flash attention from the inside

## The objective

Write the inner loop of a flash attention kernel in Pallas, on an H100.

Everything around it is already done: the `pallas_call`, the grid, the block
specs, the backend flags. You write the **online softmax recurrence** — the
four lines FlashAttention is actually famous for. They live in `body()` in
`kernel.py`.

```sh
python3 bench.py --check-only   # correctness only, fast
python3 bench.py                # S=2048
python3 bench.py --big          # S=8192, the number that counts
```

## Why this and not something else

The naive implementation builds the full `seq x seq` score matrix. At S=8192
that is **17 GB** of HBM traffic for a result that only needs a few hundred MB.
Flash attention never materialises it: walk k/v in tiles, carry a running
softmax, rescale when the running max moves.

That one idea is worth about **2x**, measured on the hardware below.

## The ladder

Measured on one H100 80GB HBM3, CUDA 12.9, JAX 0.11.1, bf16,
B=4 S=8192 H=16 D=128, clocks locked:

| | time | vs naive | % of bf16 peak |
| --- | ---: | ---: | ---: |
| naive (materialises scores) | 12,529 µs | 1.00x | 17.7% |
| **what you are writing** | ~6,700 µs | ~1.87x | ~33% |
| Pallas library `mha` | 6,317 µs | 1.98x | 35.2% |
| cuDNN flash | 3,830 µs | 3.27x | 58.1% |
| FLOP floor | 2,223 µs | 5.63x | 100% |

**You will not beat cuDNN, and that is the interesting part.** A correct,
reasonable tiled kernel lands around a third of peak. cuDNN gets 58%. The
distance between those two numbers is warp specialisation, TMA, and WGMMA
pipelining — which is the conversation this session exists to start.

There is no leaderboard. The target is physics, and everyone can see how far
off it they are.

## What this is not

Not a megakernel. We checked: `jax.jit` on a fused MLP block already reaches
**92–111% of its theoretical floor**, so a hand-written MLP kernel loses to the
compiler. Attention is the place where hand-written tiling still wins, because
the quadratic intermediate is something XLA will not restructure away.

## Setup

See [`help/setup.md`](help/setup.md). One command creates a node with
everything preinstalled — nobody installs anything.

## Files

```
kernel.py     edit this, and only this
bench.py      correctness + timing. do not edit
solution.py   a working answer. open it when you want to, not before
help/         setup and the two Pallas traps that will cost you an hour
```
