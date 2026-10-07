"""Check your kernel against the reference, then time it.

    python3 bench.py            # small, fast — use this while debugging
    python3 bench.py --big      # the real shape, for the number that counts
"""

import argparse
import time

import jax
import jax.numpy as jnp

import kernel

PEAK_BF16 = 989e12
TOL = 2e-2

TARGETS_BIG = {"pallas reference": 6317e-6, "cudnn flash": 3830e-6}
TARGETS_SMALL = {"pallas reference": 474e-6, "cudnn flash": 284e-6}


def reference(q, k, v):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) * scale
    probs = jax.nn.softmax(scores, axis=-1).astype(q.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", probs, v)


def make_inputs(b, s, h, d):
    keys = jax.random.split(jax.random.key(0), 3)
    shape = (b, s, h, d)
    return tuple(jax.random.normal(k, shape, jnp.bfloat16) for k in keys)


def bench(fn, *args, n=20):
    out = fn(*args)
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="S=8192 instead of S=2048")
    ap.add_argument("--check-only", action="store_true", help="skip timing")
    args = ap.parse_args()

    b, h, d = 4, 16, 128
    s = 8192 if args.big else 2048
    targets = TARGETS_BIG if args.big else TARGETS_SMALL

    q, k, v = make_inputs(b, s, h, d)
    print(f"\n  B={b} S={s} H={h} D={d}  bfloat16  on {jax.devices()[0].device_kind}\n")

    want = reference(q, k, v)
    try:
        got = jax.jit(kernel.attention)(q, k, v)
    except Exception as exc:
        print(f"  kernel raised {type(exc).__name__}: {exc}")
        return 1

    if got.shape != want.shape:
        print(f"  WRONG SHAPE  got {got.shape}, want {want.shape}")
        return 1

    err = jnp.abs(got.astype(jnp.float32) - want.astype(jnp.float32))
    worst = float(err.max())
    if worst > TOL:
        bad = jnp.unravel_index(jnp.argmax(err), err.shape)
        print(f"  INCORRECT    max abs err {worst:.3e} > tol {TOL:.0e}")
        print(f"               worst at index {tuple(int(i) for i in bad)}")
        print(f"               got {float(got[bad]):+.5f}  want {float(want[bad]):+.5f}")
        print("\n  Fix correctness before looking at speed.\n")
        return 1

    print(f"  correct      max abs err {worst:.3e}  (tol {TOL:.0e})\n")
    if args.check_only:
        return 0

    flops = 4 * b * h * s * s * d
    floor = flops / PEAK_BF16
    t_ref = bench(jax.jit(reference), q, k, v)
    t_you = bench(jax.jit(kernel.attention), q, k, v)

    rows = [("naive", t_ref), ("yours", t_you), (None, None)]
    rows += [(name, t) for name, t in targets.items()]
    rows += [("flop floor", floor)]

    for name, t in rows:
        if name is None:
            print("  " + "-" * 46)
            continue
        mark = " <-" if name == "yours" else ""
        print(f"  {name:<18}{t * 1e6:>9.0f} us{t_ref / t:>8.2f}x{floor / t * 100:>8.1f}% of peak{mark}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
