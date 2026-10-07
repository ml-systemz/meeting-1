import argparse
import importlib
import sys
import time

import jax
import jax.numpy as jnp

PEAK_FLOPS = 989e12
PEAK_BW = 3.35e12
TOL = 2e-2


def timeit(fn, *args, n=20):
    out = fn(*args)
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / n


def problem(which, a):
    if which == 1:
        m, n = a.m, a.n
        k0, k1 = jax.random.split(jax.random.key(0))
        x = jax.random.normal(k0, (m, n), jnp.bfloat16)
        w = jax.random.normal(k1, (n,), jnp.bfloat16)

        def ref(x, w, eps=1e-6):
            f = x.astype(jnp.float32)
            return (f * jax.lax.rsqrt(jnp.mean(f * f, -1, keepdims=True) + eps)).astype(x.dtype) * w

        bytes_moved = 2 * m * n * 2 + n * 2
        return (x, w), ref, 0.0, bytes_moved, f"M={m} N={n}"

    if which == 2:
        m, k, n = a.m, a.k, a.n
        k0, k1 = jax.random.split(jax.random.key(0))
        x = jax.random.normal(k0, (m, k), jnp.bfloat16)
        y = jax.random.normal(k1, (k, n), jnp.bfloat16)
        ref = lambda x, y: x @ y
        return (x, y), ref, 2 * m * k * n, (m * k + k * n + m * n) * 2, f"M={m} K={k} N={n}"

    b, s, h, d = a.b, a.s, a.h, a.d
    ks = jax.random.split(jax.random.key(0), 3)
    q, k, v = (jax.random.normal(z, (b, s, h, d), jnp.bfloat16) for z in ks)

    def ref(q, k, v):
        scale = 1.0 / (d ** 0.5)
        sc = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) * scale
        return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(sc, -1).astype(q.dtype), v)

    return (q, k, v), ref, 4 * b * h * s * s * d, 4 * b * s * h * d * 2, f"B={b} S={s} H={h} D={d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("which", type=int, choices=(1, 2, 3))
    p.add_argument("--m", type=int, default=8192)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--b", type=int, default=4)
    p.add_argument("--s", type=int, default=4096)
    p.add_argument("--h", type=int, default=16)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--kw", action="append", default=[],
                   help="extra kernel argument, key=value, repeatable")
    p.add_argument("--iters", type=int, default=20)
    a = p.parse_args()

    sys.path.insert(0, "../kernel")
    sys.path.insert(0, "kernel")
    mod = importlib.import_module(f"kernel{a.which}")
    fn = {1: "rmsnorm", 2: "matmul", 3: "attention"}[a.which]
    kernel = getattr(mod, fn)

    kw = {}
    for item in a.kw:
        key, _, val = item.partition("=")
        try:
            kw[key] = int(val)
        except ValueError:
            kw[key] = val

    args, ref, flops, bytes_moved, label = problem(a.which, a)
    extra = f"  {kw}" if kw else ""
    print(f"\n  kernel {a.which}: {fn}   {label}{extra}")
    print(f"  {jax.devices()[0].device_kind}\n")

    try:
        got = kernel(*args, **kw)
        jax.block_until_ready(got)
    except NotImplementedError:
        print("  not written yet\n")
        return 1
    except Exception as exc:
        print(f"  {type(exc).__name__}: {str(exc)[:300]}\n")
        return 1

    try:
        want = jax.jit(ref)(*args)
        jax.block_until_ready(want)
    except Exception:
        want = None

    if want is None:
        t_you = timeit(lambda *z: kernel(*z, **kw), *args, n=a.iters)
        floor = max(flops / PEAK_FLOPS if flops else 0.0, bytes_moved / PEAK_BW)
        print("  the reference cannot run at this size - it runs out of memory.")
        print("  your kernel is the only thing here that produces an answer.\n")
        print(f"  yours        {t_you*1e6:>9.1f} us")
        print(f"  floor        {floor*1e6:>9.1f} us")
        if flops:
            print(f"\n  you          {flops/t_you/1e12:>9.1f} TFLOP/s   "
                  f"{floor/t_you*100:.1f}% of roofline")
        print()
        return 0

    if got.shape != want.shape:
        print(f"  WRONG SHAPE  got {got.shape}, want {want.shape}\n")
        return 1

    g32, w32 = got.astype(jnp.float32), want.astype(jnp.float32)
    scale = float(jnp.maximum(jnp.abs(w32).max(), 1e-6))
    err = float((jnp.abs(g32 - w32) / scale).max())
    if err > TOL:
        bad = jnp.unravel_index(jnp.argmax(jnp.abs(g32 - w32)), g32.shape)
        idx = tuple(int(i) for i in bad)
        print(f"  WRONG   rel err {err:.2e} > {TOL:.0e}")
        print(f"          worst at {idx}: got {float(g32[bad]):+.5f} want {float(w32[bad]):+.5f}\n")
        return 1
    print(f"  correct   rel err {err:.2e}")

    t_ref = timeit(jax.jit(ref), *args, n=a.iters)
    t_you = timeit(lambda *z: kernel(*z, **kw), *args, n=a.iters)

    compute_floor = flops / PEAK_FLOPS if flops else 0.0
    memory_floor = bytes_moved / PEAK_BW
    floor = max(compute_floor, memory_floor)
    bound = "compute" if compute_floor > memory_floor else "memory"

    print(f"\n  xla/cublas   {t_ref*1e6:>9.1f} us")
    print(f"  yours        {t_you*1e6:>9.1f} us   {t_ref/t_you:>5.2f}x")
    print(f"  floor        {floor*1e6:>9.1f} us   {bound}-bound")
    if flops:
        print(f"\n  you          {flops/t_you/1e12:>9.1f} TFLOP/s   "
              f"{floor/t_you*100:.1f}% of roofline")
    else:
        print(f"\n  you          {bytes_moved/t_you/1e12:>9.2f} TB/s     "
              f"{floor/t_you*100:.1f}% of roofline")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
