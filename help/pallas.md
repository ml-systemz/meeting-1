# Pallas traps

Two of these cost an hour each if you meet them cold. Both are already handled
in `kernel.py` — this is here so you understand the scaffolding you were given,
and so you can debug it when you start changing things.

## 1. `None` squeezes a dimension; `1` does not

```python
pl.BlockSpec((1,    block_q, 1,    d), ...)   # ref is 4-D, keeps size-1 axes
pl.BlockSpec((None, block_q, None, d), ...)   # ref is 2-D [block_q, d]
```

With `1` you get a four-dimensional ref and have to index `q_ref[0, :, 0, :]`
everywhere. Worse, it changes how the block is staged, which leads directly
to trap 2.

## 2. On Hopper, Pallas defaults to Mosaic GPU, which will blow shared memory

This is the one that will stop you dead:

```
ValueError: Mosaic GPU kernel exceeds available shared memory:
smem_bytes=1116168 > max_smem_bytes=232448
```

Flash attention wants a block spec covering the **whole** k/v sequence, then
slices tiles out of it inside the loop. Mosaic GPU stages every block into
shared memory eagerly, so a full-sequence block is 1.1 MB against a 232 KB
budget, and it refuses.

The Triton backend loads lazily and is fine with it. Ask for it explicitly:

```python
from jax.experimental.pallas import triton as plt

pl.pallas_call(
    ...,
    compiler_params=plt.CompilerParams(num_warps=8, num_stages=2),
)
```

`num_warps` and `num_stages` are Triton concepts, so passing them is also how
you can tell which backend a piece of code is using. The flash attention that
ships in JAX does exactly this — it imports `jax.experimental.pallas.triton`,
not `mosaic_gpu`, despite running on Hopper.

Both are worth tuning once your kernel is correct. `num_warps` 4 or 8,
`num_stages` 2 or 3.

## Things that are useful to know

- `plt.dot(a, b)` is the tensor-core matmul inside a Triton-backend kernel.
- `pl.dslice(start, size)` makes a dynamic slice for indexing a ref in a loop.
- `jax.lax.fori_loop` carries your `(acc, m, l)` tuple across k/v tiles.
- Accumulate in **float32** even though the inputs are bf16. The running sum
  is where precision goes if you do not.
- `jax.experimental.pallas.ops.gpu.attention` is a working reference you can
  read with `inspect.getsource`. It is deprecated in favour of
  [tokamax](https://github.com/openxla/tokamax), but it still runs.

## If your numbers are wrong but plausible

Check the rescale. When the running max moves from `m_prev` to `m_curr`,
**both** `l` and `acc` have to be multiplied by `exp(m_prev - m_curr)` before
the new tile is added. Forgetting it on `acc` but not `l` gives output that
looks roughly right and is quietly wrong.

`bench.py` prints the worst index and both values when the check fails.
