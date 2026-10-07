# Pallas traps

Four things that cost an hour each if you meet them cold.

## `None` squeezes a dimension, `1` does not

```python
pl.BlockSpec((1,    block_m, 1,    d), ...)   # 4-D ref, keeps the size-1 axes
pl.BlockSpec((None, block_m, None, d), ...)   # 2-D ref [block_m, d]
```

With `1` you index `q_ref[0, :, 0, :]` everywhere, and it changes how the
block is staged, which leads straight to the next one.

## On Hopper, Pallas defaults to a backend that will blow shared memory

```
ValueError: Mosaic GPU kernel exceeds available shared memory:
smem_bytes=1116168 > max_smem_bytes=232448
```

Mosaic GPU stages every block into shared memory up front. A block spec
covering a whole sequence is 1.1 MB against a 232 KB budget, and it refuses.

Two ways out. Ask for the Triton backend, which loads lazily:

```python
from jax.experimental.pallas import triton as plt
pl.pallas_call(..., compiler_params=plt.CompilerParams(num_warps=8, num_stages=2))
```

Or keep Mosaic and put the big operand in global memory:

```python
pl.BlockSpec(memory_space=plgpu.GMEM)
```

The flash attention that ships in JAX takes the first route.

## No scalar indexing

```
NotImplementedError: Unimplemented primitive in Pallas Triton lowering: slice
```

`x[i]` on an array inside a kernel does not lower. Reductions do, so reach for
one instead — `jnp.exp(jnp.sum(jnp.log(a)))` in place of a cumulative product's
last element.

## Accumulate in float32

Inputs are bf16. Running sums are where precision goes. Cast on the way in,
cast back on the way out.

## Worth knowing

- `plt.dot(a, b)` is the tensor-core matmul inside a Triton-backend kernel.
- `pl.dslice(start, size)` indexes a ref dynamically inside a loop.
- `jax.lax.fori_loop` carries your accumulator across tiles.
- `num_warps` 4 or 8, `num_stages` 1 to 3. These are not cosmetic — on a
  kernel carrying a 64 KB state, 4 vs 8 warps was a 3.4x difference.
- `jax.experimental.pallas.ops.gpu.attention` is a working kernel you can read
  with `inspect.getsource`.

## Reading

- [Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html)
- [GPU reference](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)
- [Pipelining](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html)
