"""Your attention kernel. Edit `body` only; the scaffolding is already correct.

The reference builds the full seq x seq score matrix. At S=8192 that is 17 GB
of HBM traffic that never had to exist. Flash attention never materialises it:
it walks k/v in tiles and carries a running softmax across them.

Three values are carried across tiles, per query row:

    m   running max of the logits seen so far
    l   running sum of exp(logit - m)
    acc running sum of exp(logit - m) * v

When tile j arrives with logits `qk`, the max may increase to m_new. Everything
accumulated under the old max is off by exp(m_old - m_new), so rescale both
l and acc by that factor before adding the new tile's contribution.

At the end, acc / l is the answer.

    python3 bench.py --check-only    # fast, correctness only
    python3 bench.py                 # S=2048
    python3 bench.py --big           # S=8192, the number that counts
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def _flash_kernel(q_ref, k_ref, v_ref, o_ref, *, sm_scale, block_q, block_k):
    kv_len = k_ref.shape[0]
    head_dim = q_ref.shape[-1]
    q = q_ref[...]

    def body(j, carry):
        acc, m_prev, l_prev = carry
        sl = pl.dslice(j * block_k, block_k)
        k = k_ref[sl, :]
        v = v_ref[sl, :]
        qk = plt.dot(q, k.T) * sm_scale

        raise NotImplementedError(
            "fill in body(): update m, l and acc from qk and v, then return them"
        )

    acc = jnp.zeros((block_q, head_dim), jnp.float32)
    m0 = jnp.full((block_q,), -jnp.inf, jnp.float32)
    l0 = jnp.zeros((block_q,), jnp.float32)
    acc, _, l = jax.lax.fori_loop(0, kv_len // block_k, body, (acc, m0, l0))
    o_ref[...] = (acc / l[:, None]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("block_q", "block_k"))
def attention(q, k, v, block_q=128, block_k=128):
    """Tiled attention. Tune block_q / block_k once body() is correct."""
    b, s, h, d = q.shape
    kernel = functools.partial(
        _flash_kernel, sm_scale=1.0 / (d ** 0.5), block_q=block_q, block_k=block_k
    )
    return pl.pallas_call(
        kernel,
        grid=(s // block_q, b, h),
        in_specs=[
            pl.BlockSpec((None, block_q, None, d), lambda i, j, k_: (j, i, k_, 0)),
            pl.BlockSpec((None, s, None, d), lambda i, j, k_: (j, 0, k_, 0)),
            pl.BlockSpec((None, s, None, d), lambda i, j, k_: (j, 0, k_, 0)),
        ],
        out_specs=pl.BlockSpec((None, block_q, None, d), lambda i, j, k_: (j, i, k_, 0)),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=plt.CompilerParams(num_warps=8, num_stages=2),
        name="flash_attention",
    )(q, k, v)
