"""Tiled attention in Pallas: never materialise the seq x seq score matrix."""

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
        m_curr = jnp.maximum(m_prev, jnp.max(qk, axis=-1))
        correction = jnp.exp(m_prev - m_curr)
        p = jnp.exp(qk - m_curr[:, None])
        l_curr = correction * l_prev + jnp.sum(p, axis=-1)
        acc = acc * correction[:, None] + plt.dot(p.astype(v.dtype), v)
        return acc, m_curr, l_curr

    acc = jnp.zeros((block_q, head_dim), jnp.float32)
    m0 = jnp.full((block_q,), -jnp.inf, jnp.float32)
    l0 = jnp.zeros((block_q,), jnp.float32)
    acc, _, l = jax.lax.fori_loop(0, kv_len // block_k, body, (acc, m0, l0))
    o_ref[...] = (acc / l[:, None]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("block_q", "block_k"))
def attention(q, k, v, block_q=128, block_k=128):
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
