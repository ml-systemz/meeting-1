# kernel 3 - attention

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def attention(q, k, v):
    raise NotImplementedError
