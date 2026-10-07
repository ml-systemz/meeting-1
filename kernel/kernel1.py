# kernel 1 - rmsnorm

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def rmsnorm(x, weight, eps=1e-6):
    raise NotImplementedError
