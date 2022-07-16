from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp

Array = jnp.ndarray


class ReLU(eqx.Module):
    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        return jax.nn.relu(x)


class GeLU(eqx.Module):
    """GELU activation (https://arxiv.org/abs/1606.08415)
    as used in Sparse Transformers (https://arxiv.org/abs/1904.10509)."""

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        return jax.nn.gelu(x)


class ConcatelU(eqx.Module):
    """Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead."""

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        return jax.nn.relu(jnp.concatenate([x, -x], axis=0))


class SoftPlus(eqx.Module):
    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        return jax.nn.softplus(x)


class Tanh(eqx.Module):
    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        return jax.nn.tanh(x)


class GatedTanh(eqx.Module):
    dim: float = eqx.static_field()

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        x_tanh, x_sigmoid = jnp.array_split(x, 2, axis=self.dim)
        return jax.nn.tanh(x_tanh) * jax.nn.sigmoid(x_sigmoid)


class Swish(eqx.Module):
    """Swish activation (https://arxiv.org/abs/1710.05941)."""

    beta: float = eqx.static_field()

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        return x * jax.nn.sigmoid(self.beta * x)


class Sine(eqx.Module):
    """Sine Activation Function"""

    w0: Array = eqx.static_field()

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def __call__(
        self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        return jnp.sin(self.w0 * x)


def get_activation(activation: str = "relu", **kwargs):
    if activation == "identity":
        return eqx.nn.Identity()
    elif activation == "relu":
        return ReLU()
    elif activation == "tanh":
        return Tanh()
    elif activation == "softplus":
        return SoftPlus()
    elif activation == "swish":
        return Swish(beta=kwargs.get("beta", 1.0))
    elif activation == "sine":
        return Sine(w0=kwargs.get("w0", 1.0))
    else:
        raise ValueError(f"Unrecognized activation: {activation}")
