from typing import Callable, List, Optional
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

Array = jnp.ndarray

class Sine(eqx.Module):
    """Sine Activation Function"""
    w0: Array = eqx.static_field()

    def __init__(self, w0: float=1.0, *args, **kwargs):
        """**Arguments:**
        
        - `w0` : the weight for the activation
        """
        super().__init__()
        self.w0 = w0

    def __call__(self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None) -> Array:
        """**Arguments**
        - `x`: The input. JAX Array, shape `(in_features,)`
        """
        return jnp.sin(self.w0 * x)

    
class Siren(eqx.Module):
    """Siren Layer"""
    weight: Array
    bias: Array
    w0: Array = eqx.static_field()
    activation: eqx.Module

    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        key: Array,  
        w0: float=1., 
        c: float=6.,
        activation=None
    ):
        super().__init__()
        w_key, b_key = jrandom.split(key)
        if w0 is None:
            # First layer
            w_max = 1 / in_dim
            b_max = 1 / jnp.sqrt(in_dim)
        else:
            w_max = jnp.sqrt(c / in_dim) / w0
            b_max = 1 / jnp.sqrt(in_dim) / w0
        self.weight = jrandom.uniform(
            key, (out_dim, in_dim), minval=-w_max, maxval=w_max
        )
        self.bias = jrandom.uniform(key, (out_dim,), minval=-b_max, maxval=b_max)
        self.w0 = w0
        self.activation = Sine(w0) if activation is None else activation

    def __call__(self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None) -> Array:
        x = self.weight @ x + self.bias
        x = self.activation(x)
        return x


class SirenNet(eqx.Module):
    """SirenNet"""
    layers: List[Siren]
    num_layers: Array = eqx.static_field()
    hidden_dim: Array = eqx.static_field()
    final_scale: Array = eqx.static_field()
    final_activation: Callable[[Array], Array]

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden: int,
        key: Array,
        w0_initial: float=30,
        w0: float=1.0,
        c: float=6.0,
        final_scale: float=1.0,
        final_activation: Callable[[Array], Array] = eqx.nn.Identity()
    ):
        super().__init__()
        """"""
        keys = jrandom.split(key, n_hidden + 1)
        
        # First layer
        self.layers = [
            Siren(
                in_dim, hidden_dim, w0=w0_initial, c=c, key=keys[0], activation=None
            )
        ]
        
        # Hidden layers
        for ikey in keys[1:-1]:
            self.layers.append(
                Siren(
                    hidden_dim, hidden_dim, w0=w0, c=c, key=ikey, activation=None
                )
            )
        # Last layer
        self.layers.append(
            Siren(
                hidden_dim, out_dim, key=keys[-1], w0=w0, c=c, activation=final_activation
            )
        )

        self.num_layers = n_hidden + 1
        self.hidden_dim = hidden_dim
        self.final_scale = final_scale
        self.final_activation = final_activation

    def __call__(self, x: Array, *, key: Optional["jax.random.PRNGKey"] = None) -> Array:
        for layer in self.layers:
            x = layer(x)
        return self.final_activation(x * self.fi