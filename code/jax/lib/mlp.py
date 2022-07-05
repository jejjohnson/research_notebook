from typing import Callable, List, Optional
import jax
import jax.random as jrandom
import equinox as eqx
from .activations import ReLU

    
    
class MLP(eqx.Module):
    linear : eqx.Module
    activation : eqx.Module
    
    def __init__(self, in_dim, out_dim, key, activation: Optional[eqx.Module]=ReLU()):
        
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)
        self.activation = activation if not None else eqx.nn.Identity()
        
    def __call__(self, x, *, key: Optional["jax.random.PRNGKey"] = None):
        x = self.linear(x)
        x = self.activation(x)
        return x
        
    
    
class MLPNet(eqx.Module):
    layers : List[eqx.Module]
    activation: eqx.Module
    last_activation: eqx.Module
    
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        hidden_dim, 
        n_hidden, 
        key, 
        activation=ReLU(), 
        last_activation=eqx.nn.Identity()
    ):
    
        self.activation = activation
        self.last_activation = last_activation
        
        
        keys = jrandom.split(key, n_hidden + 2)
    
        # First layer
        self.layers = [eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])]
        
        # Hidden layers
        for key in keys[1:-1]:
            self.layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=key))
            
        # Output Layer
        self.layers.append(eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1]))
        
    def __call__(self, x, *, key: Optional["jax.random.PRNGKey"] = None):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            
        x = self.layers[-1](x)
        x = self.last_activation(x)
            
        return x