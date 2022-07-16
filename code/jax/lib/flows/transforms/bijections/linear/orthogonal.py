from jax.nn.initializers import orthogonal
import treex as tx
import jax.numpy as jnp


class RandomRotation(tx.Module):
    V: jnp.ndarray

    def __call__(self, x):

        if self.initializing():

            # random initialization
            key = tx.next_key()
            self.V = orthogonal()(key=key, shape=[x.shape[1], x.shape[1]])

        return self.forward(x)

    def forward(self, x):
        return jnp.dot(x, self.V)

    def inverse(self, x):
        return jnp.dot(x, self.V.T)

    def forward_and_log_det(self, x):
        z = jnp.dot(x, self.V)
        ldj = jnp.zeros_like(x)
        return z, ldj
