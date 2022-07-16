import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro

numpyro.set_platform("gpu")  # "cpu"
# import funsor; funsor.set_backend("jax")


def logmatmulexp(x, y):
    x_shift = x.max(-1, keepdims=True)
    y_shift = y.max(-2, keepdims=True)
    return (
        jnp.log(jnp.exp(x - x_shift) @ jnp.exp(y - y_shift)) + x_shift + y_shift,
        None,
    )


@jax.jit
def sequential(x_init, xs):
    o, _ = jax.lax.scan(logmatmulexp, xs[0], xs[1:])
    o = logmatmulexp(jnp.expand_dims(x_init, -2), o)[0]
    return logsumexp(o.squeeze(-2), -1)


@jax.jit
def forward(x_init, xs):
    o, _ = jax.lax.scan(logmatmulexp, jnp.expand_dims(x_init, -2), xs)
    return logsumexp(o.squeeze(-2), -1)


@jax.jit
def parallel(x_init, xs):
    batch_shape = xs.shape[:-3]
    state_dim = xs.shape[-1]
    while xs.shape[-3] > 1:
        time = xs.shape[-3]
        even_time = time // 2 * 2
        even_part = xs[..., :even_time, :, :]
        a_b = even_part.reshape(batch_shape + (even_time // 2, 2, state_dim, state_dim))
        a, b = a_b[..., 0, :, :], a_b[..., 1, :, :]
        contracted = logmatmulexp(a, b)[0]
        if time > even_time:
            contracted = jnp.concatenate((contracted, xs[..., -1:, :, :]), axis=-3)
        xs = contracted
    o = logmatmulexp(jnp.expand_dims(x_init, -2), xs.squeeze(-3))[0]
    return logsumexp(o.squeeze(-2), -1)


# @jax.jit
# def funsor_scan(x_init, xs):
#     trans = funsor.Tensor(xs)["time", "prev", "curr"]
#     o = funsor.sum_product.sequential_sum_product(
#         funsor.ops.logaddexp,
#         funsor.ops.add,
#         trans,
#         funsor.Variable("time", funsor.Bint[xs.shape[0]]),
#         {"prev": "curr"})
#     o = logmatmulexp(jnp.expand_dims(x_init, -2), o.data)[0]
#     return logsumexp(o.squeeze(-2), -1)


dim = 3
x = jax.random.normal(jax.random.PRNGKey(0), (2000, dim, dim))
x_init = jax.random.normal(jax.random.PRNGKey(1), (dim,))
sequential(x_init, x)
parallel(x_init, x)
forward(x_init, x)
# funsor_scan(x_init, x)
y = sequential(x_init, x).copy()
y = parallel(x_init, x).copy()
y = forward(x_init, x).copy()
# %timeit y = funsor_scan(x_init, x).copy()
