from distrax._src.distributions.normal import Normal
from jax.nn import log_sigmoid, logsumexp, log_softmax
import jax.numpy as jnp
from einops import repeat
from flowjax._src.ops.iterative_inversion import bisection_inverse


def gaussian_mixture_transform(
    inputs, logit_weights, means, log_scales, inverse=False, eps=1e-5, max_iters=100
):

    log_weights = log_softmax(logit_weights, axis=-1)

    dist = Normal(loc=means, scale=jnp.exp(log_scales))

    def mix_cdf(x):
        x = jnp.expand_dims(x, axis=-1)
        return jnp.sum(jnp.exp(log_weights) * dist.cdf(x), axis=-1)

    def mix_log_pdf(x):
        x = jnp.expand_dims(x, axis=-1)
        return logsumexp(log_weights + dist.log_prob(x), axis=-1)

    if inverse:
        max_scales = jnp.sum(jnp.exp(log_scales), axis=-1, keepdims=True)
        init_lower, _ = jnp.min(means - 20 * max_scales, axis=-1)
        init_upper, _ = jnp.min(means + 20 * max_scales, axis=-1)

        x = bisection_inverse(
            fn=mix_cdf,
            z=inputs,
            init_x=jnp.zeros_like(inputs),
            init_lower=init_lower,
            init_upper=init_upper,
            eps=eps,
            max_iters=max_iters,
        )
        ldj = mix_log_pdf(inputs)

        return x, ldj

    else:
        z = mix_cdf(inputs)
        ldj = mix_log_pdf(inputs)
        return z, ldj
