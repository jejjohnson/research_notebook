def get_loss_fn(args):
    
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    if args.loss == "mse":
        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def make_step(model, x, y):
            pred_y = jax.vmap(model)(x)
            # Trains with respect to binary cross-entropy
            return jnp.mean((pred_y - y)**2)

        @eqx.filter_jit
        def val_step(model, x, y):
            pred_y = jax.vmap(model)(x)
            # Trains with respect to binary cross-entropy
            return jnp.mean((pred_y - y)**2)

    else:
        raise ValueError(f"Unrecognized loss function: {args.loss}")
    return make_step, val_step
