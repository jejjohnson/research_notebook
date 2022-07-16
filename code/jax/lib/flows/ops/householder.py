import jax.numpy as jnp


def householder_product(vectors: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        vectors [K,D] - q vectors for the reflections

    Returns:
        R [D, D] - householder reflections
    """
    num_reflections, num_dimensions = vectors.shape

    squared_norms = jnp.sum(vectors**2, axis=-1)

    # initialize reflection
    H = jnp.eye(num_dimensions)

    for vector, squared_norm in zip(vectors, squared_norms):
        temp = H @ vector  # Inner product.
        temp = jnp.outer(temp, (2.0 / squared_norm) * vector)  # Outer product.
        H = H - temp

    return H


# def householder_product(inputs: jnp.ndarray, q_vector: jnp.ndarray) -> jnp.ndarray:
#     """
#     Args:
#         inputs (Array) : inputs for the householder product
#         (D,)
#         q_vector (Array): vector to be multiplied
#         (D,)

#     Returns:
#         outputs (Array) : outputs after the householder product
#     """
#     # norm for q_vector
#     squared_norm = jnp.sum(q_vector ** 2)
#     # inner product
#     temp = jnp.dot(inputs, q_vector)
#     # outer product
#     temp = jnp.outer(temp, (2.0 / squared_norm) * q_vector).squeeze()
#     # update
#     output = inputs - temp
#     return output


# def _householder_product_body(carry: jnp.array, inputs: jnp.array) -> Tuple[jnp.array, int]:
#     """Helper function for the scan product"""
#     return householder_product(carry, inputs), 0


# def householder_transform(inputs: jnp.array, vectors: jnp.array) -> jnp.array:
#     """
#     Args:
#         inputs (Array) : inputs for the householder product
#             (D,)
#         q_vector (Array): vectors to be multiplied in the
#             (D,K)

#     Returns:
#         outputs (Array) : outputs after the householder product
#             (D,)
#     """
#     return jax.lax.scan(_householder_product_body, inputs, vectors)[0]


# def householder_inverse_transform(inputs: jnp.array, vectors: jnp.array) -> jnp.array:
#     """
#     Args:
#         inputs (Array) : inputs for the householder product
#             (D,)
#         q_vector (Array): vectors to be multiplied in the reverse order
#             (D,K)

#     Returns:
#         outputs (Array) : outputs after the householder product
#             (D,)
#     """
#     return jax.lax.scan(_householder_product_body, inputs, vectors[::-1])[0]
