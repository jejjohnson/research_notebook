from jax.scipy.linalg import cholesky, cho_factor, cho_solve


def solve(P, Q):
    """
    Compute P^-1 Q, where P is a PSD matrix, using the Cholesky factorisation
    """
    L = cho_factor(P, lower=True)
    return cho_solve(L, Q)
