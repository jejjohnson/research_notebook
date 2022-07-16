def sum_except_batch(x, num_dims=1):
    """
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def mean_except_batch(x, num_dims=1):
    """
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)
