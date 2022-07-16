import jax.numpy as jnp
from jax.lax import conv_general_dilated


def conv1x1_transform(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """1x1 Convolution function
    This function will perform a 1x1 convolutions given an input
    array and a kernel. The output will be the same size as the input
    array.

    Parameters
    ----------
    x: Array
        input array for convolutions of shape
        (n_samples, height, width, n_channels)
    kernel: Array
        input kernel of shape (n_channels, n_channels)

    Returns
    -------
    output: Array
        the output array after the convolutions of shape
        (n_samples, height, width, n_channels)

    References
    ----------
    * https://hackerstreak.com/1x1-convolution/
    * https://iamaaditya.github.io/2016/03/one-by-one-convolution/
    * https://sebastianraschka.com/faq/docs/fc-to-conv.html
    """
    return conv_general_dilated(
        lhs=x,
        rhs=kernel[..., None, None],
        window_strides=(1, 1),
        padding="SAME",
        lhs_dilation=(1, 1),
        rhs_dilation=(1, 1),
        dimension_numbers=("NHWC", "IOHW", "NHWC"),
    )
