"""Standard-normal helpers built from :mod:`keras.ops` primitives.

The flow needs Φ, Φ⁻¹, and log φ for a standard normal. Rather than
depend on a backend-specific SciPy shim, we build them directly from
``keras.ops.erf`` / ``keras.ops.erfinv`` so the same code runs on any
Keras backend.
"""

from __future__ import annotations

import math

from keras import ops

SQRT2 = math.sqrt(2.0)
LOG_2PI = math.log(2.0 * math.pi)


def norm_cdf(x):
    """Standard-normal CDF Φ(x)."""
    return 0.5 * (1.0 + ops.erf(x / SQRT2))


def norm_icdf(p):
    """Standard-normal quantile Φ⁻¹(p).

    ``p`` should lie in (0, 1); caller is responsible for clamping.
    """
    return SQRT2 * ops.erfinv(2.0 * p - 1.0)


def norm_log_pdf(x):
    """Standard-normal log pdf log φ(x)."""
    return -0.5 * (x * x + LOG_2PI)
