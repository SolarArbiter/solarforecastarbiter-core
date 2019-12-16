"""Summary statistics (observations and forecasts)."""

import numpy as np


# - var: np.nanvar()
# - percentile: np.nanpercentile()  <== unnecessary with quantile?
# - quantile: np.nanquantile()      <== unnecessary with percentile?


def mean(ts):
    """Mean value.

    Parameters
    ----------
    ts : array_like
        Time-series of observations or forecasts.

    Returns
    -------
    avg : float
        Mean value.

    """

    return np.nanmean(ts)


def std(ts):
    return np.nanstd(ts)


def min(ts):
    return np.nanmin(ts)


def max(ts):
    return np.nanmax(ts)


def median(ts):
    return np.nanmedian(ts)


def var(ts):
    return np.nanvar(ts)


def quantile(ts, q):
    return np.nanquantile(ts, q)
