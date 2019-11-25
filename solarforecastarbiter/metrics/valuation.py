"""Metrics for quantifying the (monetary) value of forecasts."""

import numpy as np


def fixed_error_cost(fx, obs, cost):
    """Fixed cost per forecast error.

        total_cost = sum_{i=1}^n |fx_i - obs_i| * cost

    where cost is the fixed cost per forecast error (e.g. USD per MW of error)
    and total_cost is cost of the entire forecast time-series.

    Parameters
    ----------
    fx : (n,) array_like
        Forecasts.
    obs : (n,) array_like
        Observations.
    cost : float
        The fixed cost per error (e.g. USD per MW).

    Returns
    -------
    total_cost : float
        Total cost (e.g. USD).

    Examples
    --------

    Forecast power [kW], cost: 10 USD per kW
    >>> fx = np.array([1, 2, 3])
    >>> obs = np.array([1, 3, 4])
    >>> cost = 10  # 10 USD per kW
    >>> fixed_error_cost(fx, obs, cost)
    20

    """

    error = np.abs(fx - obs)
    total_cost = np.sum(error * cost)
    return total_cost
