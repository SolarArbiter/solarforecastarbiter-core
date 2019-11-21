"""Metrics for quantifying the (monetary) value of forecasts."""

import numpy as np


def fixed_error_cost(fx, obs, cost):
    """Fixed cost per forecast error.

        C = sum_{i=1}^n |fx_i - obs_i| * cost_i

    where cost_i is the fixed cost per forecast error (e.g. USD per MW of
    error) and C is the total cost.

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

    """

    error = np.abs(fx - obs)
    total_cost = np.sum(error * cost)
    return total_cost
