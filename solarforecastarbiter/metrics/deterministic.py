"""Deterministic metrics."""

import numpy as np


def _mean_absolute(y_true, y_pred):
    """Mean absolute error (MAE).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    mae : float
        The MAE between the true and predicted values.
    """

    return np.mean(np.abs(y_true - y_pred))


def _mean_bias(y_true, y_pred):
    """Mean bias error (MBE).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    mbe : float
        The MBE between the true and predicted values.

    """

    return np.mean(y_pred - y_true)


def _root_mean_square(y_true, y_pred):
    """Root mean square error (RMSE).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    rmse : float
        The RMSE between the true and predicted values.

    """

    return np.sqrt(np.mean((y_true - y_pred) ** 2))
