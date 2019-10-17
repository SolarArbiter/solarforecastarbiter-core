"""Deterministic forecast error metrics."""

import numpy as np

__all__ = [
    "mean_absolute",
    "mean_bias",
    "root_mean_square",
    "mean_absolute_percentage",
    "normalized_root_mean_square",
    "forecast_skill",
    "pearson_correlation_coeff",
    "coeff_determination",
    "centered_root_mean_square",
    "kolmogorov_smirnov_integral",
    "over",
    "combined_performance_index",
]


def mean_absolute(y_true, y_pred):
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


def mean_bias(y_true, y_pred):
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


def root_mean_square(y_true, y_pred):
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


def mean_absolute_percentage(y_true, y_pred):
    """Mean absolute percentage error (MAPE).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    mape : float
        The MAPE [%] between the true and predicted values.

    """

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0


def normalized_root_mean_square(y_true, y_pred, y_norm):
    """Normalized root mean square error (NRMSE).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    y_norm : float
        Normalized factor, in the same units as y_true and y_pred.

    Returns
    -------
    nrmse : float
        The NRMSE [%] between the true and predicted values.

    """

    return root_mean_square(y_true, y_pred) / y_norm * 100.0


def forecast_skill(y_true, y_pred, y_ref):
    """Forecast skill (s).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    y_ref: array_like
        Reference forecast values.

    Returns
    -------
    s : float
        The forecast skill [-] between the true and predicted values compared
        to a reference forecast.

    """

    rmse_pred = root_mean_square(y_true, y_pred)
    rmse_ref = root_mean_square(y_true, y_ref)
    return 1.0 - rmse_pred / rmse_ref


def pearson_correlation_coeff(y_true, y_pred):
    """Pearson correlation coefficient (r).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    r : float
        The correlation coefficient (r [-]) between the true and predicted
        values.

    """

    x1 = y_pred - np.mean(y_pred)
    x2 = y_true - np.mean(y_true)
    return np.sum(x1 * x2) / (
        np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2))
    )


def coeff_determination(y_true, y_pred):
    """Coefficient of determination (R^2).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    r2 : float
        The coefficient of determination (R^2 [-]) between the true and
        predicted values.

    """

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def centered_root_mean_square(y_true, y_pred):
    """Centered (unbiased) root mean square error (CRMSE):

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    crmse : float
        The CRMSE between the true and predicted values.

    """

    return np.sqrt(np.mean(
        ((y_pred - np.mean(y_pred)) - (y_true - np.mean(y_true))) ** 2
    ))


def kolmogorov_smirnov_integral():
    """Kolmogorov-Smirnov Test Integral (KSI).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    ksi : float
        The KSI between the true and predicted values.

    """
    return None

def over():
    """OVER metric.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    over : float
        The OVER metric between the true and predicted values.

    """
    return None


def combined_performance_index():
    """Combined Performance Index (CPI) metric.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    cpi : float
        The CPI between the true and predicted values.

    """
    return None
