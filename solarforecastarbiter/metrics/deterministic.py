"""Deterministic forecast error metrics."""

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

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


def mean_absolute(obs, fx):
    """Mean absolute error (MAE).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    mae : float
        The MAE of the forecast.

    """

    return np.mean(np.abs(fx - obs))


def mean_bias(obs, fx):
    """Mean bias error (MBE).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    mbe : float
        The MBE of the forecast.

    """

    return np.mean(fx - obs)


def root_mean_square(obs, fx):
    """Root mean square error (RMSE).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    rmse : float
        The RMSE of the forecast.

    """

    return np.sqrt(np.mean((fx - obs) ** 2))


def mean_absolute_percentage(obs, fx):
    """Mean absolute percentage error (MAPE).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    mape : float
        The MAPE [%] of the forecast.

    """

    return np.mean(np.abs((obs - fx) / obs)) * 100.0


def normalized_root_mean_square(obs, fx, norm):
    """Normalized root mean square error (NRMSE).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    norm : float
        Normalized factor, in the same units as obs and fx.

    Returns
    -------
    nrmse : float
        The NRMSE [%] of the forecast.

    """

    return root_mean_square(obs, fx) / norm * 100.0


def forecast_skill(obs, fx, ref):
    """Forecast skill (s).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    ref : (n,) array_like
        A reference forecast.

    Returns
    -------
    s : float
        The forecast skill [-] of the forecast relative to a reference
        forecast.

    """

    rmse_fx = root_mean_square(obs, fx)
    rmse_ref = root_mean_square(obs, ref)
    return 1.0 - rmse_fx / rmse_ref


def pearson_correlation_coeff(obs, fx):
    """Pearson correlation coefficient (r).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    r : float
        The correlation coefficient (r [-]) of the observations and forecasts.

    """

    x1 = fx - np.mean(fx)
    x2 = obs - np.mean(obs)
    return np.sum(x1 * x2) / (
        np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2))
    )


def coeff_determination(obs, fx):
    """Coefficient of determination (R^2).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    r2 : float
        The coefficient of determination (R^2 [-]) of the observations and
        forecasts.

    """

    ss_res = np.sum((obs - fx) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - ss_res / ss_tot


def centered_root_mean_square(obs, fx):
    """Centered (unbiased) root mean square error (CRMSE):

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    crmse : float
        The CRMSE of the forecast.

    """

    return np.sqrt(np.mean(
        ((fx - np.mean(fx)) - (obs - np.mean(obs))) ** 2
    ))


def kolmogorov_smirnov_integral(obs, fx, normed=False):
    """Kolmogorov-Smirnov Test Integral (KSI).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    normed : bool, optional
        If True, return the normalized KSI [%].

    Returns
    -------
    ksi : float
        The KSI of the forecast.

    Notes
    -----
    The calculation of the empirical CDF uses a right endpoint rule (the
    default of the statsmodels ECDF function). For example, if the data is
    [1.0, 2.0], then ECDF output is 0.5 for any input less than 1.0.

    """

    # empirical CDF
    ecdf_obs = ECDF(obs)
    ecdf_fx = ECDF(fx)

    # evaluate CDFs
    x = np.unique(np.concatenate((obs, fx)))
    y_o = ecdf_obs(x)
    y_f = ecdf_fx(x)

    # compute metric
    D = np.abs(y_o - y_f)
    ksi = np.sum(D[:-1] * np.diff(x))

    if normed:
        Vc = 1.63 / np.sqrt(len(obs))
        a_critical = Vc * (x.max() - x.min())
        return ksi / a_critical * 100.0
    else:
        return ksi


def over(obs, fx):
    """The OVER metric.

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    over : float
        The OVER metric of the forecast.

    Notes
    -----
    The calculation of the empirical CDF uses a right endpoint rule (the
    default of the statsmodels ECDF function). For example, if the data is
    [1.0, 2.0], then ECDF output is 0.5 for any input less than 1.0.

    """

    # empirical CDF
    ecdf_obs = ECDF(obs)
    ecdf_fx = ECDF(fx)

    # evaluate CDFs
    x = np.unique(np.concatenate((obs, fx)))
    y_o = ecdf_obs(x)
    y_f = ecdf_fx(x)

    # compute metric
    D = np.abs(y_o - y_f)
    Vc = 1.63 / np.sqrt(len(obs))
    Dstar = D - Vc
    Dstar[D <= Vc] = 0.0
    over = np.sum(Dstar[:-1] * np.diff(x))
    return over


def combined_performance_index(obs, fx):
    """Combined Performance Index (CPI) metric.

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.

    Returns
    -------
    cpi : float
        The CPI between the true and predicted values.

    """
    ksi = kolmogorov_smirnov_integral(obs, fx)
    ov = over(obs, fx)
    rmse = root_mean_square(obs, fx)
    cpi = 1 / 4 * (ksi + ov + 2 * rmse)
    return cpi
