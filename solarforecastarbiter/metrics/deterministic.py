"""Deterministic forecast error metrics."""

import numpy as np
import scipy as sp
from statsmodels.distributions.empirical_distribution import ECDF


def mean_absolute(obs, fx):
    """Mean absolute error (MAE).

        MAE = 1/n sum_{i=1}^n |fx_i - obs_i|

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

        MBE = 1/n sum_{i=1}^n (fx_i - obs_i)

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

        RMSE = sqrt( 1/n sum_{i=1}^n (fx_i - obs_i)^2 )

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

        MAPE = 1/n sum_{i=1}^n |(fx_i - obs_i) / obs_i| * 100%

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


def normalized_mean_absolute(obs, fx, norm):
    """Normalized mean absolute error (NMAE).

        NMAE = MAE / norm * 100%

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
    nmae : float
        The NMAE [%] of the forecast.

    """

    return mean_absolute(obs, fx) / norm * 100.0


def normalized_mean_bias(obs, fx, norm):
    """Normalized mean bias error (NMBE).

        NMBE = MBE / norm * 100%

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
    nmbe : float
        The NMBE [%] of the forecast.

    """

    return mean_bias(obs, fx) / norm * 100.0


def normalized_root_mean_square(obs, fx, norm):
    """Normalized root mean square error (NRMSE).

        NRMSE = RMSE / norm * 100%

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

        s = 1 - RMSE_fx / RMSE_ref

    where RMSE_fx is the RMSE of the forecast and RMSE_ref is the RMSE of the
    reference forecast (e.g. Persistence).

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    ref : (n,) array_like
        Reference forecast values.

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

        r = A / (B * C)

    where:

        A = sum_{i=1}^n (fx_i - fx_avg) * (obs_i - obs_avg)
        B = sqrt( sum_{i=1}^n (fx_i - fx_avg)^2 )
        C = sqrt( sum_{i=1}^n (obs_i - obs_avg)^2 )
        fx_avg = 1/n sum_{i=1} fx_i
        obs_avg = 1/n sum_{i=1} obs_i

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

    try:
        r, _ = sp.stats.pearsonr(obs, fx)
    except ValueError as e:
        #print(e)
        #r = np.nan
        raise e

    return r


def coeff_determination(obs, fx):
    """Coefficient of determination (R^2).

        R^2 = 1 - (A / B)

    where:

        A = sum_{i=1}^n (obs_i - fx_i)^2
        B = sum_{i=1}^n (obs_i - obs_avg)^2
        obs_avg = 1/n sum_{i=1} obs_i

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

        CRMSE = sqrt( 1/n sum_{i=1}^n ((fx_i - fx_avg) - (obs_i - obs_avg))^2 )

    where:

        fx_avg = 1/n sum_{i=1} fx_i
        obs_avg = 1/n sum_{i=1} obs_i

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

        KSI = int_{p_min}^{p_max} D_n(p) dp

    where:

        D_n(p) = max(|CDF_obs(p) - CDF_fx(p)|)

    and CDF_obs and CDF_fx are the empirical CDFs of the observations and
    forecasts, respectively. KSI can be normalized as:

        KSI [%] = KSI / a_critical * 100%

    where:

        a_critical = V_c * (p_max - p_min)
        V_c = 1.63 / sqrt(n)

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

        OVER = int_{p_min}^{p_max} D_n^*(p) dp

    where:

        D_n^* = (D_n - V_c) if D_n > V_c, else D_n^* = 0

    with D_n and V_c defined the same as in KSI.

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

        CPI = (KSI + OVER + 2 * RMSE) / 4

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
    cpi = (ksi + ov + 2 * rmse) / 4.0
    return cpi


# Add new metrics to this map to map shorthand to function
_MAP = {
    'mae': mean_absolute,
    'mbe': mean_bias,
    'rmse': root_mean_square,
    'mape': mean_absolute_percentage,
    'nmae': normalized_mean_absolute,
    'nmbe': normalized_mean_bias,
    'nrmse': normalized_root_mean_square,
    's': forecast_skill,
    'r': pearson_correlation_coeff,
    'r^2': coeff_determination,
    'crmse': centered_root_mean_square,
    'ksi': kolmogorov_smirnov_integral,
    'over': over,
    'cpi': combined_performance_index,
}

__all__ = [m.__name__ for m in _MAP.values()]

# Functions that require a reference forecast
_REQ_REF_FX = ['s']

# Functions that require normalized factor
_REQ_NORM = ['nmae', 'nmbe', 'nrmse']
