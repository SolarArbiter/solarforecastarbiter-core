"""Deterministic forecast error metrics."""
import datetime as dt
from functools import partial


import numpy as np
import pandas as pd
import scipy as sp
from statsmodels.distributions.empirical_distribution import ECDF


def deadband_mask(obs, fx, deadband):
    """Calculate deadband mask.

    .. math:: \\text{mask_i} = | fx_i - obs_i | <= deadband * | obs_i |

    Floating point arithmetic makes the equality difficult to guarantee
    so do not rely on it.

    Equality

    Parameters
    ----------
    obs : (n,) array_like
        Observed values.
    fx : (n,) array_like
        Forecasted values.
    deadband : float
        Fractional tolerance relative to the observed values.

    Returns
    -------
    mask : array_like
        1 if a point is within the deadband, 0 if not
    """
    return np.isclose(fx, obs, rtol=deadband)


def error(obs, fx):
    """The difference ..math:: fx - obs"""
    return fx - obs


def error_deadband(obs, fx, deadband):
    """Error fx - obs, accounting for a deadband.

    error = 0 for points where error <= deadband * obs

    error = fx - obs for points where error > deadband * obs

    Parameters
    ----------
    obs : (n,) array_like
        Observed values.
    fx : (n,) array_like
        Forecasted values.
    deadband : float
        Fractional tolerance

    Returns
    -------
    error : array_like
        The error accounting for a deadband.
    """
    error = fx - obs
    mask = deadband_mask(obs, fx, deadband)
    error = np.where(mask, 0, error)
    return error


def mean_absolute(obs, fx, error_fnc=error):
    """Mean absolute error (MAE).

    .. math:: \\text{MAE} = 1/n \\sum_{i=1}^n |\\text{fx}_i - \\text{obs}_i|

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    error_fnc : function
        A function that returns the error, default fx - obs. First
        argument is obs, second argument is fx.

    Returns
    -------
    mae : float
        The MAE of the forecast.

    Examples
    --------
    Standard MAE:
    >>> obs = np.array([1, 2, 3, 4])
    >>> fx = np.array([2, 2.04, 2, 3.96])
    >>> mean_absolute(obs, fx)
    0.52

    MAE with a deadband:
    >>> error_fnc = partial(error_deadband, deadband=0.05)
    >>> mean_absolute(obs, fx, error_fnc=error_fnc)
    0.5
    """
    error = error_fnc(obs, fx)
    return np.mean(np.abs(error))


def mean_bias(obs, fx, error_fnc=error):
    """Mean bias error (MBE).

    .. math:: \\text{MBE} = 1/n \\sum_{i=1}^n (\\text{fx}_i - \\text{obs}_i)

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    error_fnc : function
        A function that returns the error, default fx - obs. First
        argument is obs, second argument is fx.

    Returns
    -------
    mbe : float
        The MBE of the forecast.

    """
    error = error_fnc(obs, fx)
    return np.mean(error)


def root_mean_square(obs, fx, error_fnc=error):
    """Root mean square error (RMSE).

    .. math::
        \\text{RMSE} = \\sqrt{
            1/n \\sum_{i=1}^n (\\text{fx}_i - \\text{obs}_i)^2 }

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    error_fnc : function
        A function that returns the error, default fx - obs. First
        argument is obs, second argument is fx.

    Returns
    -------
    rmse : float
        The RMSE of the forecast.

    """
    error = error_fnc(obs, fx)
    return np.sqrt(np.mean(error * error))


def mean_absolute_percentage(obs, fx, error_fnc=error):
    """Mean absolute percentage error (MAPE).

    .. math::
        \\text{MAPE} = 1/n \\sum_{i=1}^n |
            (\\text{fx}_i - \\text{obs}_i) / \\text{obs}_i
        | * 100%

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    error_fnc : function
        A function that returns the error, default fx - obs. First
        argument is obs, second argument is fx.

    Returns
    -------
    mape : float
        The MAPE [%] of the forecast.

    """
    error = error_fnc(obs, fx)
    return np.mean(np.abs(error / obs)) * 100.0


def normalized_mean_absolute(obs, fx, norm, error_fnc=error):
    """Normalized mean absolute error (NMAE).

    .. math:: \\text{NMAE} = \\text{MAE} / \\text{norm} * 100%

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    norm : float
        Normalized factor, in the same units as obs and fx.
    error_fnc : function
        A function that returns the error, default fx - obs. First
        argument is obs, second argument is fx.

    Returns
    -------
    nmae : float
        The NMAE [%] of the forecast.

    """

    return mean_absolute(obs, fx, error_fnc=error_fnc) / norm * 100.0


def normalized_mean_bias(obs, fx, norm, error_fnc=error):
    """Normalized mean bias error (NMBE).

    .. math:: \\text{NMBE} = \\text{MBE} / \\text{norm} * 100%

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    norm : float
        Normalized factor, in the same units as obs and fx.
    error_fnc : function
        A function that returns the error, default fx - obs. First
        argument is obs, second argument is fx.

    Returns
    -------
    nmbe : float
        The NMBE [%] of the forecast.

    """

    return mean_bias(obs, fx, error_fnc=error_fnc) / norm * 100.0


def normalized_root_mean_square(obs, fx, norm, error_fnc=error):
    """Normalized root mean square error (NRMSE).

    .. math:: \\text{NRMSE} = \\text{RMSE} / \\text{norm} * 100%

    Parameters
    ----------
    obs : (n,) array-like
        Observed values.
    fx : (n,) array-like
        Forecasted values.
    norm : float
        Normalized factor, in the same units as obs and fx.
    error_fnc : function
        A function that returns the error, default fx - obs. First
        argument is obs, second argument is fx.

    Returns
    -------
    nrmse : float
        The NRMSE [%] of the forecast.

    """

    return root_mean_square(obs, fx, error_fnc=error_fnc) / norm * 100.0


def forecast_skill(obs, fx, ref, error_fnc=error):
    """Forecast skill (s).

    .. math:: s = 1 - \\text{RMSE}_\\text{fx} / \\text{RMSE}_\\text{ref}

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
    error_fnc : function
        A function that returns the error, default fx - obs. First
        argument is obs, second argument is fx.

    Returns
    -------
    s : float
        The forecast skill [-] of the forecast relative to a reference
        forecast.

    Notes
    -----
    This function returns 0 if RMSE_fx and RMSE_ref are both 0.
    """

    rmse_fx = root_mean_square(obs, fx, error_fnc=error_fnc)
    rmse_ref = root_mean_square(obs, ref, error_fnc=error_fnc)
    # avoid 0 / 0 --> nan
    if rmse_fx == rmse_ref:
        return 0.
    elif rmse_ref == 0.:
        # avoid divide by 0.
        # typically caused by deadbands and short time periods
        return np.NINF
    else:
        return 1.0 - rmse_fx / rmse_ref


def pearson_correlation_coeff(obs, fx):
    """Pearson correlation coefficient (r).

    .. math:: r = A / (B * C)

    where:

    .. math:: A = \\sum_{i=1}^n (\\text{fx}_i - \\text{fx}_\\text{avg}) *
                  (\\text{obs}_i - \\text{obs}_\\text{avg})
    .. math:: B = \\sqrt{ \\sum_{i=1}^n (\\text{fx}_i - \\text{fx}_\\text{avg})^2 }
    .. math:: C = \\sqrt{ \\sum_{i=1}^n (\\text{obs}_i - \\text{obs}_\\text{avg})^2 }
    .. math:: \\text{fx}_\\text{avg} = 1/n \\sum_{i=1} \\text{fx}_i
    .. math:: \\text{obs}_\\text{avg} = 1/n \\sum_{i=1} \\text{obs}_i

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

    """  # NOQA

    try:
        r, _ = sp.stats.pearsonr(obs, fx)
    except ValueError:
        r = np.nan

    return r


def coeff_determination(obs, fx):
    """Coefficient of determination (R^2).

    .. math:: R^2 = 1 - (A / B)

    where:

    .. math:: A = \\sum_{i=1}^n (\\text{obs}_i - \\text{fx}_i)^2
    .. math:: B = \\sum_{i=1}^n (\\text{obs}_i - \\text{obs}_\\text{avg})^2
    .. math:: \\text{obs}_\\text{avg} = 1/n \\sum_{i=1} \\text{obs}_i

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

    .. math:: \\text{CRMSE} = \\sqrt{1/n \\sum_{i=1}^n (
        (\\text{fx}_i - \\text{fx}_\\text{avg}) -
        (\\text{obs}_i - \\text{obs}_\\text{avg}))^2 }

    where:

    .. math:: \\text{fx}_\\text{avg} = 1/n \\sum_{i=1} \\text{fx}_i
    .. math:: \\text{obs}_\\text{avg} = 1/n \\sum_{i=1} \\text{obs}_i

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

    .. math:: \\text{KSI} = \\int_{p_\\min}^{p_\\max} D_n(p) dp

    where:

    .. math:: D_n(p) = \\max(|\\text{CDF}_\\text{obs}(p) - \\text{CDF}_\\text{fx}(p)|)

    and CDF_obs and CDF_fx are the empirical CDFs of the observations and
    forecasts, respectively. KSI can be normalized as:

    .. math:: \\text{KSI [%]} = \\text{KSI} / a_\\text{critical} * 100%

    where:

    .. math:: a_\\text{critical} = V_c * (p_\\max - p_\\min)
    .. math:: V_c = 1.63 / \\sqrt{n}

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

    """  # NOQA

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

    .. math:: \\text{OVER} = \\int_{p_\\min}^{p_\\max} D_n^*(p) dp

    where:

    .. math::

       D_n^* = \\begin{cases}
          (D_n - V_c) & D_n > V_c \\\\
          D_n^* = 0 & \\text{otherwise}
       \\end{cases}

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

    .. math::  \\text{CPI} = (\\text{KSI} + \\text{OVER} + 2 * \\text{RMSE}) / 4

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

    """  # NOQA
    ksi = kolmogorov_smirnov_integral(obs, fx)
    ov = over(obs, fx)
    rmse = root_mean_square(obs, fx)
    cpi = (ksi + ov + 2 * rmse) / 4.0
    return cpi


def _np_agg_fnc(agg_str, net):
    fnc = _AGG_OPTIONS[agg_str]
    if net:
        return lambda x: fnc(x)
    else:
        return lambda x: fnc(np.abs(x))


def constant_cost(obs, fx, cost_params, error_fnc=error):
    """Compute cost using a datamodel.ConstantCost object
    """
    cost_const = cost_params.cost
    agg_fnc = _np_agg_fnc(cost_params.aggregation, cost_params.net)

    errors = error_fnc(obs, fx)
    return agg_fnc(errors) * cost_const


def _make_time_of_day_cost_ser(times, costs, index, tz, fill):
    if len(index) == 0 or len(times) == 0:
        return 0
    dates = list(np.unique(index.date))
    # extend dates +- 1 day so that index is within the cost
    # ser we construct
    dates.insert(0, min(dates) - dt.timedelta(days=1))
    dates.insert(-1, max(dates) + dt.timedelta(days=1))

    # insert the last cost at 00:00 if not present so forward
    # fill works (sometimes one date w/ tz adjust not enough)
    if fill == 'ffill' and dt.time(0) not in times:
        max_ind = np.argmax(times)
        times.insert(0, dt.time(0))
        costs.insert(0, costs[max_ind])
    # insert the first cost at 23:59 if not present so back
    # fill works even if tz adjusts index
    elif fill == 'bfill' and dt.time(23, 59, 59) not in times:
        min_ind = np.argmin(times)
        times.insert(-1, dt.time(23, 59, 59))
        costs.insert(-1, costs[min_ind])
    # make the cost series
    prod = [(pd.Timestamp.combine(x, y[0]), y[1])
            for x in dates for y in zip(times, costs)]
    base_ser = pd.DataFrame(
        prod, columns=['timestamp', 'cost']
    ).set_index('timestamp')['cost'].tz_localize(tz).sort_index()
    # only get those values at index filling as appropriate
    ser = base_ser.reindex(index, method=fill)
    return ser


def time_of_day_cost(obs, fx, cost_params, error_fnc=error):
    """Compute cost according to a datamodel.TimeOfDayCost"""
    agg_fnc = _np_agg_fnc(cost_params.aggregation, cost_params.net)
    fill = _FILL_OPTIONS[cost_params.fill]

    errors = error_fnc(obs, fx)
    tz = cost_params.timezone or errors.index.tzinfo
    cost_ser = _make_time_of_day_cost_ser(
        cost_params.times, cost_params.cost, errors.index, tz, fill)
    error_cost = errors * cost_ser
    return agg_fnc(error_cost)


def datetime_cost(obs, fx, cost_params, error_fnc=error):
    """Compute cost according to a datamodel.DatetimeCost"""
    agg_fnc = _np_agg_fnc(cost_params.aggregation, cost_params.net)
    fill = _FILL_OPTIONS[cost_params.fill]
    cost_ser = pd.Series(cost_params.cost,
                         index=pd.DatetimeIndex(cost_params.datetimes),
                         dtype=float)

    errors = error_fnc(obs, fx)

    if cost_ser.index.tzinfo is None:
        if cost_params.timezone is not None:
            cost_ser = cost_ser.tz_localize(cost_params.timezone)
        else:
            cost_ser = cost_ser.tz_localize(errors.index.tzinfo)

    error_cost = errors * cost_ser.reindex(errors.index, method=fill)
    return agg_fnc(error_cost)


def _band_masks(bands, errors):
    """Make masks for each band based on which band errors falls in"""
    prev = np.zeros(errors.shape, dtype=bool)
    out = []
    for band in bands:
        emin, emax = band.error_range
        new = (errors >= emin) & (errors <= emax)
        # only those new locations that not also in prev should be used
        both = prev & new
        new[both] = False
        out.append(new)
        prev |= new
    return out


def error_band_cost(obs, fx, cost_params, error_fnc=error):
    """Calculate cost using datamodel.BandedCost parameters"""
    bands = cost_params.bands
    band_cost_functions = [
        partial(_COST_FUNCTION_MAP[band.cost_function],
                cost_params=band.cost_function_parameters,
                error_fnc=error_fnc)
        for band in bands
    ]

    errors = error_fnc(obs, fx)
    out = 0
    masks = _band_masks(bands, errors)
    for mask, fnc in zip(masks, band_cost_functions):
        if not mask.any():
            continue
        mobs = obs[mask]
        mfx = fx[mask]
        out += fnc(mobs, mfx)
    return out


def cost(obs, fx, cost_params, error_fnc=error):
    """
    GOODER DOCS
    """
    fnc = _COST_FUNCTION_MAP[cost_params.type]
    return fnc(obs, fx, cost_params.parameters, error_fnc)


_COST_FUNCTION_MAP = {
    'constant': constant_cost,
    'timeofday': time_of_day_cost,
    'datetime': datetime_cost,
    'errorband': error_band_cost
}

_FILL_OPTIONS = {
    'forward': 'ffill',
    'backward': 'bfill'
}

_AGG_OPTIONS = {
    'sum': np.sum,
    'mean': np.mean
}

# Add new metrics to this map to map shorthand to function
_MAP = {
    'mae': (mean_absolute, 'MAE'),
    'mbe': (mean_bias, 'MBE'),
    'rmse': (root_mean_square, 'RMSE'),
    'mape': (mean_absolute_percentage, 'MAPE'),
    'nmae': (normalized_mean_absolute, 'NMAE'),
    'nmbe': (normalized_mean_bias, 'NMBE'),
    'nrmse': (normalized_root_mean_square, 'NRMSE'),
    's': (forecast_skill, 'Skill'),
    'r': (pearson_correlation_coeff, 'r'),
    'r^2': (coeff_determination, 'R^2'),
    'crmse': (centered_root_mean_square, 'CRMSE'),
    'ksi': (kolmogorov_smirnov_integral, 'KSI'),
    'over': (over, 'OVER'),
    'cpi': (combined_performance_index, 'CPI'),
    'cost': (cost, 'Cost')
}

__all__ = [m[0].__name__ for m in _MAP.values()]

# Functions that require a reference forecast
_REQ_REF_FX = ['s']

# Functions that require normalized factor
_REQ_NORM = ['nmae', 'nmbe', 'nrmse']

_DEADBAND_ALLOWED = [
    'mae', 'mbe', 'rmse', 'mape', 'nmae', 'nmbe', 'nrmse', 's', 'cost']
