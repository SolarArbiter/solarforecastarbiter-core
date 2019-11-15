"""Probablistic forecast error metrics."""

import numpy as np

__all__ = [
    "brier_score",
    "brier_skill_score",
    "reliability",
    "resolution",
    "uncertainty",
    "sharpness",
]


def brier_score(fx, fx_prob, obs):
    """Brier Score (BS).

    Parameters
    ----------
    fx : (n,) array_like
        Forecasts (physical units).
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.
    obs : (n,) array_like
        Observations (physical unit).

    Returns
    -------
    score : float
        The Brier Score.

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    BS = np.mean((f - o) ** 2)
    return BS


def brier_skill_score(fx, fx_prob, ref, ref_prob, obs):
    """Brier Skill Score (BSS).

        BSS = 1 - BS_f / BS_ref

    where BS_fx is the evaluated forecast and BS_ref is a reference forecast.

    Parameters
    ----------
    fx : (n,) array_like
        Forecasts (physical units).
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.
    ref : (n,) array_like
        Reference forecast (physical units).
    ref_prob : (n,) array_like
        Probability [%] associated with the reference forecast.
    obs : (n,) array_like
        Observations (physical unit).

    Returns
    -------
    skill : float
        The Brier Skill Score [unitless].

    """
    bs_fx = brier_score(fx, fx_prob, obs)
    bs_ref = brier_score(ref, ref_prob, obs)
    return 1.0 - bs_fx / bs_ref


def reliability(fx, fx_prob, obs):
    """Reliability (REL) of the forecast.

    Parameters
    ----------
    fx : (n,) array_like
        Forecasts (physical units).
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.
    obs : (n,) array_like
        Observations (physical unit).

    Returns
    -------
    REL : float
        The reliability of the forecast, where a perfectly reliable forecast
        has REL = 0.

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # get unique forecast probabilities
    if len(f) < 1000:
        n_decimals = 1
    else:
        n_decimals = 2

    f = np.around(f, decimals=n_decimals)
    REL = 0.0
    for f_i in np.unique(f):
        N_i = len(f[f == f_i])  # no. forecasts per set
        o_i = np.mean(o[f == f_i])    # mean event value per set
        REL += N_i * (f_i - o_i) ** 2

    REL = REL / len(f)
    return REL


def resolution(fx, fx_prob, obs):
    """Resolution (RES) of the forecast.

    Parameters
    ----------
    fx : (n,) array_like
        Forecasts (physical units).
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.
    obs : (n,) array_like
        Observations (physical unit).

    Returns
    -------
    RES : float
        The resolution of the forecast, where higher values are better.

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # get unique forecast probabilities
    if len(f) < 1000:
        n_decimals = 1
    else:
        n_decimals = 2

    f = np.around(f, decimals=n_decimals)
    RES = 0.0
    o_avg = np.mean(o)
    for f_i in np.unique(f):
        N_i = len(f[f == f_i])  # no. forecasts per set
        o_i = np.mean(o[f == f_i])    # mean event value per set
        RES += N_i * (o_i - o_avg) ** 2

    RES = RES / len(f)
    return RES


def uncertainty(fx, obs):
    """Uncertainty (UNC) of the forecast.

        UNC = base_rate * (1 - base_rate)

    where base_rate = 1/n * sum_{t=1}^n o_t.

    Parameters
    ----------
    fx : (n,) array_like
        Forecasts (physical units).
    obs : (n,) array_like
        Observations (physical unit).

    Returns
    -------
    UNC : float
        The uncertainty (lower values indicate event being forecasted occurs
        only rarely).

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    base_rate = np.mean(o)
    return base_rate * (1.0 - base_rate)


def sharpness(fx_lower, fx_upper):
    """Sharpness (SH).

        SH = 1/n * sum_{i=1}^n (f_{u,i} - f_{l,i})

    Parameters
    ----------
    fx_lower : (n,) array_like
    fx_upper : (n,) array_like

    Returns
    -------
    SH : float
        The sharpness.

    """
    return np.mean(fx_upper - fx_lower)
