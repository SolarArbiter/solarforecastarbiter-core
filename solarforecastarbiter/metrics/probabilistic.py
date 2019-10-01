"""Probablistic forecast error metrics."""

import numpy as np

__all__ = [
    "brier_score",
    "brier_skill_score",
    "reliability",
    "resolution",
    "uncertainty",
    "sharpness",
    "crps",
]


def brier_score(fx, obs):
    """Brier Score (BS).

        BS = 1/n * \sum_{i=1}^n (fx_i - obs_i)^2

    Parameters
    ----------
    fx : (n,) array_like
        Forecasted probability of the event (between 0 and 1) for n samples.
    obs : (n,) array_like
        Actual outcome of the event (0=did not happen, 1=did happen) for n
        samples.

    Returns
    -------
    score : float
        The Brier Score.

    """
    return np.mean((fx - obs) ** 2)


def brier_skill_score(fx, obs, ref):
    """Brier Skill Score (BSS).

        BSS = 1 - BS_f / BS_ref

    where BS_f is the evaluated forecast and BS_ref is a reference forecast.

    Parameters
    ----------
    fx: (n,) array_like
        Forecasted probability of the event (between 0 and 1) for n samples.
    obs: (n,) array_like
        Actual outcome of the event (0=did not happen, 1=did happen) for n
        samples.
    ref: (n,) array_like
        Reference forecast of the probability of the event (between 0 and 1).

    Returns
    -------
    skill : float
        The Brier Skill Score [-].

    """
    bs_f = brier_score(fx, obs)
    bs_ref = brier_score(ref, obs)
    return 1.0 - bs_f / bs_ref


def reliability(fx, obs):
    """Reliability (REL).

    Parameters
    ----------
    fx :
    obs :

    Returns
    -------
    REL : float
        The reliability (REL=0 means the forecast is perfectly reliable).

    """
    return None


def resolution():
    """Resolution (RES).

    Parameters
    ----------

    Returns
    -------
    RES : float
        The resolution (higher values are better).

    """
    return None


def uncertainty():
    """Uncertainty (UNC).

    Parameters
    ----------

    Returns
    -------
    UNC : float
        The uncertainty (lower values indicate event being forecasted occurs
        only rarely).

    """
    return None


def sharpness(fx_lower, fx_upper):
    """Sharpness (SH).

        SH = 1/n * \sum_{i=1}^n (f_{u,i} - f_{l,i})

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


def crps():
    """Continuous Ranked Probability Score (CRPS).

    Parameters
    ----------
    F : (n, m) array_like
        Predicted CDF.
    O : (n, m) array_like
        Cumulative-probability step function.
    q : (m,) array_like
        The quantiles.

    Returns
    -------
    CRPS : float
        The CRPS value for the given forecasts.

    """
    return None
