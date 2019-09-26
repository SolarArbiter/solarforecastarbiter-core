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


def brier_score(f, o):
    """Brier Score (BS).

        BS = 1/n * \sum_{i=1}^n (f_i - o_i)^2

    Parameters
    ----------
    f : (n,) array_like
        Forecasted probability of the event (between 0 and 1) for n samples.
    o : (n,) array_like
        Actual outcome of the event (0=did not happen, 1=did happen) for n
        samples.

    Returns
    -------
    score : float
        The Brier Score.

    """
    return np.mean((f - o) ** 2)


def brier_skill_score(fx, obs, ref):
    """Brier Skill Score (BSS).

        BSS = 1 - BS_f / BS_ref

    where BS_f is the evaluated forecast and BS_ref is a reference forecast.

    Parameters
    ----------
    fx: (n,) array_like
    obs: (n,) array_like
    ref: (n,) array_like
        Reference forecast.

    Returns
    -------
    skill : float
        The Brier Skill Score [-].

    """
    bs_f = brier_score(fx, obs)
    bs_ref = brier_score(ref, obs)
    return 1.0 - bs_f / bs_ref


def reliability():
    """Reliability (REL)."""
    return None


def resolution():
    """Resolution (RES)."""
    return None


def uncertainty():
    """Uncertainty (UNC)."""
    return None


def sharpness():
    """Sharpness (SH)."""
    return None


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
