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

        BS = 1/n sum_{i=1}^n (f_i - o_i)^2

    where n is the number of forecasts, f_i is the forecasted probability of
    event i, and o_i is the observed event indicator (o_i=0: event did not
    occur, o_i=1: event occured).

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
    bs : float
        The Brier Score [unitless], bounded between 0 and 1, where values
        closer to 0 indicate better forecast performance and values closer to 1
        indicate worse performance.

    Notes
    -----
    The Brier Score implemented in this function is for binary outcomes only,
    rather than the more general (but less commonly used) categorical version.

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    bs = np.mean((f - o) ** 2)
    return bs


def brier_skill_score(fx, fx_prob, ref, ref_prob, obs):
    """Brier Skill Score (BSS).

        BSS = 1 - BS_f / BS_ref

    where BS_fx is the Brier Score of the evaluated forecast and BS_ref is the
    Brier Score of a reference forecast.

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
    skill = 1.0 - bs_fx / bs_ref
    return skill


def _unique_forecasts(f):
    """Convert forecast probabilities to a set of unique probability values.

    Parameters
    ----------
    f : (n,) array_like
        Probability [unitless] associated with the forecasts.

    Returns
    -------
    f_uniq : (n,) array_like
        The probabilities converted to lie within a unique set of values.

    """

    if len(f) > 1000:
        n_decimals = 2
    else:
        n_decimals = 1

    f_uniq = np.around(f, decimals=n_decimals)
    return f_uniq

def reliability(fx, fx_prob, obs):
    """Reliability (REL) of the forecast.

        REL = 1/n sum_{i=1}^I N_i (f_i - o_{i,avg})^2

    where n is the total number of forecasts, I is the number of unique
    forecasts (f_1, f_2, ..., f_I), N_i is the number of times each unique
    forecast occurs, o_{i,avg} is the average of the observed events during
    which the forecast was f_i.

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
    rel : float
        The reliability of the forecast [unitless], where a perfectly reliable
        forecast has value of 0.

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # get unique forecast probabilities
    f = _unique_forecasts(f)

    rel = 0.0
    for f_i, N_i in np.nditer(np.unique(f, return_counts=True)):
        o_i = np.mean(o[f == f_i])
        rel += N_i * (f_i - o_i) ** 2
    rel /= len(f)
    return rel


def resolution(fx, fx_prob, obs):
    """Resolution (RES) of the forecast.

        RES = 1/n sum_{i=1}^I N_i (o_{i,avg} - o_{avg})^2

    where n is the total number of forecasts, I is the number of unique
    forecasts (f_1, f_2, ..., f_I), N_i is the number of times each unique
    forecast occurs, o_{i,avg} is the average of the observed events during
    which the forecast was f_i, and o_{avg} is the average of all observed
    events.

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
    res : float
        The resolution of the forecast [unitless], where higher values are
        better.

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # get unique forecast probabilities
    f = _unique_forecasts(f)

    res = 0.0
    o_avg = np.mean(o)
    for f_i, N_i in np.nditer(np.unique(f, return_counts=True)):
        o_i = np.mean(o[f == f_i])    # mean event value per set
        res += N_i * (o_i - o_avg) ** 2
    res /= len(f)
    return res


def uncertainty(fx, obs):
    """Uncertainty (UNC) of the forecast.

        UNC = base_rate * (1 - base_rate)

    where base_rate = 1/n sum_{i=1}^n o_i, and o_i is the observed event.

    Parameters
    ----------
    fx : (n,) array_like
        Forecasts (physical units).
    obs : (n,) array_like
        Observations (physical unit).

    Returns
    -------
    unc : float
        The uncertainty [unitless], where lower values indicate the event being
        forecasted occurs rarely.

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    base_rate = np.mean(o)
    unc = base_rate * (1.0 - base_rate)
    return unc


def sharpness(fx_lower, fx_upper):
    """Sharpness (SH).

        SH = 1/n sum_{i=1}^n (f_{u,i} - f_{l,i})

    where n is the total number of forecasts, f_{u,i} is the upper prediction
    interval value and f_{l,i} is the lower prediction interval value for
    sample i.

    Parameters
    ----------
    fx_lower : (n,) array_like
        The lower prediction interval values (physical units).
    fx_upper : (n,) array_like
        The upper prediction interval values (physical units).

    Returns
    -------
    SH : float
        The sharpness (physical units), where smaller sharpness values indicate
        "tighter" prediction intervals.

    """
    sh = np.mean(fx_upper - fx_lower)
    return sh
