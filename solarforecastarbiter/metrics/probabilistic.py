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


def _transform(fx, obs, kind="actuals", x=[25, 50, 75, 100]):
    """Transform data to standard form for metrics.

    Parameters
    ----------
    fx : (m, n) array_like
        The forecasts for m samples and n categories, where either a) the
        categories are fixed percentiles and the values are predicted actuals
        or b) the categories are fixed actuals and the values are percentiles.
    obs : (m,) array_like
        The observed actuals for m samples, in units of flux (e.g. W/m^2),
        power (e.g. kW) or energy (e.g. kWh).
    kind : str {"actuals", "percentiles"}
        The type of forecast values provided:
        - "actuals": predictions of the actuals (in units of flux, power or
          energy) for n categories of percentiles
        - "percentiles": predictions of the percentiles [%] for n categories of
          actuals
    x : (n,) array_like
        The grid corresponding to the forecast categories (columns). If
        `kind="actuals"`, then `x` is a list of percentiles [%]. If
        `kind="percentiles"`, then `x` is a list of actuals (units of flux,
        power or energy). In both cases, the grid is assumed to be non-negative
        and monotonically increasing.

    Returns
    -------
    F : (m, n) array_like
    O : (m, n) array_like

    Examples
    --------
    >>> # predict actuals for a grid of probability values
    >>> kind = "actuals"
    >>> x = np.array([25, 50, 75])          # grid: percentiles [%]
    >>> obs = np.array([0.11, 0.25, 0.30])  # actuals [MW]
    >>> fx = np.array([                     # predict actuals [MW]
    ...     [0.02, 0.06, 0.13],
    ...     [0.09, 0.16, 0.28],
    ...     [0.14, 0.19, 0.35],
    ... ])
    >>> F, O = _transform(fx, obs, kind=kind, x=x)
    >>> print(O)
    array([])
    >>> print(F)
    array([])

    >>> # predict probabilities for a grid of actuals
    >>> kind = "percentiles"
    >>> x = np.array([10, 20, 30])   # grid: actuals [MW]
    >>> obs = np.array([8, 13, 21])  # actuals [MW]
    >>> fx = np.array([              # predict percentiles [%]
    ...     [33, 61, 98],  # Pr(x < 10), Pr(x < 20), Pr(x < 30)
    ...     [],
    ...     [],
    ... ])
    >>> F, O = _transform(fx, obs, kind=kind, x=x)

    """

    m, n = fx.shape

    if kind == "percentiles":  # predict percentiles for grid of actuals
        # binary event CDF [-]:
        # - if actual < category: 1, else: 0
        #
        # example:
        # - observed: 12 MW
        # - category: < 20 MW
        # - output: 1 (since 12 MW < 20 MW is true)
        X = np.tile(x, [m, 1])    # grid: (n,) => (m, n)
        A = np.tile(obs, [1, n])  # observed: (m,) => (m, n)
        O = np.zeros([m, n])
        O[A < X] = 1

        # predicted percentiles [-] (CDF)
        F = fx / 100.0

        return F, O
    elif kind == "actuals":  # predict actuals for a grid of percentiles
        # binary event CDF [-]:
        A = np.tiles(obs, [1, n])   # observed: (m,) => (m, n)
        X = np.copy(fx)             # predicted: (m, n)
        O = np.zeros([m, n])
        O[A < X] = 1

        # predicted percentiles [-] (CDF)
        F = np.tile(x / 100.0, [m, 1])

        return F, O
    else:
        raise NameError


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
    o = np.zeros_like(obs)
    o[obs <= fx] = 1.0

    # forecast probabilities [-]
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
        The Brier Skill Score [-].

    """
    bs_fx = brier_score(fx, fx_prob, obs)
    bs_ref = brier_score(ref, ref_prob, obs)
    return 1.0 - bs_fx / bs_ref


def reliability(fx, obs):
    """Reliability (REL) of the forecast.

    Parameters
    ----------
    fx: (n,) array_like
        Forecasted probability of the event (between 0 and 1) for n samples.
    obs: (n,) array_like
        Actual outcome of the event (0=did not happen, 1=did happen) for n
        samples.

    Returns
    -------
    REL : float
        The reliability of the forecast, where a perfectly reliable forecast
        has REL = 0.

    """
    return None


def resolution(fx, obs):
    """Resolution (RES) of the forecast.

    Parameters
    ----------
    fx: (n,) array_like
        Forecasted probability of the event (between 0 and 1) for n samples.
    obs: (n,) array_like
        Actual outcome of the event (0=did not happen, 1=did happen) for n
        samples.

    Returns
    -------
    RES : float
        The resolution of the forecast, where higher values are better.

    """
    return None


def uncertainty(obs):
    """Uncertainty (UNC) of the forecast.

        UNC = base_rate * (1 - base_rate)

    where base_rate = 1/n * sum_{t=1}^n obs_t

    Parameters
    ----------
    obs: (n,) array_like
        Actual outcome of the event (0=did not happen, 1=did happen) for n
        samples.

    Returns
    -------
    UNC : float
        The uncertainty (lower values indicate event being forecasted occurs
        only rarely).

    """
    base_rate = np.mean(obs)
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


def crps(F, O, q):
    """Continuous Ranked Probability Score (CRPS).

        CRPS = 1/n * sum_{i=1}^n (int |F_i(x) - O_i(x)| dx)

    Parameters
    ----------
    F : (m, n) array_like
        Predicted CDF for m samples and n bins.
    O : (m, n) array_like
        Cumulative-probability step function for m samples and n bins.
    q : (n,) array_like
        The n bins.

    Returns
    -------
    CRPS : float
        The CRPS value for the given forecasts.

    """

    return np.mean(np.trapz(np.abs(F - O), x=q))
