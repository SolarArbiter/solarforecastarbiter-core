"""Probablistic forecast error metrics."""

import numpy as np


def brier_score(obs, fx, fx_prob):
    """Brier Score (BS).

        BS = 1/n sum_{i=1}^n (f_i - o_i)^2

    where n is the number of forecasts, f_i is the forecasted probability of
    event i, and o_i is the observed event indicator (o_i=0: event did not
    occur, o_i=1: event occured). The forecasts are supplied as the
    right-hand-side of a CDF interval, e.g., forecast <= 10 MW at time i, and
    therefore o_i is defined as:

        o_i = 1 if obs_i <= fx_i, else o_i = 0

    where fx_i and obs_i are the forecast and observation at time i,
    respectively.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n,) array_like
        Forecasts (physical units) of the right-hand-side of a CDF interval,
        e.g., fx = 10 MW is interpreted as forecasting <= 10 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.

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


def brier_skill_score(obs, fx, fx_prob, ref, ref_prob):
    """Brier Skill Score (BSS).

        BSS = 1 - BS_fx / BS_ref

    where BS_fx is the Brier Score of the evaluated forecast and BS_ref is the
    Brier Score of a reference forecast.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n,) array_like
        Forecasts (physical units) of the right-hand-side of a CDF interval,
        e.g., fx = 10 MW is interpreted as forecasting <= 10 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.
    ref : (n,) array_like
        Reference forecast (physical units) of the right-hand-side of a CDF
        interval.
    ref_prob : (n,) array_like
        Probability [%] associated with the reference forecast.

    Returns
    -------
    skill : float
        The Brier Skill Score [unitless].

    """
    bs_fx = brier_score(obs, fx, fx_prob)
    bs_ref = brier_score(obs, ref, ref_prob)
    skill = 1.0 - bs_fx / bs_ref
    return skill


def quantile_score(obs, fx, fx_prob):
    """Quantile Score (QS).

    .. math::

        \\text{QS} = \\frac{1}{n} \\sum_{i=1}^n (fx_i - obs_i) * (p - 1\\{obs_i > fx_i\\})

    where :math:`n` is the number of forecasts, :math:`obs_i` is an
    observation, :math:`fx_i` is a forecast, :math:`1\\{obs_i > fx_i\\}` is an
    indicator function (1 if :math:`obs_i > fx_i`, 0 otherwise) and :math:`p`
    is the probability that :math:`obs_i <= fx_i`. [1]_ [2]_

    If :math:`obs > fx`, then we have:

    .. math::

        (fx - obs) < 0 \\\\
        (p - 1\\{obs > fx\\}) = (p - 1) <= 0 \\\\
        (fx - obs) * (p - 1) >= 0

    If instead :math:`obs < fx`, then we have:

    .. math::

        (fx - obs) > 0 \\\\
        (p - 1\\{obs > fx\\}) = (p - 0) >= 0 \\\\
        (fx - obs) * p >= 0

    Therefore, the quantile score is non-negative regardless of the obs and fx.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n,) array_like
        Forecasts (physical units) of the right-hand-side of a CDF interval,
        e.g., fx = 10 MW is interpreted as forecasting <= 10 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    qs : float
        The Quantile Score, with the same units as the observations.

    Notes
    -----
    Quantile score is meant to be computed for a single probability of
    :math:`n` samples.

    Examples
    --------
    >>> obs = 100     # observation [MW]
    >>> fx = 80       # forecast [MW]
    >>> fx_prob = 60  # probability [%]
    >>> quantile_score(obs, fx, fx_prob)   # score [MW]
    8.0

    References
    ----------
    .. [1] Koenker and Bassett, Jr. (1978) "Regression Quantiles", Econometrica
       46 (1), pp. 33-50. doi: 10.2307/1913643
    .. [2] Wilks (2020) "Forecast Verification". In "Statistical Methods in the
       Atmospheric Sciences" (3rd edition). Academic Press. ISBN: 9780123850225

    """  # NOQA: E501,W605

    # Prob(obs <= fx) = p
    p = fx_prob / 100.0
    qs = np.mean((fx - obs) * (p - np.where(obs > fx, 1.0, 0.0)))
    return qs


def quantile_skill_score(obs, fx, fx_prob, ref, ref_prob):
    """Quantile Skill Score (QSS).

    .. math::

        \\text{QSS} = 1 - \\text{QS}_{\\text{fx}} / \\text{QS}_{\\text{ref}}

    where :math:`\\text{QS}_{\\text{fx}}` is the Quantile Score of the
    evaluated forecast and :math:`\\text{QS}_{\\text{ref}}` is the Quantile
    Score of a reference forecast. [1]_

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n,) array_like
        Forecasts (physical units) of the right-hand-side of a CDF interval,
        e.g., fx = 10 MW is interpreted as forecasting <= 10 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.
    ref : (n,) array_like
        Reference forecast (physical units) of the right-hand-side of a CDF
        interval.
    ref_prob : (n,) array_like
        Probability [%] associated with the reference forecast.

    Returns
    -------
    skill : float
        The Quantile Skill Score [unitless].

    References
    ----------
    .. [1] Bouallegue, Pinson and Friederichs (2015) "Quantile forecast
       discrimination ability and value", Quarterly Journal of the Royal
       Meteorological Society 141, pp. 3415-3424. doi: 10.1002/qj.2624

    Notes
    -----
    This function returns 0 if QS_fx and QS_ref are both 0.

    See Also
    --------
    :py:func:`solarforecastarbiter.metrics.probabilistic.quantile_score`

    """

    qs_fx = quantile_score(obs, fx, fx_prob)
    qs_ref = quantile_score(obs, ref, ref_prob)

    # avoid 0 / 0 --> nan
    if qs_fx == qs_ref:
        return 0.0
    elif qs_ref == 0.0:
        # avoid divide by 0
        # typically caused by deadbands and short time periods
        return np.NINF
    else:
        return 1.0 - qs_fx / qs_ref


def _unique_forecasts(f):
    """Convert forecast probabilities to a set of unique values.

    Determine a set of unique forecast probabilities, based on input forecast
    probabilities of arbitrary precision, and approximate the input
    probabilities to lie within the set of unique values.

    Parameters
    ----------
    f : (n,) array_like
        Probability [unitless] associated with the forecasts.

    Returns
    -------
    f_uniq : (n,) array_like
        The converted forecast probabilities [unitless].

    Notes
    -----
    This implementation determines the set of unique forecast probabilities by
    rounding the input probabilities to a precision determined by the number of
    input probability values: if less than 1000 samples, bin by tenths;
    otherwise bin by hundredths.

    Examples
    --------
    >>> f = np.array([0.1234, 0.156891, 0.10561])
    >>> _unique_forecasts(f)
    array([0.1, 0.2, 0.1])

    """

    if len(f) >= 1000:
        n_decimals = 2  # bin by hundredths (0.01, 0.02, etc.)
    else:
        n_decimals = 1  # bin by tenths (0.1, 0.2, etc.)

    f_uniq = np.around(f, decimals=n_decimals)
    return f_uniq


def brier_decomposition(obs, fx, fx_prob):
    """The 3-component decomposition of the Brier Score.

        BS = REL - RES + UNC

    where REL is the reliability, RES is the resolution and UNC is the
    uncertatinty.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n,) array_like
        Forecasts (physical units) of the right-hand-side of a CDF interval,
        e.g., fx = 10 MW is interpreted as forecasting <= 10 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    rel : float
        The reliability of the forecast [unitless], where a perfectly reliable
        forecast has value of 0.
    res : float
        The resolution of the forecast [unitless], where higher values are
        better.
    unc : float
        The uncertainty [unitless], where lower values indicate the event being
        forecasted occurs rarely.

    Notes
    -----
    The current implementation iterates over the unique forecasts to compute
    the reliability and resolution, rather than using a vectorized formulation.
    While a vectorized formulation may be more computationally efficient, the
    clarity of the iterate version outweighs the efficiency gains from the
    vectorized version. Additionally, the number of unique forecasts is
    currently capped at 100, which small enough that there is likely no
    practical difference in computation time between the iterate vs vectorized
    versions.

    """

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # get unique forecast probabilities by binning
    f = _unique_forecasts(f)

    # reliability and resolution
    rel, res = 0.0, 0.0
    o_avg = np.mean(o)
    for f_i, N_i in np.nditer(np.unique(f, return_counts=True)):
        o_i = np.mean(o[f == f_i])      # mean event value per set
        rel += N_i * (f_i - o_i) ** 2
        res += N_i * (o_i - o_avg) ** 2
    rel /= len(f)
    res /= len(f)

    # uncertainty
    base_rate = np.mean(o)
    unc = base_rate * (1.0 - base_rate)

    return rel, res, unc


def reliability(obs, fx, fx_prob):
    """Reliability (REL) of the forecast.

        REL = 1/n sum_{i=1}^I N_i (f_i - o_{i,avg})^2

    where n is the total number of forecasts, I is the number of unique
    forecasts (f_1, f_2, ..., f_I), N_i is the number of times each unique
    forecast occurs, o_{i,avg} is the average of the observed events during
    which the forecast was f_i.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n,) array_like
        Forecasts (physical units) of the right-hand-side of a CDF interval,
        e.g., fx = 10 MW is interpreted as forecasting <= 10 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    rel : float
        The reliability of the forecast [unitless], where a perfectly reliable
        forecast has value of 0.

    See Also
    --------
    brier_decomposition : 3-component decomposition of the Brier Score

    """

    rel = brier_decomposition(obs, fx, fx_prob)[0]
    return rel


def resolution(obs, fx, fx_prob):
    """Resolution (RES) of the forecast.

        RES = 1/n sum_{i=1}^I N_i (o_{i,avg} - o_{avg})^2

    where n is the total number of forecasts, I is the number of unique
    forecasts (f_1, f_2, ..., f_I), N_i is the number of times each unique
    forecast occurs, o_{i,avg} is the average of the observed events during
    which the forecast was f_i, and o_{avg} is the average of all observed
    events.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n,) array_like
        Forecasts (physical units) of the right-hand-side of a CDF interval,
        e.g., fx = 10 MW is interpreted as forecasting <= 10 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    res : float
        The resolution of the forecast [unitless], where higher values are
        better.

    See Also
    --------
    brier_decomposition : 3-component decomposition of the Brier Score

    """

    res = brier_decomposition(obs, fx, fx_prob)[1]
    return res


def uncertainty(obs, fx, fx_prob):
    """Uncertainty (UNC) of the forecast.

        UNC = base_rate * (1 - base_rate)

    where base_rate = 1/n sum_{i=1}^n o_i, and o_i is the observed event.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n,) array_like
        Forecasts (physical units) of the right-hand-side of a CDF interval,
        e.g., fx = 10 MW is interpreted as forecasting <= 10 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    unc : float
        The uncertainty [unitless], where lower values indicate the event being
        forecasted occurs rarely.

    See Also
    --------
    brier_decomposition : 3-component decomposition of the Brier Score

    """

    unc = brier_decomposition(obs, fx, fx_prob)[2]
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


def continuous_ranked_probability_score(obs, fx, fx_prob):
    """Continuous Ranked Probability Score (CRPS).

        CRPS = 1/n sum_{i=1}^n int (F_i - O_i)^2 dx

    where F_i is the CDF of the forecast at time i and O_i is the CDF
    associated with the observed value obs_i:

        O_{i, j} = 1 if obs_i <= fx_{i, j}, else O_{i, j} = 0

    where obs_i is the observation at time i, and fx_{i, j} is the forecast at
    time i for CDF interval j.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n, d) array_like
        Forecasts (physical units) of the right-hand-side of a CDF with d
        intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is interpreted as
        <= 10 MW, <= 20 MW, <= 30 MW.
    fx_prob : (n, d) array_like
        Probability [%] associated with the forecasts.

    Returns
    -------
    crps : float
        The Continuous Ranked Probability Score [unitless].

    Raises
    ------
    ValueError
        If the forecasts have incorrect dimensions; either a) the forecasts are
        for a single sample (n=1) with d CDF intervals but are given as a 1D
        array with d values or b) the forecasts are given as 2D arrays (n,d)
        but do not contain at least 2 CDF intervals (i.e. d < 2).

    Examples
    --------

    Forecast probabilities of <= 10 MW and <= 20 MW:
    >>> fx = np.array([[10, 20], [10, 20]])
    >>> fx_prob = np.array([[30, 50], [65, 100]])
    >>> obs = np.array([8, 12])
    >>> continuous_ranked_probability_score(obs, fx, fx_prob)
    4.5625

    Forecast thresholds for constant probabilities (25%, 75%):
    >>> fx = np.array([[5, 15], [8, 14]])
    >>> fx_prob = np.array([[25, 75], [25, 75]])
    >>> obs = np.array([8, 10])
    >>> continuous_ranked_probability_score(obs, fx, fx_prob)
    0.5

    """

    # match observations to fx shape: (n,) => (n, d)
    if np.ndim(fx) < 2:
        raise ValueError("forecasts must be 2D arrays (expected (n,d), got"
                         f"{np.shape(fx)})")
    elif np.shape(fx)[1] < 2:
        raise ValueError("forecasts must have d >= 2 CDF intervals "
                         f"(expected >= 2, got {np.shape(fx)[1]})")
    else:
        obs = np.tile(obs, (fx.shape[1], 1)).T

    # event: 0=did not happen, 1=did happen
    o = np.where(obs <= fx, 1.0, 0.0)

    # forecast probabilities [unitless]
    f = fx_prob / 100.0

    # integrate along each sample, then average all samples
    integrand = (f - o) ** 2
    dx = np.diff(fx, axis=1)
    crps = np.mean(np.sum(integrand[:, :-1] * dx, axis=1))
    return crps


def crps_skill_score(obs, fx, fx_prob, ref, ref_prob):
    """CRPS skill score.

        CRPSS = 1 - CRPS_fx / CRPS_ref

    where CRPS_fx is the CPRS of the evaluated forecast and CRPS_ref is the
    CRPS of a reference forecast.

    Parameters
    ----------
    obs : (n,) array_like
        Observations (physical unit).
    fx : (n, d) array_like
        Forecasts (physical units) of the right-hand-side of a CDF with d
        intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is interpreted as
        <= 10 MW, <= 20 MW, <= 30 MW.
    fx_prob : (n,) array_like
        Probability [%] associated with the forecasts.
    ref : (n, d) array_like
        Reference forecasts (physical units) of the right-hand-side of a CDF
        with d intervals (d >= 2), e.g., fx = [10 MW, 20 MW, 30 MW] is
        interpreted as <= 10 MW, <= 20 MW, <= 30 MW.
    ref_prob : (n,) array_like
        Probability [%] associated with the reference forecast.

    Returns
    -------
    skill : float
        The CRPS skill score [unitless].

    See Also
    --------
    :py:func:`solarforecastarbiter.metrics.probabilistic.continuous_ranked_probability_score`

    """

    if np.isscalar(ref):
        return np.nan
    else:
        crps_fx = continuous_ranked_probability_score(obs, fx, fx_prob)
        crps_ref = continuous_ranked_probability_score(obs, ref, ref_prob)

        if crps_fx == crps_ref:
            return 0.0
        elif crps_ref == 0.0:
            # avoid divide by zero
            return np.NINF
        else:
            return 1 - crps_fx / crps_ref


# Add new metrics to this map to map shorthand to function
_MAP = {
    'bs': (brier_score, 'BS'),
    'bss': (brier_skill_score, 'BSS'),
    'rel': (reliability, 'REL'),
    'res': (resolution, 'RES'),
    'unc': (uncertainty, 'UNC'),
    'qs': (quantile_score, 'QS'),
    'qss': (quantile_skill_score, 'QSS'),
    # 'sh': (sharpness, 'SH'),  # TODO
    'crps': (continuous_ranked_probability_score, 'CRPS'),
    'crpss': (crps_skill_score, 'CRPSS'),
}

__all__ = [m[0].__name__ for m in _MAP.values()]

# Functions that require a reference forecast
_REQ_REF_FX = ['bss', 'qss', 'crpss']

# Functions that require normalized factor
_REQ_NORM = []

# Functions that require full distribution forecasts (as 2dim)
_REQ_DIST = ['crps', 'crpss']

# TODO: Functions that require two forecasts (e.g., sharpness)
# _REQ_FX_FX = ['sh']
