"""Event forecast error metrics."""

import numpy as np


def _event2count(obs, fx):
    """Convert events (True/False) into counts.

    Given forecasts and observations of events (True=event occurred,
    False=event did not occur), the pairs of forecasts and observations can be
    placed into four categories:
    - True Positive (TP): forecast = event, observed = event
    - False Positive (FP): forecast = event, observed = no event
    - True Negative (TN): forecast = no event, observed = no event
    - False Negative (FN): forecast = no event, observed = event

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values (True=event, False=no event).

    Returns
    -------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    tn : int
        Number of true negatives.
    fn : int
        Number of false negatives.

    Raises
    ------
    RuntimeError
        If there is no forecast or observation timeseries data, or the forecast
        and observation timeseries data do not have the same length.

    """

    obs = np.asarray(obs)
    fx = np.asarray(fx)

    if len(obs) == 0:
        raise RuntimeError("No Observation timeseries data.")
    elif len(fx) == 0:
        raise RuntimeError("No Forecast timeseries data.")
    elif len(obs) != len(fx):
        raise RuntimeError("Forecast and Observation timeseries data do not "
                           "have the same length.")

    tp = np.count_nonzero(np.logical_and(fx, obs))
    fp = np.count_nonzero(np.logical_and(fx, ~obs))
    tn = np.count_nonzero(np.logical_and(~fx, ~obs))
    fn = np.count_nonzero(np.logical_and(~fx, obs))
    return tp, fp, tn, fn


def probability_of_detection(obs, fx):
    """Probability of Detection (POD).

    .. math:: \\text{POD} = \\text{TP} / (\\text{TP} + \\text{FN})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values (True=event, False=no event).

    Returns
    -------
    pod : float
        The POD of the forecast.

    Raises
    ------
    RuntimeError
        If there is no forecast or observation timeseries data, or the forecast
        and observation timeseries data do not have the same length.

    """

    tp, fp, tn, fn = _event2count(obs, fx)
    if (tp + fn) == 0:
        return 0.0
    else:
        return tp / (tp + fn)


def false_alarm_ratio(obs, fx):
    """False Alarm Ratio (FAR).

    .. math:: \\text{FAR} = \\text{FP} / (\\text{TP} + \\text{FP})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values (True=event, False=no event).

    Returns
    -------
    far : float
        The FAR of the forecast.

    Raises
    ------
    RuntimeError
        If there is no forecast or observation timeseries data, or the forecast
        and observation timeseries data do not have the same length.

    """
    tp, fp, tn, fn = _event2count(obs, fx)
    if (tp + fp) == 0:
        return 0.0
    else:
        return fp / (tp + fp)


def probability_of_false_detection(obs, fx):
    """Probability of False Detection (POFD).

    .. math:: \\text{POFD} = \\text{FP} / (\\text{FP} + \\text{TN})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values (True=event, False=no event).

    Returns
    -------
    pofd : float
        The POFD of the forecast.

    Raises
    ------
    RuntimeError
        If there is no forecast or observation timeseries data, or the forecast
        and observation timeseries data do not have the same length.

    """

    tp, fp, tn, fn = _event2count(obs, fx)
    if (fp + tn) == 0:
        return 0.0
    else:
        return fp / (fp + tn)


def critical_success_index(obs, fx):
    """Critical Success Index (CSI).

    .. math:: \\text{CSI} = \\text{TP} / (\\text{TP} + \\text{FP} + \\text{FN})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values (True=event, False=no event).

    Returns
    -------
    csi : float
        The CSI of the forecast.

    Raises
    ------
    RuntimeError
        If there is no forecast or observation timeseries data, or the forecast
        and observation timeseries data do not have the same length.

    """

    tp, fp, tn, fn = _event2count(obs, fx)
    if (tp + fp + fn) == 0:
        return 0.0
    else:
        return tp / (tp + fp + fn)


def event_bias(obs, fx):
    """Event Bias (EBIAS).

    .. math:: \\text{EBIAS} = (\\text{TP} + \\text{FP}) / (\\text{TP} + \\text{FN})  # NOQA

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values (True=event, False=no event).

    Returns
    -------
    ebias : float
        The EBIAS of the forecast.

    Raises
    ------
    RuntimeError
        If there is no forecast or observation timeseries data, or the forecast
        and observation timeseries data do not have the same length.

    """

    tp, fp, tn, fn = _event2count(obs, fx)
    if (tp + fn) == 0:
        return 0.0
    else:
        return (tp + fp) / (tp + fn)


def event_accuracy(obs, fx):
    """Event Accuracy (EA).

    .. math:: \\text{EA} = (\\text{TP} + \\text{TN}) / n

    where n is the number of samples.

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values (True=event, False=no event).

    Returns
    -------
    ea : float
        The EA of the forecast.

    Raises
    ------
    RuntimeError
        If there is no forecast or observation timeseries data, or the forecast
        and observation timeseries data do not have the same length.

    """

    n = len(obs)
    tp, fp, tn, fn = _event2count(obs, fx)
    return (tp + tn) / n


# Add new metrics to this map to map shorthand to function
_MAP = {
    'pod': (probability_of_detection, 'POD'),
    'far': (false_alarm_ratio, 'FAR'),
    'pofd': (probability_of_false_detection, 'POFD'),
    'csi': (critical_success_index, 'CSI'),
    'ebias': (event_bias, 'EBIAS'),
    'ea': (event_accuracy, 'EA')
}

__all__ = [m[0].__name__ for m in _MAP.values()]

# Functions that require a reference forecast
_REQ_REF_FX = []

# Functions that require normalized factor
_REQ_NORM = []
