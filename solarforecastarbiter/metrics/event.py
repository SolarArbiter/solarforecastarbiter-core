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
        Forecasted event values.

    Returns
    -------
    tp : int
        True positive.
    fp : int
        False positive.
    tn : int
        True positive.
    fn : int
        False negative.
    """

    tp = np.sum((fx == True) & (obs == True))    # True Positive (TP)
    fp = np.sum((fx == True) & (obs == False))   # False Positive (FP)
    tn = np.sum((fx == False) & (obs == False))  # True Negative (TN)
    fn = np.sum((fx == False) & (obs == True))   # False Negative (FN)

    return tp, fp, tn, fn


def probability_of_detection(obs, fx):
    """Probability of Detection (POD).

    .. math:: \\text{POD} = \\text{TP} / (\\text{TP} + \\text{FN})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values.

    Returns
    -------
    pod : float
        The POD of the forecast.

    """

    tp, fp, tn, fn = _event2count(obs, fx)
    return tp / (tp + fn)


def false_alarm_ratio(obs, fx):
    """False Alarm Ratio (FAR).

    .. math:: \\text{FAR} = \\text{FP} / (\\text{TP} + \\text{FP})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values.

    Returns
    -------
    far : float
        The FAR of the forecast.

    """
    tp, fp, tn, fn = _event2count(obs, fx)
    return fp / (tp + fp)


def probability_of_false_detection(obs, fx):
    """Probability of False Detection (POFD).

    .. math:: \\text{POFD} = \\text{FP} / (\\text{FP} + \\text{TN})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values.

    Returns
    -------
    pofd : float
        The POFD of the forecast.

    """

    tp, fp, tn, fn = _event2count(obs, fx)
    return fp / (fp + tn)


def critical_success_index(obs, fx):
    """Critical Success Index (CSI).

    .. math:: \\text{CSI} = \\text{TP} / (\\text{TP} + \\text{FN} + \\text{FN})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values.

    Returns
    -------
    csi : float
        The CSI of the forecast.

    """

    tp, fp, tn, fn = _event2count(obs, fx)
    return tp / (tp + fp + fn)


def event_bias(obs, fx):
    """Event Bias (EBIAS).

    .. math:: \\text{EBIAS} = (\\text{TP} + \\text{FP}) / (\\text{TP} + \\text{FN})

    Parameters
    ----------
    obs : (n,) array-like
        Observed event values (True=event, False=no event).
    fx : (n,) array-like
        Forecasted event values.

    Returns
    -------
    ebias : float
        The EBIAS of the forecast.

    """

    tp, fp, tn, fn = _event2count(obs, fx)
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
        Forecasted event values.

    Returns
    -------
    ea : float
        The EA of the forecast.

    """

    n = len(obs)
    tp, fp, tn, fn = _event2count(obs, fx)
    return (tp + tn) / n
