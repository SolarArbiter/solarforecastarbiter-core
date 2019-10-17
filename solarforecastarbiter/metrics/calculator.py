"""
Metric calculation functions.

Right now placeholder so we can delete report.metrics.py.
Needs to cleaned up and expanded.

Todo
----
* Support probabilistic metrics and forecasts with new functions
* Support event metrics and forecasts with new functions
"""
from collections import defaultdict

import pandas as pd

from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import deterministic


AVAILABLE_CATEGORIES = [
    'total', 'year', 'month', 'day', 'hour', 'date', 'weekday'
]


def calculate_metrics(processed_pairs, categories, metrics,
                      ref_pair=None, normalizer=1.0):
    """
    Loop through the forecast-observation pairs and calculate metrics.

    Parameters
    ----------
    processed_pairs :
        List of solarforecastarbiter.datamodel.ProcessedForecastObservation.
    categories : list of str
        List of categories to compute metrics over.
    metrics : list of str
        List of metrics to be computed.
    ref_fx_obs :
        solarforecastarbiter.datamodel.ProcessedForecastObservation`
        Reference forecast to be used when calculating skill metrics. Default
        is None and no skill metrics will be calculated.
    normalizer : float
        Normalized factor (should be in the same units as data). Default is
        1.0 and only needed for normalized metrics.

    Returns
    -------
    dict
        List of pd.DataFrame/dict with the results.
        Keys are ProcessedForecastObservation and values are pd.DataFrame?

    Todo
    ----
    * validate categories are supported
    * validate metrics are supported
    * Support probabilistic metrics and forecasts
    * Support event metrics and forecasts
    """
    calc_metrics = {}

    for fxobs in processed_pairs:

        # Deterministic
        if isinstance(fxobs, datamodel.ProcessedForecastObservation):
            metrics_ = calculate_deterministic_metrics(fxobs,
                                                       categoires,
                                                       metrics,
                                                       ref_fx_obs=ref_pair,
                                                       normalizer=normalizer)
            calc_metrics[fxobs] = metrics_

    return calc_metrics


def _apply_deterministic_metric_func(metric, fx, obs, **kwargs):
    """Helper function to deal with variable number of arguments possible for
    metric functions. """
    metric_func = deterministic._MAP[metric]
    if metric in deterministic._REQ_REF_FX:
        return metric_func(obs, fx, kwargs['ref_fx'])
    elif metric in deterministic._REQ_NORM:
        return metric_func(obs, fx, kwargs['normalizer'])
    else:
        return metric_func(obs, fx)


def calculate_deterministic_metrics(processed_fx_obs, categories, metrics,
                                    ref_fx_obs=None, normalizer=1.0):
    """
    Calculate deterministic metrics for the processed data using the provided
    categories and metric types.

    Parameters
    ----------
    processed_fx_obs :
        solarforecastarbiter.datamodel.ProcessedForecastObservation
    categories : list of str
        List of categories to compute metrics over.
    metrics : list of str
        List of metrics to be computed.
    ref_fx_obs :
        solarforecastarbiter.datamodel.ProcessedForecastObservation
        Reference forecast to be used when calculating skill metrics. Default
        is None and no skill metrics will be calculated.
    normalizer : float
        Normalized factor (should be in the same units as data). Default is
        1.0 and only needed for normalized metrics.

    Returns
    -------
    pd.DataFrame or dict:
        Contains all the computed metrics by categories.
        Structure is:
        1. Category type as tuple (e.g., ('total'), ('month', 'hour'))
        2. Metric name (e.g., 'mae', 'rmse')
        3. Category group (e.g, 0, 1, 2 ..., 11 for month)
        4. Value
        If no forecast data is found an empty dictionary is returned.
    """
    calc_metrics = defaultdict(dict)
    fx = processed_fx_obs.forecast_values
    obs = processed_fx_obs.observation_values

    # Check reference forecast is from processed pair, if needed
    ref_fx = None
    if any(m in deterministic._REQ_REF_FX for m in metrics):
        ref_fx = ref_fx_obs.forecast_values

    # No forecast data or metrics
    if fx.empty or len(metrics) == 0:
        return calc_metrics

    # Calculate metrics
    for category in set(categories):
        calc_metrics[category] = {}

        # total (special category)
        if category == 'total':
            for metric_ in metrics:
                r = _apply_deterministic_metric_func(metric_, fx, obs,
                                                     ref_fx=ref_fx,
                                                     normalizer=normalizer)
                calc_metrics[category][metric_] = r
        else:
            # dataframe for grouping
            df = pd.concat({'forecast': fx,
                            'observation': obs,
                            'reference': ref_fx}, axis=1)
            index_category = getattr(df.index, category)

            for name, group in df.groupby(index_category):
                calc_metrics[category][name] = {}

                for metric_ in metrics:
                    r = _apply_deterministic_metric_func(metric_, fx, obs,
                                                         ref_fx=ref_fx,
                                                         normalizer=normalizer)
                    calc_metrics[category][name][metric_] = r

    return calc_metrics
