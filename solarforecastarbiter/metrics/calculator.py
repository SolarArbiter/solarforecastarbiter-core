"""
Metric calculation functions.

Right now placeholder so we can delete report.metrics.py.
Needs to cleaned up and expanded.

Todo
----
* Support probabilistic metrics and forecasts with new functions
* Support event metrics and forecasts with new functions
"""
import calendar

import pandas as pd

from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import deterministic


# Keys are the category names, values are the mapping used to groupby
AVAILABLE_CATEGORIES = {
    'Total': 'total',
    'Year': 'year',
    'Month of the year': 'month',
    'Day of the month': 'day',
    'Hour of the day': 'hour',
    'Date': 'date',
    'Day of the week': 'weekday',
}


def calculate_metrics(processed_pairs, categories, metrics,
                      ref_pair=None, normalizer=1.0):
    """
    Loop through the forecast-observation pairs and calculate metrics.

    Parameters
    ----------
    processed_pairs :
        List of solarforecastarbiter.datamodel.ProcessedForecastObservation
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
    list
        List of dict with the metric results and a key with original forecast
        name for identification.

    Todo
    ----
    * validate categories are supported
    * validate metrics are supported
    * Support probabilistic metrics and forecasts
    * Support event metrics and forecasts
    """
    calc_metrics = []

    for proc_fxobs in processed_pairs:

        # TODO: support ProbabilisticForecast
        if isinstance(proc_fxobs.original.forecast,
                      datamodel.ProbabilisticForecast):
            raise NotImplementedError
        else:
            # calculate_deterministic_metrics
            metrics_ = calculate_deterministic_metrics(proc_fxobs,
                                                       categories,
                                                       metrics,
                                                       ref_fx_obs=ref_pair,
                                                       normalizer=normalizer)
            calc_metrics.append(metrics_)

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
    processed_fx_obs : datamodel.ProcessedForecastObservation
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

          * dictionary of forecast 'name' and category types as tuple
            (e.g., ('Total'), )
          * dictionary with key of metric type (e.g., 'mae', 'rmse')
          * values of pandas.Series with Index of category values

    Raises
    ------
    RuntimeError
        If there is no forecast, obersvation timeseries data or no metrics
        are specified.

    """
    calc_metrics = {}
    calc_metrics['name'] = processed_fx_obs.original.forecast.name

    fx = processed_fx_obs.forecast_values
    obs = processed_fx_obs.observation_values

    # Check reference forecast is from processed pair, if needed
    ref_fx = None
    if any(m in deterministic._REQ_REF_FX for m in metrics):
        ref_fx = ref_fx_obs.forecast_values

    # No data or metrics
    if fx.empty:
        raise RuntimeError("No Forecast timeseries data.")
    elif obs.empty:
        raise RuntimeError("No Observation timeseries data.")
    elif len(metrics) == 0:
        raise RuntimeError("No metrics specified.")

    # Dataframe for grouping
    df = pd.concat({'forecast': fx,
                    'observation': obs,
                    'reference': ref_fx}, axis=1)

    # Calculate metrics
    for category in set(categories):
        calc_metrics[category] = {}

        # total (special category)
        if category == 'Total':
            for metric_ in metrics:
                res = _apply_deterministic_metric_func(
                    metric_, fx, obs, ref_fx=ref_fx, normalizer=normalizer)
                calc_metrics[category][metric_] = res
        else:
            groupby_category = AVAILABLE_CATEGORIES[category]
            index_category = getattr(df.index, groupby_category)

            # Calculate each metric
            for metric_ in metrics:

                metric_values = []
                cat_values = []

                # Group by category
                for id, group in df.groupby(index_category):

                    # Calculate
                    res = _apply_deterministic_metric_func(
                        metric_, group.forecast, group.observation,
                        ref_fx=ref_fx, normalizer=normalizer)

                    # Change id of the group
                    if category == 'Month of the year':
                        id = calendar.month_abbr[id]
                    elif category == 'Day of the week':
                        id = calendar.day_abbr[id]

                    metric_values.append(res)
                    cat_values.append(id)

                calc_metrics[category][metric_] = pd.Series(metric_values,
                                                            index=cat_values)

    return calc_metrics
