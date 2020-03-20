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
import logging


import pandas as pd


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import deterministic


logger = logging.getLogger(__name__)


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
        List of solarforecastarbiter.datamodel.MetricResult

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
            method_ = calculate_probabilistic_metrics
        else:
            method_ = calculate_deterministic_metrics
        try:
            metrics_ = method_(
                proc_fxobs,
                categories,
                metrics,
                ref_fx_obs=ref_pair,
                normalizer=normalizer
            )
        except RuntimeError as e:
            logger.error('Failed to calculate metrics for %s: %s',
                         proc_fxobs.name, e)
        else:
            calc_metrics.append(metrics_)

    return calc_metrics


def _apply_deterministic_metric_func(metric, fx, obs, **kwargs):
    """Helper function to deal with variable number of arguments possible for
    metric functions. """
    metric_func = deterministic._MAP[metric][0]
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
    solarforecastarbiter.datamodel.MetricResult
        Contains all the computed metrics by categories.

    Raises
    ------
    RuntimeError
        If there is no forecast, observation timeseries data or no metrics
        are specified.

    """
    out = {
        'name': processed_fx_obs.name,
        'forecast_id': processed_fx_obs.original.forecast.forecast_id,
    }

    try:
        out['observation_id'] = processed_fx_obs.original.observation.observation_id  # NOQA
    except AttributeError:
        out['aggregate_id'] = processed_fx_obs.original.aggregate.aggregate_id

    fx = processed_fx_obs.forecast_values
    obs = processed_fx_obs.observation_values

    # Check reference forecast is from processed pair, if needed
    ref_fx = None
    if any(m in deterministic._REQ_REF_FX for m in metrics):
        if not ref_fx_obs:
            raise RuntimeError("No reference forecast provided but it is "
                               "required for desired metrics")

        ref_fx = ref_fx_obs.forecast_values
        out['reference_forecast_id'] = ref_fx_obs.original.forecast.forecast_id
        if ref_fx.empty:
            raise RuntimeError("No reference forecast timeseries data")
        elif ref_fx_obs.interval_label != processed_fx_obs.interval_label:
            raise ValueError("Mismatched `interval_label` between "
                             "observation and reference forecast.")

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

    # Force `groupby` to be consistent with `interval_label`, i.e., if
    # `interval_label == ending`, then the last interval should be in the bin
    if processed_fx_obs.interval_label == "ending":
        df.index -= pd.Timedelta("1ns")

    metric_vals = []
    # Calculate metrics
    for category in set(categories):
        # total (special category)
        if category == 'total':
            index_category = lambda x: 0  # NOQA
        else:
            index_category = getattr(df.index, category)

        # Calculate each metric
        for metric_ in set(metrics):
            # Group by category
            for cat, group in df.groupby(index_category):

                # Calculate
                res = _apply_deterministic_metric_func(
                    metric_, group.forecast, group.observation,
                    ref_fx=ref_fx, normalizer=normalizer)

                # Change category label of the group from numbers
                # to e.g. January or Monday
                if category == 'month':
                    cat = calendar.month_abbr[cat]
                elif category == 'weekday':
                    cat = calendar.day_abbr[cat]

                metric_vals.append(datamodel.MetricValue(
                    category, metric_, str(cat), res))

    out['values'] = tuple(metric_vals)
    calc_metrics = datamodel.MetricResult.from_dict(out)
    return calc_metrics


def calculate_probabilistic_metrics(processed_fx_obs, categories, metrics,
                                    ref_fx_obs=None, normalizer=1.0):
    """
    Calculate probabilistic metrics for the processed data using the provided
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
    solarforecastarbiter.datamodel.MetricResult (TBD)
        Contains all the computed metrics by categories.

    Raises
    ------
    RuntimeError
        If there is no forecast, observation timeseries data or no metrics
        are specified.

    """
    out = {
        'name': processed_fx_obs.name,
        'forecast_id': processed_fx_obs.original.forecast.forecast_id,
    }

    try:
        out['observation_id'] = processed_fx_obs.original.observation.observation_id  # NOQA
    except AttributeError:
        out['aggregate_id'] = processed_fx_obs.original.aggregate.aggregate_id

    fx = processed_fx_obs.forecast_values
    obs = processed_fx_obs.observation_values
