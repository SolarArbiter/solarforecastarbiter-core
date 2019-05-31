"""
Provides evaluation of observations to forecasts.

Provides tools for evaluation of forecast performance from a given metrics'
context :py:mod:`solarforecastarbiter.metrics.context` and returns the results
:py:mod:`solarforecastarbiter.metrics.results`.
"""

import numpy as np
import pandas as pd

from solarforecastarbiter.metrics import context, results, errors


def evaluate(observations, forecasts, metrics_context):
    """
    Evaluate the performance of the forecasts to the observations using the
    context to define the preprocessing, metrics and results to returnself.

    Parameters
    ----------
    observations : pd.DataFrame
        observations are a DataFrame of values and quality flags with timestamp
        as index
    forecasts : pd.Series
        forecasts are a Series of values with timestamp as index
    metrics_context : dict
        a context dictionary as defined in
        :py:mod:`solarforecastarbiter.metrics.context`

    Returns
    -------
    dict
        a results dictionary as defined in
        :py:mod:`solarforecastarbiter.metrics.results`

    Raises
    ------
    SfaMetricsInputError : if invalid input data.
    """
    # Create empty results
    result = results.EVALUATOR_RESULTS

    # Verify input
    if not are_valid_observations(observations):
        raise errors.SfaMetricsInputError("Observations must be a pandas DataFrame \
                                          with value and quality_flag columns \
                                          and an index of datetimes.")

    if not are_valid_forecasts(forecasts):
        raise errors.SfaMetricsInputError("Forecasts must be a pandas Series \
                                          with value \
                                          and an index of datetimes")

    # TODO:
    # - enforce interval consistency
    # - replace this with mapping to preprocessing functions and add decorators

    # Preprocessing observations
    obs_context = metrics_context['preprocessing']['observations']
    obs_method = obs_context['fill_method']
    if obs_method in context.supported_fill_functions():
        fill_func = context._FILL_FUNCTIONS_MAP[obs_method]
        obs_values = fill_func(observations.value, observations.quality_flag)
    else:
        obs_values = observations.values

    # Preprocessing forecasts
    fx_context = metrics_context['preprocessing']['forecasts']
    fx_method = fx_context['fill_method']
    if fx_method in context.supported_fill_functions():
        fill_func = context._FILL_FUNCTIONS_MAP[fx_method]
        fx_values = fill_func(forecasts)
    else:
        fx_values = forecasts

    # Copy preprocessed timeseries to result
    if metrics_context['results']['timeseries']['observations']:
        result['timeseries']['observations'] = obs_values

    if metrics_context['results']['timeseries']['forecasts']:
        result['timeseries']['forecasts'] = fx_values

    # Calculate metrics
    for metric, do_calc in metrics_context['metrics'].items():
        if do_calc and metric in context.supported_metrics():

            # For all
            metric_func = context._METRICS_MAP[metric]
            result['metrics']['total'][metric] = metric_func(obs_values,
                                                             fx_values)

            # Group by calculations
            group_context = metrics_context['results']['groupings']
            for groupby, do_group in group_context.items():
                if do_group and groupby in context.supported_groupings():

                    if groupby not in result['metrics'].keys():
                        result['metrics'][groupby] = {}

                    result['metrics'][groupby][metric] = \
                        evaluate_by_group(obs_values,
                                          fx_values,
                                          groupby,
                                          metric_func)

    return result


def evaluate_by_group(observation, forecast, groupby, metric_func):
    """Evaluate the performance according to the groupby.

    Parameters
    ----------
    observation : pd.Series
    forecast : pd.Series
    groupby : string
        one of the supported types of groupings
    metric_func : function
        a py:func:`solarforecastarbiter.metrics` metrics function

    Returns
    -------
    pd.Series :

    Raises
    ------
    SfaMetricsConfigError if `groupby` is not a supported groupby type.
    """
    result = None

    # Verify timestamps
    assert np.array_equal(observation.index, forecast.index)

    # Group into single DataFrame
    df = pd.DataFrame({'observation': observation,
                       'forecast': forecast},
                      index=observation.index)

    # Determine proper groupby
    if groupby == 'month':
        df_group = df.groupby(df.index.month)
    elif groupby == 'weekday':
        df_group = df.groupby(df.index.weekday)
    elif groupby == 'hour':
        df_group = df.groupby(df.index.hour)
    elif groupby == 'date':
        df_group = df.groupby(df.index.date)
    else:
        raise errors.SfaMetricsConfigError(
            f"No supported groupby type {groupby}.")

    # Calculate metrics for each group
    result = df_group.apply(
        lambda x: metric_func(x['observation'], x['forecast']))

    return result


def are_valid_observations(observations):
    """Validate observations in expected format."""
    if not isinstance(observations, pd.DataFrame):
        return False
    if not ['value', 'quality_flag'] == list(observations.columns):
        return False
    if not observations.index.is_all_dates:
        return False
    return True


def are_valid_forecasts(forecasts):
    """Validate forecasts are in expected format."""
    if not isinstance(forecasts, pd.Series):
        return False
    if not forecasts.index.is_all_dates:
        return False
    return True
