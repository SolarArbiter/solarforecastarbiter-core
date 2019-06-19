"""
Provides evaluation of observations to forecasts.

Provides tools for evaluation of forecast performance from a given metrics'
context :py:mod:`solarforecastarbiter.metrics.context` and returns the results
:py:mod:`solarforecastarbiter.metrics.results`.
"""

import copy
import pytz
import itertools
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
    # Copies of timeseries
    obs_copy = copy.deepcopy(observations)
    fx_copy = copy.deepcopy(forecasts)

    # Verify input
    if not are_valid_observations(obs_copy):
        raise errors.SfaMetricsInputError("Observations must be a pandas DataFrame \
                                          with value and quality_flag columns \
                                          and an index of datetimes.")

    if not are_valid_forecasts(fx_copy):
        raise errors.SfaMetricsInputError("Forecasts must be a pandas Series \
                                          with value \
                                          and an index of datetimes")

    # Timezone aware and conversion of timeseries
    if metrics_context['timezone']:
        if not obs_copy.index.tz:
            obs_copy.index = obs_copy.index.tz_localize(pytz.utc)
        obs_copy.index = obs_copy.index.tz_convert(
            metrics_context['timezone'])

        if not fx_copy.index.tz:
            fx_copy.index = fx_copy.index.tz_localize(pytz.utc)
        fx_copy.index = fx_copy.index.tz_convert(
            metrics_context['timezone'])

    # TODO:
    # - enforce interval consistency
    # - replace this with mapping to preprocessing functions and add decorators

    # Preprocessing observations
    obs_context = metrics_context['preprocessing']['observations']
    obs_method = obs_context['fill_method']
    if obs_method in context.supported_fill_functions():
        fill_func = context._FILL_FUNCTIONS_MAP[obs_method]
        obs_values = fill_func(obs_copy.value, obs_copy.quality_flag)
    else:
        obs_values = obs_copy.value

    # Preprocessing forecasts
    fx_context = metrics_context['preprocessing']['forecasts']
    fx_method = fx_context['fill_method']
    if fx_method in context.supported_fill_functions():
        fill_func = context._FILL_FUNCTIONS_MAP[fx_method]
        fx_values = fill_func(fx_copy)
    else:
        fx_values = fx_copy

    # Force consistency of indexes
    obs_values.drop(obs_values.index.difference(fx_values.index),
                    axis=0,
                    inplace=True)
    fx_values.drop(fx_values.index.difference(obs_values.index),
                   axis=0,
                   inplace=True)

    # Create results
    result = results.MetricsResult(obs_values, fx_values, metrics_context)
    groups = create_groups_list_from_context(metrics_context)
    metrics = create_metrics_list_from_context(metrics_context)

    # Group into single DataFrame
    df = pd.DataFrame({'observation': obs_values,
                       'forecast': fx_values},
                      index=obs_values.index)

    group_combinations = [itertools.combinations(groups, i) 
                          for i in range(1,len(groups)+1)]
    group_combinations = list(itertools.chain(*group_combinations))  # flatten
    group_combinations.append('total')

    # Calculate metrics for each grouping combination
    for group in group_combinations:
        
        # Temp metrics DataFrame
        temp_df = pd.DataFrame()
        
        for metric in metrics:

            # Get metric function
            metric_func = context._METRICS_MAP[metric]

            # Calculate metrics
            if group != ('total'):
                indexed_groups = [getattr(df.index, g) for g in group]
                # HACK : half to convert date to nump.int64
                # in order to be consistent type with np.nan
                indexed_groups = [pd.to_datetime(a).astype(np.int64) 
                                  if a.dtype == object else a 
                                  for a in indexed_groups]
                df_grouped = df.groupby(indexed_groups)
                temp_temp_df = df_grouped.apply(
                    lambda x: pd.Series({metric: 
                        metric_func(x['observation'], x['forecast'])})
                )
                temp_temp_df.index.names = group
            else:
                # special case for total
                temp_temp_df = pd.DataFrame(data=\
                    {metric: [metric_func(df['observation'], df['forecast'])]},
                    index=pd.MultiIndex.from_product(
                        [[np.nan]]*len(groups),
                        names=groups)
                )

            # Concatenate metrics
            temp_df = pd.concat([temp_df, temp_temp_df], axis=1, sort=False)

        if group != ('total'):
            # Add nulls for missing groups and add to index
            misses = list(set(groups).difference(group))
            for missed_group in misses:
                temp_df[missed_group] = np.nan
                temp_df.set_index(missed_group, append=True)

        # Merge into MetricsResults.metrics
        if result.metrics is None:
            result.metrics = temp_df
        else:
            result.metrics = pd.merge(result.metrics.reset_index(),
                                      temp_df.reset_index(),
                                      how='outer').set_index(groups)

    return result


def evaluate_by_group(observation, forecast, groupby, metric_func):
    """
    Evaluate the performance according to the groupby.

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
    pd.Series

    Raises
    ------
    py:class:`solarforecastarbiter.metrics.SfaMetricsConfigError` if `groupby`
        is not a supported groupby type.
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


def create_groups_list_from_context(metrics_context):
    """Create list of datetime groups form a 
    `solarforecastarbiter.metrics.MetricsContext`
    """
    dt_groups = []
    for groupby, use in metrics_context['results']['groupings'].items():
        if use:
            dt_groups.append(groupby)

    # Reorder
    dt_groups = [x for x in results.PREF_METRIC_GROUP_ORDER if x in dt_groups]

    return dt_groups


def create_metrics_list_from_context(metrics_context):
    """Create list of metics groups form a 
    `solarforecastarbiter.metrics.MetricsContext`
    """
    metric_groups = []
    for metric, use in metrics_context['metrics'].items():
        if use:
            metric_groups.append(metric)

    return metric_groups
