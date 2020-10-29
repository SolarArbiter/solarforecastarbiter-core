"""
Metric calculation functions.
"""
import calendar
from functools import partial
import logging

import numpy as np
import pandas as pd


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import (
    deterministic, probabilistic, event, summary)


logger = logging.getLogger(__name__)


def calculate_metrics(processed_pairs, categories, metrics):
    """
    Loop through the forecast-observation pairs and calculate metrics.

    Normalization is determined by the attributes of the input objects.

    If ``processed_fx_obs.uncertainty`` is not ``None``, a deadband
    equal to the uncertainty will be used by the metrics that support it.

    Parameters
    ----------
    processed_pairs :
        List of datamodel.ProcessedForecastObservation
    categories : list of str
        List of categories to compute metrics over.
    metrics : list of str
        List of metrics to be computed.

    Returns
    -------
    list
        List of datamodel.MetricResult for each
        datamodel.ProcessedForecastObservation
    """
    calc_metrics = []

    for proc_fxobs in processed_pairs:

        # determine type of metrics to calculate
        if isinstance(proc_fxobs.original.forecast,
                      (datamodel.ProbabilisticForecast,
                       datamodel.ProbabilisticForecastConstantValue)):
            try:
                calc_metrics.append(calculate_probabilistic_metrics(
                    proc_fxobs,
                    categories,
                    metrics))
            except (RuntimeError, ValueError) as e:
                logger.error('Failed to calculate probabilistic metrics'
                             ' for %s: %s', proc_fxobs.name, e)
        elif isinstance(proc_fxobs.original.forecast, datamodel.EventForecast):
            try:
                calc_metrics.append(calculate_event_metrics(
                    proc_fxobs, categories, metrics
                ))
            except RuntimeError as e:
                logger.error('Failed to calculate event metrics for %s: %s',
                             proc_fxobs.name, e)
        else:
            try:
                calc_metrics.append(calculate_deterministic_metrics(
                    proc_fxobs,
                    categories,
                    metrics))
            except RuntimeError as e:
                logger.error('Failed to calculate deterministic metrics'
                             ' for %s: %s',
                             proc_fxobs.name, e)

    return calc_metrics


def _apply_deterministic_metric_func(metric, fx, obs, **kwargs):
    """Helper function to deal with variable number of arguments possible for
    deterministic metric functions."""
    metric_func = deterministic._MAP[metric][0]

    # the keyword arguments that will be passed to the functions
    _kw = {}

    # deadband could be supplied as None, so this is a little cleaner
    # than a try/except pattern
    deadband = kwargs.get('deadband', None)
    if metric in deterministic._DEADBAND_ALLOWED and deadband:
        # metrics assumes fractional deadband, datamodel assumes %, so / 100
        deadband_frac = deadband / 100
        _kw['error_fnc'] = partial(
            deterministic.error_deadband, deadband=deadband_frac)

    if metric == 'cost':
        _kw['cost_params'] = kwargs['cost_params']

    # ref is an arg, but seems cleaner to handle as a kwarg here
    if metric in deterministic._REQ_REF_FX:
        _kw['ref'] = kwargs['ref_fx']

    # same arg/kwarg comment as for ref
    if metric in deterministic._REQ_NORM:
        _kw['norm'] = kwargs['normalization']

    return metric_func(obs, fx, **_kw)


def _apply_probabilistic_metric_func(metric, fx, fx_prob, obs, **kwargs):
    """Helper function to deal with variable number of arguments possible for
    probabilistic metric functions."""
    metric_func = probabilistic._MAP[metric][0]
    if metric in probabilistic._REQ_REF_FX:
        return metric_func(obs, fx, fx_prob,
                           kwargs['ref_fx'], kwargs['ref_fx_prob'])
    else:
        return metric_func(obs, fx, fx_prob)


def _apply_event_metric_func(metric, fx, obs, **kwargs):
    """Helper function to deal with variable number of arguments possible for
    event metric functions."""
    metric_func = event._MAP[metric][0]
    return metric_func(obs, fx)


def calculate_deterministic_metrics(processed_fx_obs, categories, metrics):
    """
    Calculate deterministic metrics for the processed data using the provided
    categories and metric types.

    Normalization is determined by the attributes of the input objects.

    If ``processed_fx_obs.uncertainty`` is not ``None``, a deadband
    equal to the uncertainty will be used by the metrics that support it.

    Parameters
    ----------
    processed_fx_obs : datamodel.ProcessedForecastObservation
    categories : list of str
        List of categories to compute metrics over.
    metrics : list of str
        List of metrics to be computed.

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
        out['observation_id'] = processed_fx_obs.original.observation.observation_id  # NOQA: E501
    except AttributeError:
        out['aggregate_id'] = processed_fx_obs.original.aggregate.aggregate_id

    fx = processed_fx_obs.forecast_values
    obs = processed_fx_obs.observation_values

    # Check reference forecast is from processed pair, if needed
    ref_fx = processed_fx_obs.reference_forecast_values
    if ref_fx is None:
        ref_fx = np.nan  # avoids issues with None and deadband masking

    # No data or metrics
    if fx.empty:
        raise RuntimeError("No forecast timeseries data.")
    elif obs.empty:
        raise RuntimeError("No observation timeseries data.")
    elif len(metrics) == 0:
        raise RuntimeError("No metrics specified.")

    # Dataframe for grouping
    df = pd.DataFrame({'forecast': fx,
                       'observation': obs,
                       'reference': ref_fx})

    # get normalization factor
    normalization = processed_fx_obs.normalization_factor

    # get uncertainty.
    deadband = processed_fx_obs.uncertainty

    cost_params = processed_fx_obs.cost

    # Force `groupby` to be consistent with `interval_label`, i.e., if
    # `interval_label == ending`, then the last interval should be in the bin
    if processed_fx_obs.interval_label == "ending":
        df.index -= pd.Timedelta("1ns")

    metric_vals = []
    # Calculate metrics
    for category in set(categories):
        index_category = _index_category(category, df)

        # Calculate each metric
        for metric_ in metrics:

            # Group by category
            for cat, group in df.groupby(index_category):

                # Calculate
                res = _apply_deterministic_metric_func(
                    metric_, group.forecast, group.observation,
                    ref_fx=group.reference, normalization=normalization,
                    deadband=deadband, cost_params=cost_params)

                # Change category label of the group from numbers
                # to e.g. January or Monday
                if category == 'month':
                    cat = calendar.month_abbr[cat]
                elif category == 'weekday':
                    cat = calendar.day_abbr[cat]

                metric_vals.append(datamodel.MetricValue(
                    category, metric_, str(cat), res))

    out['values'] = _sort_metrics_vals(
        metric_vals, datamodel.ALLOWED_DETERMINISTIC_METRICS)
    calc_metrics = datamodel.MetricResult.from_dict(out)
    return calc_metrics


def _sort_metrics_vals(metrics_vals, mapping):
    """
    Parameters
    ----------
    metrics_vals : list
        Elements are datamodel.MetricValue
    mapping : dict
        Metrics mapping defined in datamodel.

    Returns
    -------
    tuple
        Sorted elements. Sorting order is:
            * Category
            * Metric
            * Index
            * Value
    """
    metric_ordering = list(mapping.keys())
    category_ordering = list(datamodel.ALLOWED_CATEGORIES.keys())

    def sorter(metricval):
        if metricval.category == 'month':
            index_order = calendar.month_abbr[0:13].index(metricval.index)
        elif metricval.category == 'weekday':
            index_order = calendar.day_abbr[0:7].index(metricval.index)
        else:
            index_order = metricval.index

        return (
            category_ordering.index(metricval.category),
            metric_ordering.index(metricval.metric),
            index_order,
            metricval.value
            )

    metrics_sorted = tuple(sorted(metrics_vals, key=sorter))
    return metrics_sorted


def calculate_probabilistic_metrics(processed_fx_obs, categories, metrics):
    """
    Calculate probabilistic metrics for the processed data using the provided
    categories and metric types. Will calculate distribution metrics (e.g.,
    Continuous Ranked Probability Score (CRPS)) over all constant values if
    forecast is a datamodel.ProbabilisticForecast is provided, or single value
    metrics (e.g., Briar Score (BS)) if forecast is a
    datamodel.ProbabilisticForecastConstantValue.

    Parameters
    ----------
    processed_fx_obs : datamodel.ProcessedForecastObservation
        Forecasts must be datamodel.ProbabilisticForecast (CDFs) or
        datamodel.ProbabilisticForecastConstantValue (single values)
    categories : list of str
        List of categories to compute metrics over.
    metrics : list of str
        List of metrics to be computed.

    Returns
    -------
    datamodel.MetricsResult

    Raises
    ------
    RuntimeError
        If there is no forecast, observation timeseries data or no metrics
        are specified.
    ValueError
        If original and reference forecast ``axis`` values do not match.
    """
    out = {'name': processed_fx_obs.name}
    try:
        obs_dict = {'observation_id':
                    processed_fx_obs.original.observation.observation_id}
    except AttributeError:
        obs_dict = {'aggregate_id':
                    processed_fx_obs.original.aggregate.aggregate_id}
    out.update(obs_dict)

    # Determine forecast physical and probability values by axis
    fx_fx_prob = _transform_prob_forecast_value_and_prob(processed_fx_obs)
    obs = processed_fx_obs.observation_values

    if processed_fx_obs.reference_forecast_values is None:
        ref_fx_fx_prob = [(np.nan, np.nan)]*len(fx_fx_prob)
    else:
        if (processed_fx_obs.original.forecast.axis !=
                processed_fx_obs.original.reference_forecast.axis):
            # could put this check in datamodel (and others from preprocessing)
            raise ValueError("Mismatched `axis` between "
                             "forecast and reference forecast.")
        ref_fx_fx_prob = _transform_prob_forecast_value_and_prob(
            processed_fx_obs, attr='reference_forecast_values')
        out.update({'reference_forecast_id': processed_fx_obs.original.reference_forecast.forecast_id})  # NOQA: E501

    # No data or metrics
    if (not fx_fx_prob or
            any([fx[0].empty or fx[1].empty for fx in fx_fx_prob])):
        raise RuntimeError("Missing probabilistic forecast timeseries data.")
    elif obs.empty:
        raise RuntimeError("No observation timeseries data.")
    elif len(metrics) == 0:
        raise RuntimeError("No metrics specified.")

    fx = processed_fx_obs.original.forecast
    out['forecast_id'] = fx.forecast_id

    # Determine proper metrics by instance type
    if isinstance(fx, datamodel.ProbabilisticForecastConstantValue):
        _metrics = list(set(metrics) - set(probabilistic._REQ_DIST))
    elif isinstance(fx, datamodel.ProbabilisticForecast):
        _metrics = list(set(metrics) & set(probabilistic._REQ_DIST))

    # Compute metrics and create MetricResult
    results = _calculate_probabilistic_metrics_from_df(
        _create_prob_dataframe(obs, fx_fx_prob, ref_fx_fx_prob),
        categories,
        _metrics,
        processed_fx_obs.original.forecast.interval_label)
    out['values'] = _sort_metrics_vals(
        results, datamodel.ALLOWED_PROBABILISTIC_METRICS)
    result = datamodel.MetricResult.from_dict(out)
    return result


def _calculate_probabilistic_metrics_from_df(data_df, categories, metrics,
                                             interval_label):
    """
    Calculate probabilistic metrics for the processed data using the provided
    categories and metric types.

    Parameters
    ----------
    data_df : pandas.DataFrame
        DataFrame that contains all timeseries values on the same index.
    categories : list of str
        List of categories to compute metrics over.
    metrics : list of str
        List of metrics to be computed.
    interval_label : str

    Returns
    -------
    list of tuples of datamodel.MetricValue
        Contains all the computed metrics by categories. Each tuple is
        associated with a datamodel.ProbabilisticForecastConstantValue.
    """
    metric_values = []

    # Force `groupby` to be consistent with `interval_label`, i.e., if
    # `interval_label == ending`, then the last interval should be in the
    # bin
    if interval_label == "ending":
        data_df.index -= pd.Timedelta("1ns")

    # Calculate metrics
    for category in set(categories):
        index_category = _index_category(category, data_df)

        # Calculate each metric
        for metric_ in set(metrics):
            # Group by category
            for cat, group in data_df.groupby(index_category):

                try:
                    ref_fx_vals = group.xs('reference_forecast', level=1, axis=1).to_numpy()  # NOQA: E501
                    ref_fx_prob_vals = group.xs('reference_probability', level=1, axis=1).to_numpy()  # NOQA E501
                    if ref_fx_vals.size == ref_fx_vals.shape[0]:
                        ref_fx_vals = ref_fx_vals.T[0]
                        ref_fx_prob_vals = ref_fx_prob_vals.T[0]
                except KeyError:
                    ref_fx_vals = np.nan
                    ref_fx_prob_vals = np.nan

                fx_vals = group.xs('forecast', level=1, axis=1).to_numpy()
                fx_prob_vals = group.xs('probability', level=1, axis=1).to_numpy()  # NOQA: E501
                if fx_vals.size == fx_vals.shape[0]:
                    fx_vals = fx_vals.T[0]
                    fx_prob_vals = fx_prob_vals.T[0]
                obs_vals = group[(None, 'observation')].to_numpy()

                # Calculate
                res = _apply_probabilistic_metric_func(
                    metric_, fx_vals, fx_prob_vals, obs_vals,
                    ref_fx=ref_fx_vals, ref_fx_prob=ref_fx_prob_vals)

                # Change category label of the group from numbers
                # to e.g. January or Monday
                if category == 'month':
                    cat = calendar.month_abbr[cat]
                elif category == 'weekday':
                    cat = calendar.day_abbr[cat]

                metric_values.append(datamodel.MetricValue(
                    category, metric_, str(cat), res))

    return metric_values


def _create_prob_dataframe(obs, fx_fx_prob, ref_fx_fx_prob):
    """
    Creates a DataFrame for grouping for the probabilistic forecasts.

    Parameters
    ----------
    obs : pandas.Series
    fx_fx_prob : list of tuple of pandas.Series
    ref_fx_fx_prob : list of tuple of pandas.Series

    Returns
    -------
    pandas.DataFrame
        DataFrame combining all data with column names being tuples of
        constant_value and one of 'observation', 'forecast', 'probability',
        'reference_forecast' or 'reference_probability'. For 'observation' the
        constant_value is None.
    """
    data_dict = {(None, 'observation'): obs}

    for orig, ref in zip(fx_fx_prob, ref_fx_fx_prob):
        fx, fx_prob = orig
        ref_fx, ref_fx_prob = ref
        data_dict[(fx.name, 'forecast')] = fx
        data_dict[(fx_prob.name, 'probability')] = fx_prob
        try:
            data_dict[(ref_fx.name, 'reference_forecast')] = ref_fx
            data_dict[(ref_fx_prob.name, 'reference_probability')] = ref_fx_prob  # NOQA: E501
        except AttributeError:
            pass

    return pd.DataFrame(data_dict)


def _transform_prob_forecast_value_and_prob(proc_fx_obs,
                                            attr='forecast_values'):
    """
    Helper function that returns ordered list of series of physical values and
    probabilities for a datamodel.ProcessedForecastObservation

    Parameters
    ----------
    proc_fx_obs : datamodel.ProcessedForecastObservation
    attr : str
        The attribute of proc_fx_obs that contains the forecast value of
        interest. Typically forecast_values or
        reference_forecast_values.

    Returns
    -------
    fx_fx_prob : list of tuples of (n,) array_like
        Forecast physical values and forecast probabilities.
    """
    fx_fx_prob = []
    data = getattr(proc_fx_obs, attr)

    # Need to convert to DataFrame of ProbabilisticForecastConstantValue
    # for consistency with ProbabilisticForecas that provides a DataFrame
    if isinstance(data, pd.Series):
        # set the name to a float because it will default to 'value'
        data = data.to_frame(
            name=proc_fx_obs.original.forecast.constant_value)

    for col in data.columns:
        if proc_fx_obs.original.forecast.axis == 'x':
            fx_prob = data[col]
            fx = pd.Series([float(col)]*fx_prob.size, index=fx_prob.index,
                           name=col)
        else:  # 'y'
            fx = data[col]
            fx_prob = pd.Series([float(col)]*fx.size, index=fx.index,
                                name=col)
        fx_fx_prob.append((fx, fx_prob))
    return fx_fx_prob


def calculate_event_metrics(proc_fx_obs, categories, metrics):
    """
    Calculate event metrics for the processed data using the provided
    categories and metric types.

    Parameters
    ----------
    proc_fx_obs : datamodel.ProcessedForecastObservation
    categories : list of str
        List of categories to compute metrics over.
    metrics : list of str
        List of metrics to be computed.

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
        'name': proc_fx_obs.name,
        'forecast_id': proc_fx_obs.original.forecast.forecast_id,
        'observation_id': proc_fx_obs.original.observation.observation_id
    }

    fx = proc_fx_obs.forecast_values
    obs = proc_fx_obs.observation_values

    # No data or metrics
    if fx.empty:
        raise RuntimeError("No Forecast timeseries data.")
    elif obs.empty:
        raise RuntimeError("No Observation timeseries data.")
    elif len(metrics) == 0:
        raise RuntimeError("No metrics specified.")

    # Dataframe for grouping
    df = pd.concat({'forecast': fx, 'observation': obs}, axis=1)

    metric_vals = []
    # Calculate metrics
    for category in set(categories):
        index_category = _index_category(category, df)

        # Calculate each metric
        for metric_ in set(metrics):
            # Group by category
            for cat, group in df.groupby(index_category):

                # Calculate
                res = _apply_event_metric_func(
                    metric_, group.forecast, group.observation
                )

                # Change category label of the group from numbers
                # to e.g. January or Monday
                if category == 'month':
                    cat = calendar.month_abbr[cat]
                elif category == 'weekday':
                    cat = calendar.day_abbr[cat]

                metric_vals.append(datamodel.MetricValue(
                    category, metric_, str(cat), res))

    out['values'] = _sort_metrics_vals(metric_vals,
                                       datamodel.ALLOWED_EVENT_METRICS)
    calc_metrics = datamodel.MetricResult.from_dict(out)
    return calc_metrics


def _index_category(category, df):
    # total (special category)
    if category == 'total':
        index_category = lambda x: 0  # NOQA: E731
    elif category == 'season':
        index_category = _season_from_months(df.index.month)
    else:
        index_category = getattr(df.index, category)
    return index_category


def _season_from_months(months):
    """Compute season (DJF, MAM, JJA, SON) from month ordinal"""
    # Copied from xarray. see xarray license in LICENSES
    seasons = np.array(["DJF", "MAM", "JJA", "SON"])
    months = np.asarray(months)
    return seasons[(months // 3) % 4]


def _is_deterministic_forecast(proc_fxobs):
    return type(proc_fxobs.original.forecast) is datamodel.Forecast


def _calculate_summary_for_frame(df):
    return df.agg(list(summary._MAP.keys()))


def calculate_summary_statistics(processed_fx_obs, categories):
    """
    Calculate summary statistics for the processed data using the provided
    categories and all metrics defined in :py:mod:`.summary`.

    Parameters
    ----------
    proc_fx_obs : datamodel.ProcessedForecastObservation
    categories : list of str
        List of categories to compute metrics over.

    Returns
    -------
    solarforecastarbiter.datamodel.MetricResult
        Contains all the computed statistics by category.

    Raises
    ------
    RuntimeError
        If there is no data to summarize
    """
    out = {'name': processed_fx_obs.name,
           'forecast_id': processed_fx_obs.original.forecast.forecast_id,
           'is_summary': True}
    try:
        out['observation_id'] = \
            processed_fx_obs.original.observation.observation_id
    except AttributeError:
        out['aggregate_id'] = \
            processed_fx_obs.original.aggregate.aggregate_id

    dfd = {'observation': processed_fx_obs.observation_values}
    # only calculate stats for deterministic forecasts
    # but always for observations
    if _is_deterministic_forecast(processed_fx_obs):
        dfd['forecast'] = processed_fx_obs.forecast_values
        ref_fx = processed_fx_obs.reference_forecast_values
        if ref_fx is not None:
            dfd['reference_forecast'] = ref_fx

    df = pd.DataFrame(dfd)
    if df.empty:
        raise RuntimeError('No data to calculate summary statistics for.')

    # Force `groupby` to be consistent with `interval_label`, i.e., if
    # `interval_label == ending`, then the last interval should be in the bin
    if processed_fx_obs.interval_label == "ending":
        df.index -= pd.Timedelta("1ns")

    metric_vals = []
    # Calculate metrics
    for category in set(categories):
        index_category = _index_category(category, df)

        # Group by category
        for cat, group in df.groupby(index_category):
            all_metrics = _calculate_summary_for_frame(group)

            # Change category label of the group from numbers
            # to e.g. January or Monday
            if category == 'month':
                cat = calendar.month_abbr[cat]
            elif category == 'weekday':
                cat = calendar.day_abbr[cat]

            metric_vals.extend([
                datamodel.MetricValue(category, f'{obj}_{met}', str(cat), val)
                for obj, ser in all_metrics.items() for met, val in ser.items()
            ])
    out['values'] = _sort_metrics_vals(
        metric_vals,
        {f'{type_}_{k}': v
         for k, v in datamodel.ALLOWED_SUMMARY_STATISTICS.items()
         for type_ in ('forecast', 'observation', 'reference_forecast')})
    calc_stats = datamodel.MetricResult.from_dict(out)
    return calc_stats


def calculate_all_summary_statistics(processed_pairs, categories):
    """
    Loop through the forecast-observation pairs and calculate summary
    statistics.

    Parameters
    ----------
    processed_pairs :
        List of datamodel.ProcessedForecastObservation
    categories : list of str
        List of categories to compute metrics over.

    Returns
    -------
    list
        List of datamodel.MetricResult for each
        datamodel.ProcessedForecastObservation
    """
    stats = []
    for proc_fxobs in processed_pairs:
        try:
            stats.append(calculate_summary_statistics(proc_fxobs, categories))
        except RuntimeError as e:
            logger.error('Failed to calculate summary statistics'
                         ' for %s: %s', proc_fxobs.name, e)
    return stats
