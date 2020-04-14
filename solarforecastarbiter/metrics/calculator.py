"""
Metric calculation functions.

Right now placeholder so we can delete report.metrics.py.
Needs to cleaned up and expanded.

Todo
----
* Support probabilistic metrics and forecasts with new functions
* Support event metrics and forecasts with new functions
"""
from functools import partial
import calendar
import logging
import copy

import pandas as pd


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import deterministic, probabilistic, event


logger = logging.getLogger(__name__)


def calculate_metrics(processed_pairs, categories, metrics,
                      ref_pair=None):
    """
    Loop through the forecast-observation pairs and calculate metrics.

    Normalization is determined by the attributes of the input objects.

    If ``processed_fx_obs.uncertainty`` is not ``None``, a deadband
    equal to the uncertainty will be used by the metrics that support it.

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

        # determine type of metrics to calculate
        if isinstance(proc_fxobs.original.forecast,
                      datamodel.ProbabilisticForecast):
            try:
                calc_metrics.append(calculate_probabilistic_metrics(
                    proc_fxobs,
                    categories,
                    metrics,
                    ref_fx_obs=ref_pair))
            except RuntimeError as e:
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
                    metrics,
                    ref_fx_obs=ref_pair))
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


def calculate_deterministic_metrics(processed_fx_obs, categories, metrics,
                                    ref_fx_obs=None):
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
    ref_fx_obs :
        solarforecastarbiter.datamodel.ProcessedForecastObservation
        Reference forecast to be used when calculating skill metrics. Default
        is None and no skill metrics will be calculated.

    Returns
    -------
    solarforecastarbiter.datamodel.MetricResult
        Contains all the computed metrics by categories.

    Raises
    ------
    RuntimeError
        If there is no forecast, observation timeseries data or no metrics
        are specified.
    ValueError
        If original and reference forecast `interval_label`s do not match.

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
                               "required for desired metrics.")

        ref_fx = ref_fx_obs.forecast_values
        out['reference_forecast_id'] = ref_fx_obs.original.forecast.forecast_id
        if ref_fx.empty:
            raise RuntimeError("No reference forecast timeseries data.")
        elif ref_fx_obs.interval_label != processed_fx_obs.interval_label:
            raise ValueError("Mismatched `interval_label` between "
                             "observation and reference forecast.")

    # No data or metrics
    if fx.empty:
        raise RuntimeError("No forecast timeseries data.")
    elif obs.empty:
        raise RuntimeError("No observation timeseries data.")
    elif len(metrics) == 0:
        raise RuntimeError("No metrics specified.")

    # Dataframe for grouping
    df = pd.concat({'forecast': fx,
                    'observation': obs,
                    'reference': ref_fx}, axis=1)

    # get normalization factor
    normalization = processed_fx_obs.normalization_factor

    # get uncertainty.
    deadband = processed_fx_obs.uncertainty

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
        for metric_ in metrics:
            # Group by category
            for cat, group in df.groupby(index_category):

                # Calculate
                res = _apply_deterministic_metric_func(
                    metric_, group.forecast, group.observation,
                    ref_fx=ref_fx, normalization=normalization,
                    deadband=deadband)

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
        return (
            category_ordering.index(metricval.category),
            metric_ordering.index(metricval.metric),
            metricval.index,
            metricval.value
            )

    metrics_sorted = tuple(sorted(metrics_vals, key=sorter))
    return metrics_sorted


def calculate_probabilistic_metrics(processed_fx_obs, categories, metrics,
                                    ref_fx_obs=None):
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
    ref_fx_obs : datamodel.ProcessedForecastObservation
        Reference forecast to be used when calculating skill metrics. Default
        is None and no skill metrics will be calculated.

    Returns
    -------
    tuple of (list of datamodel.MetricsResult, datamodel.MetricsResult)
        First value is a list of metric results calculated per
        datamodel.ProbabilisticForecastConstantValue, and
        the second value are the metrics calculated per
        datamodel.ProbabilisticForecast.

    Raises
    ------
    RuntimeError
        If there is no forecast, observation timeseries data or no metrics
        are specified.
    ValueError
        If original and reference forecast `interval_label` or `axis` values
        do not match.
    """
    single_cv_results = []
    dist_result = None

    shared_dict = {'name': processed_fx_obs.name}
    try:
        obs_dict = {'observation_id': processed_fx_obs.original.observation.observation_id}  # NOQA
    except AttributeError:
        obs_dict = {'aggregate_id': processed_fx_obs.original.aggregate.aggregate_id}  # NOQA
    shared_dict.update(obs_dict)

    # Determine forecast physical and probabiltity values by axis
    fx_fx_prob = _transform_prob_forecast_value_and_prob(processed_fx_obs)
    obs = processed_fx_obs.observation_values

    # Check reference forecast is from processed pair, if needed
    if any(m in probabilistic._REQ_REF_FX for m in metrics):
        if not ref_fx_obs:
            raise RuntimeError("No reference forecast provided but it is "
                               "required for desired metrics.")

        ref_fx_fx_prob = _transform_prob_forecast_value_and_prob(ref_fx_obs)
        shared_dict.update({'reference_forecast_id': ref_fx_obs.original.forecast.forecast_id})  # NOQA
        if (not ref_fx_fx_prob or
                any([rfx[0].empty or rfx[1].empty for rfx in ref_fx_fx_prob])):
            raise RuntimeError("Missing reference probabilistic forecast "
                               "timeseries data.")
        elif ref_fx_obs.interval_label != processed_fx_obs.interval_label:
            raise ValueError("Mismatched `interval_label` between "
                             "observation and reference forecast.")
        elif (ref_fx_obs.original.forecast.axis !=
              processed_fx_obs.original.forecast.axis):
            raise ValueError("Mismatched `axis` between "
                             "observation and reference forecast.")
    else:
        ref_fx_fx_prob = [(None, None)]*len(fx_fx_prob)

    # No data or metrics
    if (not fx_fx_prob or
            any([fx[0].empty or fx[1].empty for fx in fx_fx_prob])):
        raise RuntimeError("Missing probabilistic forecast timeseries data.")
    elif obs.empty:
        raise RuntimeError("No observation timeseries data.")
    elif len(metrics) == 0:
        raise RuntimeError("No metrics specified.")

    # Separate metrics by type
    single_metrics = list(set(metrics) - set(probabilistic._REQ_DIST))
    dist_metrics = list(set(metrics) & set(probabilistic._REQ_DIST))

    # Single (per ProbabilisticForecastConstantValue) forecast metrics
    if single_metrics:
        cvs = processed_fx_obs.original.forecast.constant_values
        for orig, ref, cv in zip(fx_fx_prob, ref_fx_fx_prob, cvs):
            single_out = copy.deepcopy(shared_dict)
            single_out['forecast_id'] = cv.forecast_id
            results = _calculate_probabilistic_metrics_from_df(
                _create_prob_dataframe(obs, [orig], [ref]),
                categories, single_metrics, cv.interval_label)
            single_out['values'] = _sort_metrics_vals(
                results, datamodel.ALLOWED_PROBABILISTIC_METRICS)
            single_cv_results.append(datamodel.MetricResult.from_dict(
                single_out))

    # Distribution (per ProbabilisticForecast) forecast metrics
    if dist_metrics:
        dist_out = copy.deepcopy(shared_dict)
        dist_out['forecast_id'] = processed_fx_obs.original.forecast.forecast_id  # NOQA
        results = _calculate_probabilistic_metrics_from_df(
            _create_prob_dataframe(obs, fx_fx_prob, ref_fx_fx_prob),
            categories,
            dist_metrics,
            processed_fx_obs.original.forecast.interval_label)
        dist_out['values'] = _sort_metrics_vals(
            results, datamodel.ALLOWED_PROBABILISTIC_METRICS)
        dist_result = datamodel.MetricResult.from_dict(dist_out)

    return (single_cv_results, dist_result)


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
        # total (special category)
        if category == 'total':
            index_category = lambda x: 0  # NOQA
        else:
            index_category = getattr(data_df.index, category)

        # Calculate each metric
        for metric_ in set(metrics):
            # Group by category
            for cat, group in data_df.groupby(index_category):

                try:
                    ref_fx_vals = group.xs('reference_forecast', level=1, axis=1).to_numpy()  # NOQA
                    ref_fx_prob_vals = group.xs('reference_probability', level=1, axis=1).to_numpy()  # NOQA
                    if ref_fx_vals.size == ref_fx_vals.shape[0]:
                        ref_fx_vals = ref_fx_vals.T[0]
                        ref_fx_prob_vals = ref_fx_prob_vals.T[0]
                except KeyError:
                    ref_fx_vals = None
                    ref_fx_prob_vals = None

                fx_vals = group.xs('forecast', level=1, axis=1).to_numpy()
                fx_prob_vals = group.xs('probability', level=1, axis=1).to_numpy()  # NOQA
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
            data_dict[(ref_fx_prob.name, 'reference_probability')] = ref_fx_prob  # NOQA
        except AttributeError:
            pass

    return pd.DataFrame(data_dict)


def _transform_prob_forecast_value_and_prob(proc_fx_obs):
    """
    Helper function that returns ordered list of series of physical values and
    probabilities for a datamodel.ProcessedForecastObservation

    Parameters
    ----------
    proc_fx_obs : datamodel.ProcessedForecastObservation

    Returns
    -------
    fx_fx_prob : list of tuples of (n,) array_like
        Forecast physical values and forecast probabilities.
    """
    fx_fx_prob = []
    df = proc_fx_obs.forecast_values
    assert isinstance(df, pd.DataFrame)
    for col in proc_fx_obs.forecast_values.columns:
        if proc_fx_obs.original.forecast.axis == 'x':
            fx_prob = df[col]
            fx = pd.Series([float(col)]*fx_prob.size, index=fx_prob.index,
                           name=col)
        else:  # 'y'
            fx = df[col]
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

    out['values'] = tuple(metric_vals)
    calc_metrics = datamodel.MetricResult.from_dict(out)
    return calc_metrics
