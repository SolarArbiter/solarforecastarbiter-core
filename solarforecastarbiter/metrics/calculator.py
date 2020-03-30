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
from solarforecastarbiter.metrics import deterministic, probabilistic


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
    all_pairs = None
    const_value = None
    current_fx_id = -1

    for proc_fxobs in processed_pairs:
        if proc_fxobs.original.forecast.forecast_id != current_fx_id:
            fx_iter = 0
            current_fx_id = proc_fxobs.original.forecast.forecast_id

        # determine type of metrics to calculate
        if isinstance(proc_fxobs.original.forecast,
                      datamodel.ProbabilisticForecast):
            const_value = proc_fxobs.original.forecast.constant_value[fx_iter]
            if any(x in probabilistic._REQ_2DFX for x in metrics):
                all_pairs = processed_pairs
            try:
                metrics_ = calculate_probabilistic_metrics(
                    proc_fxobs,
                    categories,
                    metrics,
                    ref_fx_obs=ref_pair,
                    all_proc_fx_obs=all_pairs,
                    constant_value=const_value)
            except RuntimeError as e:
                logger.error('Failed to calculate probabilistic metrics'
                             'for %s: %s',
                             proc_fxobs.name, e)
        else:
            try:
                metrics_ = calculate_deterministic_metrics(
                    proc_fxobs,
                    categories,
                    metrics,
                    ref_fx_obs=ref_pair,
                    normalizer=normalizer)
            except RuntimeError as e:
                logger.error('Failed to calculate deterministic metrics'
                             'for %s: %s',
                             proc_fxobs.name, e)
        calc_metrics.append(metrics_)

        fx_iter += 1

    return calc_metrics


def _apply_deterministic_metric_func(metric, fx, obs, **kwargs):
    """Helper function to deal with variable number of arguments possible for
    deterministic metric functions."""
    metric_func = deterministic._MAP[metric][0]
    if metric in deterministic._REQ_REF_FX:
        return metric_func(obs, fx, kwargs['ref_fx'])
    elif metric in deterministic._REQ_NORM:
        return metric_func(obs, fx, kwargs['normalizer'])
    else:
        return metric_func(obs, fx)


def _apply_probabilistic_metric_func(metric, fx, fx_prob, obs, **kwargs):
    """Helper function to deal with variable number of arguments possible for
    probabilistic metric functions."""
    metric_func = probabilistic._MAP[metric][0]
    if metric in probabilistic._REQ_REF_FX:
        return metric_func(obs, fx, fx_prob,
                           kwargs['ref_fx'], kwargs['ref_fx_prob'])
    else:
        return metric_func(obs, fx.to_numpy(), fx_prob.to_numpy())


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
                                    constant_value, ref_fx_obs=None,
                                    all_pairs=None,):
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
    constant_value : float
        Constant value associated with this ProcessedForecastObservation
    ref_fx_obs : datamodel.ProcessedForecastObservation
        Reference forecast to be used when calculating skill metrics. Default
        is None and no skill metrics will be calculated.
    all_pairs : list of datamodel.ProcessedForecastObservation
        All ProcessedForecastObservation associated with the original
        ProbabilisticForecast. Default is None and Continuous Ranked
        Probability Score (CPRS) will not be computed.

    Returns
    -------
    solarforecastarbiter.datamodel.MetricResult (TBD)
        Contains all the computed metrics by categories.

    Raises
    ------
    RuntimeError
        If there is no forecast, observation timeseries data or no metrics
        are specified.
    ValueError
        If original and reference forecast `interval_label` or `axis` values
        do not match.

    """
    out = {
        'name': processed_fx_obs.name,
        'forecast_id': processed_fx_obs.original.forecast.forecast_id,
    }

    try:
        out['observation_id'] = processed_fx_obs.original.observation.observation_id  # NOQA
    except AttributeError:
        out['aggregate_id'] = processed_fx_obs.original.aggregate.aggregate_id

    import ipdb; ipdb.set_trace()

    # Determine forecast physical and probabiltity values by axis
    fx, fx_prob = _extract_prob_forecast_value_and_prob(
        processed_fx_obs, constant_value)
    obs = processed_fx_obs.observation_values

    # Check reference forecast is from processed pair, if needed
    ref_fx = None
    ref_fx_prob = None
    if any(m in probabilistic._REQ_REF_FX for m in metrics):
        if not ref_fx_obs:
            raise RuntimeError("No reference forecast provided but it is "
                               "required for desired metrics")

        # ref_fx, ref_fx_prob = _extract_prob_forecast_value_and_prob(
        #     ref_fx_obs, constant_value=)
        out['reference_forecast_id'] = ref_fx_obs.original.forecast.forecast_id
        if ref_fx.empty:
            raise RuntimeError("No reference forecast timeseries data")
        elif ref_fx_obs.interval_label != processed_fx_obs.interval_label:
            raise ValueError("Mismatched `interval_label` between "
                             "observation and reference forecast.")
        elif (ref_fx_obs.original.forecast.axis !=
              processed_fx_obs.original.forecast.axis):
            raise ValueError("Mismatched `axis` between "
                             "observation and reference forecast.")

    # No data or metrics
    if fx.empty:
        raise RuntimeError("No Forecast timeseries data.")
    elif obs.empty:
        raise RuntimeError("No Observation timeseries data.")
    elif len(metrics) == 0:
        raise RuntimeError("No metrics specified.")

    # Dataframe for grouping
    # TODO: how to group all these
    df = pd.concat({'forecast': fx,
                    'forecast_probability': fx_prob,
                    'observation': obs,
                    'reference': ref_fx,
                    'reference_probability': ref_fx_prob}, axis=1)

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
                res = _apply_probabilistic_metric_func(
                    metric_, group.forecast, group.observation,
                    ref_fx=ref_fx, ref_fx_prob=ref_fx_prob,
                    all_pairs=all_pairs)

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
        Forecast physical values.
    """
    fx_fx_prob = []
    df = proc_fx_obs.forecast_values
    assert isinstance(df, pd.DataFrame)
    for col in proc_fx_obs.forecast_values.columns:
        if proc_fx_obs.original.forecast.axis == 'x':
            fx_prob = df[col]
            fx = pd.Series([float(col)]*fx_prob.size)
        else:  # 'y'
            fx = df[col]
            fx_prob = pd.Series([float(col)]*fx.size)
        fx_fx_prob.append((fx, fx_prob))
    return fx_fx_prob
