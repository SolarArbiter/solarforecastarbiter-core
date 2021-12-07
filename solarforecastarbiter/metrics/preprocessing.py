"""
Provides preprocessing steps to be performed on the timeseries data.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


from solarforecastarbiter import datamodel
from solarforecastarbiter.validation import quality_mapping


logger = logging.getLogger(__name__)

# Titles to refer to counts of preprocessing results
FILL_RESULT_TOTAL_STRING = "Missing {0}Forecast Values {1}"
DISCARD_DATA_STRING = "{0} Values Discarded by Alignment"
FORECAST_FILL_CONST_STRING = "Filled with {0}"
OUTAGE_DISCARD_STRING = "{0} Values Discarded Due To Outage"
FORECAST_FILL_STRING_MAP = {'drop': "Discarded",
                            'forward': "Forward Filled"}


def apply_fill(fx_data, forecast, forecast_fill_method, start, end):
    """
    Apply fill procedure to the data from the start to end timestamps.

    Parameters
    ----------
    fx_data : pandas.Series or pandas.DataFrame
        Forecast data with pandas.DatetimeIndex.
    forecast : datamodel.Forecast
    forecast_fill_method : {'drop', 'forward', float}
        Indicates what process to use for handling missing forecasts.
          * _'drop'_ drops all missing values for any row with a missing value.
          * _'forward'_ fills missing values with the most recent real value.
            If any leading missing values fill with zeros.
          * _float_ fills any missing values with the given value.
    start : pandas.Timestamp
    end : pandas.Timestamp

    Returns
    -------
    filled: pandas.Series or pandas.DataFrame
        Forecast filled according to the specified logic
    count : int
        Number of values filled or dropped
    """
    forecast_fill_method = str(forecast_fill_method)
    # Create full datetime range at resolution
    full_dt_index = pd.date_range(
        start=start, end=end, freq=forecast.interval_length,
        closed=datamodel.CLOSED_MAPPING[forecast.interval_label],
        name=fx_data.index.name)

    if forecast_fill_method == 'drop':
        # Drop any missing values.
        # If data is a DataFrame any row that is missing a value is
        # dropped for all columns.
        if isinstance(fx_data, pd.DataFrame):
            count = fx_data.isna().any(axis=1).sum() * fx_data.shape[1]
        else:
            count = fx_data.isna().sum()
        filled = fx_data.dropna(how='any').astype(float)
    elif forecast_fill_method == 'forward':
        # Reindex with expected datetime range.
        # Fills missing values with the most recent real value.
        # If any leading missing values fill with zeros.
        filled = fx_data.reindex(index=full_dt_index)
        count = filled.isna().sum()
        filled.fillna(method='ffill', inplace=True)
        filled.fillna(value=0, inplace=True)
    else:
        # Value should be numeric
        try:
            const_fill_value = pd.to_numeric(
                forecast_fill_method).astype(float)
        except ValueError:
            raise ValueError(
                f"Unsupported forecast fill missing data method: "
                f"{forecast_fill_method}")
        # Reindex with expected datetime range.
        # Fills missing values with the given constant value.
        filled = fx_data.reindex(index=full_dt_index)
        count = filled.isna().sum()
        filled.fillna(value=const_fill_value, inplace=True)

    # If data provided as DataFrame count will be a series, so sum over that
    # series to get the total count for all columns (Except for 'drop').
    if isinstance(count, pd.Series):
        count = count.sum()

    return filled, count


def _resample_event_obs(
    obs: Union[datamodel.Observation, datamodel.Aggregate],
    fx: datamodel.EventForecast,
    obs_data: pd.DataFrame,
    quality_flags: Tuple[datamodel.QualityFlagFilter, ...]
) -> Tuple[pd.Series, List[datamodel.ValidationResult]]:
    """Resample the event observation.

    Parameters
    ----------
    obs : datamodel.Observation
        The Observation being resampled.
    fx : datamodel.EventForecast
        The corresponding Forecast.
    obs_data : pandas.DataFrame
        Timeseries of values and quality flags of the
        observation/aggregate data.
    quality_flags : tuple of solarforecastarbiter.datamodel.QualityFlagFilter
        Flags to process and apply as filters during resampling.

    Returns
    -------
    obs_resampled : pandas.Series
        Timeseries data of the Observation resampled to match the Forecast.
    validation_results : list
        Elements are
        :py:class:`solarforecastarbiter.datamodel.ValidationResult`.

    Raises
    ------
    ValueError
        If the Forecast and Observation do not have the same interval length.
    """
    if fx.interval_length != obs.interval_length:
        raise ValueError("Event observation and forecast time-series "
                         "must have matching interval length.")

    # bools w/ has columns like NIGHTTIME, CLEARSKY EXCEEDED, but many of
    # these are not valid for event obs! Arguably only USER FLAGGED and
    # NIGHTTIME are valid for event obs.
    obs_flags = quality_mapping.convert_mask_into_dataframe(
        obs_data['quality_flag'])
    obs_flags['ISNAN'] = obs_data['value'].isna()

    # determine the points that should never contribute
    # combine unique elements of tuple of tuples
    discard_before_resample_flags = set(['ISNAN'])
    for f in filter(lambda x: x.discard_before_resample, quality_flags):
        discard_before_resample_flags |= set(f.quality_flags)
    discard_before_resample = obs_flags[discard_before_resample_flags]
    to_discard_before_resample = discard_before_resample.any(axis=1)
    obs_resampled = obs_data.loc[~to_discard_before_resample, 'value']

    # construct validation results
    counts = discard_before_resample.astype(int).sum(axis=0).to_dict()
    counts['TOTAL DISCARD BEFORE RESAMPLE'] = to_discard_before_resample.sum()
    validation_results = _counts_to_validation_results(counts, True)

    # resampling not allowed, so fill in 0 for discard after resample
    validation_results += _counts_to_validation_results(
        {'TOTAL DISCARD AFTER RESAMPLE': 0},
        False
    )

    return obs_resampled, validation_results


def _validate_event_dtype(ser):
    """
    Validate the event data dtype, converting to boolean values if possible.

    Parameter
    ---------
    ser : pandas.Series
        The event time-series data (observation or forecast).

    Returns
    -------
    ser : pandas.Series
        The event time-series data as boolean values.

    Raises
    ------
    TypeError
        If the event time-series data dtype cannot be converted to boolean.

    """

    if ser.dtype == bool:
        return ser
    elif ser.dtype == int and np.all(np.isin(ser.unique(), [0, 1])):
        return ser.astype(bool)
    elif ser.dtype == float and np.all(np.isin(ser.unique(), [0.0, 1.0])):
        return ser.astype(bool)
    else:
        raise TypeError("Invalid data type for event time-series; unable to "
                        "convert {} to boolean.".format(ser.dtype))


def _resample_obs(
    obs: Union[datamodel.Observation, datamodel.Aggregate],
    fx: datamodel.Forecast,
    obs_data: pd.DataFrame,
    quality_flags: Tuple[datamodel.QualityFlagFilter, ...],
    outages: Tuple[datamodel.TimePeriod, ...] = ()
) -> Tuple[pd.Series, List[datamodel.ValidationResult]]:
    """Resample observations.

    Parameters
    ----------
    obs : datamodel.Observation
        The Observation being resampled.
    fx : datamodel.Forecast
        The corresponding Forecast.
    obs_data : pandas.DataFrame
        Timeseries of values and quality flags of the
        observation/aggregate data.
    quality_flags : tuple of solarforecastarbiter.datamodel.QualityFlagFilter
        Flags to process and apply as filters during resampling.
    outages : tuple of solarforecastarbiter.datamode.TimePeriod
        Determines the time periods to drop from obs_data before resampling.

    Returns
    -------
    obs_resampled : pandas.Series
        The observation time series resampled to match the forecast
        interval_length. Time series will have missing labels where
        values failed validation.
    validation_results : list
        Elements are
        :py:class:`solarforecastarbiter.datamodel.ValidationResult`.

    Raises
    ------
    ValueError
        If fx.interval_length < obs.interval_length
    """
    if fx.interval_length < obs.interval_length:
        # typically impossible to reach this because ForecastObservation init
        # prevents it
        raise ValueError(
            'Cannot resample observation to match forecast because '
            'fx.interval_length < obs.interval_length.')

    if obs_data.empty:
        return obs_data['value'], []

    # fx label convention when resampling
    closed_fx = datamodel.CLOSED_MAPPING[fx.interval_label]

    # obs label convention when resampling
    closed_obs = datamodel.CLOSED_MAPPING[obs.interval_label]

    # drop any outage data before preprocessing
    obs_data, outage_point_count = remove_outage_periods(
        outages, obs_data, obs.interval_label
    )

    outage_result = datamodel.ValidationResult(
        flag="OUTAGE",
        count=int(outage_point_count),
        before_resample=True
    )

    # bools w/ has columns like NIGHTTIME, CLEARSKY EXCEEDED
    obs_flags = quality_mapping.convert_mask_into_dataframe(
        obs_data['quality_flag'])
    obs_flags['ISNAN'] = obs_data['value'].isna()

    # determine the points that should be discarded before resampling.
    to_discard_before_resample, val_results = _calc_discard_before_resample(
        obs_flags, quality_flags)

    val_results.append(outage_result)

    # resample using all of the data except for what was flagged by the
    # discard before resample process.
    resampled_values = \
        obs_data.loc[~to_discard_before_resample, 'value'].resample(
            fx.interval_length, closed=closed_obs, label=closed_fx).mean()

    # determine the intervals that have too many flagged points
    to_discard_after_resample, after_resample_val_results = \
        _calc_discard_after_resample(
            obs_flags,
            quality_flags,
            to_discard_before_resample,
            fx.interval_length,
            obs.interval_length,
            closed_obs,
            closed_fx
        )

    # discard the intervals with too many flagged sub-interval points.
    # resampled_values.index does not contain labels for intervals for
    # which all points were discarded, so care is needed in the next
    # indexing operation.
    good_labels = to_discard_after_resample.index[~to_discard_after_resample]
    obs_resampled = resampled_values.loc[
        resampled_values.index.intersection(good_labels)]

    # merge the val_results lists
    val_results += after_resample_val_results

    return obs_resampled, val_results


def _calc_discard_before_resample(
    obs_flags: pd.DataFrame,
    quality_flags: Tuple[datamodel.QualityFlagFilter, ...]
) -> Tuple[pd.Series, List[datamodel.ValidationResult]]:
    """Determine intervals to discard before resampling.

    Parameters
    ----------
    obs_flags : pd.DataFrame
        Output of convert_mask_into_dataframe, plus ISNAN.
    quality_flags : tuple of solarforecastarbiter.datamodel.QualityFlagFilter
        Flags to process and apply as filters during resampling.

    Returns
    -------
    to_discard_before_resample : pd.Series
        Indicates if a point should be discarded (True) or kept (False)
        before the resample.
    validation_results : list
        Elements are
        :py:class:`solarforecastarbiter.datamodel.ValidationResult`.
    """
    # determine the points that should never contribute
    # combine unique elements of tuple of tuples
    # list(dict.fromkeys()) is good enough for Raymond Hettinger
    # https://stackoverflow.com/a/39835527/2802993
    flags = ['ISNAN']
    for f in filter(lambda x: x.discard_before_resample, quality_flags):
        flags.extend(f.quality_flags)
    discard_before_resample_flags = list(dict.fromkeys(flags))
    discard_before_resample = obs_flags[discard_before_resample_flags]
    to_discard_before_resample = discard_before_resample.any(axis=1)

    # construct validation results
    counts = discard_before_resample.astype(int).sum(axis=0).to_dict()
    counts['TOTAL DISCARD BEFORE RESAMPLE'] = to_discard_before_resample.sum()

    validation_results = _counts_to_validation_results(counts, True)

    # TODO: add filters for time of day and value, OR with
    # to_discard_before_resample, add discarded number to counts

    return to_discard_before_resample, validation_results


def _calc_discard_after_resample(
    obs_flags: pd.DataFrame,
    quality_flags: Tuple[datamodel.QualityFlagFilter, ...],
    to_discard_before_resample: pd.Series,
    fx_interval_length: pd.Timedelta,
    obs_interval_length: pd.Timedelta,
    closed_obs: Optional[str],
    closed_fx: Optional[str]
) -> Tuple[pd.Series, List[datamodel.ValidationResult]]:
    """Determine intervals to discard after resampling.

    Parameters
    ----------
    obs_flags : pd.DataFrame
        Output of convert_mask_into_dataframe, plus ISNAN.
    quality_flags : tuple of solarforecastarbiter.datamodel.QualityFlagFilter
        Flags to process and apply as filters during resampling.
    to_discard_before_resample : pd.Series
        Boolean Series indicating if a point should be discarded before
        resampling. Used when determining if too many points
    fx_interval_length : pd.Timedelta
        Forecast interval length to resample to.
    obs_interval_length : pd.Timedelta
        Observation interval length.
    closed : {'left', 'right', None}
        Interval label convention.

    Returns
    -------
    to_discard_after_resample : pd.Series
        Indicates if a point should be discarded (True) or kept (False)
        before the resample.
    validation_results : list
        Elements are
        :py:class:`solarforecastarbiter.datamodel.ValidationResult`.
    """
    # number of points discarded before resampling in each interval
    to_discard_before_resample_count = to_discard_before_resample.resample(
        fx_interval_length, closed=closed_obs, label=closed_fx).sum()

    # Series to track if a given resampled interval should be discarded
    to_discard_after_resample = pd.Series(
        False, index=to_discard_before_resample_count.index)

    # will be used to determine threshold number of points
    interval_ratio = fx_interval_length / obs_interval_length

    # track number of flagged intervals in a dict
    counts = {}

    def apply_flag(quality_flag):
        # should we put ISNAN in both the before and during resample exclude?
        # use list to ensure column selection works
        quality_flags_to_exclude = list(quality_flag.quality_flags) + ['ISNAN']
        filter_name = ' OR '.join(quality_flags_to_exclude)
        # Reduce DataFrame with relevant flags to bool series.
        # could add a QualityFlagFilter.logic key to control
        # OR (.any(axis=1)) vs. AND (.all(axis=1))
        obs_flag_ser = obs_flags[quality_flags_to_exclude].any(axis=1)
        # TODO: add time of day and value boolean tests here,
        # then OR with obs_ser and adjust filter_name.
        # Series describing number of points in each interval that are flagged
        resampled_flags_count = obs_flag_ser.resample(
            fx_interval_length, closed=closed_obs, label=closed_fx).sum()
        threshold = (
            quality_flag.resample_threshold_percentage / 100. * interval_ratio)
        # If threshold is 0, any points being flagged counts, but
        # don't just throw away all data.
        if threshold == 0:
            flagged = resampled_flags_count > threshold
        else:
            flagged = resampled_flags_count >= threshold
        return filter_name, flagged

    # apply to all quality_flag objects, including those with
    # discard_before_resample == True. This ensures that we throw out
    # resampled intervals that have too few points.
    for quality_flag in quality_flags:
        filter_name, flagged = apply_flag(quality_flag)
        to_discard_after_resample |= flagged
        counts[filter_name] = flagged.sum()

    counts['TOTAL DISCARD AFTER RESAMPLE'] = to_discard_after_resample.sum()
    validation_results = _counts_to_validation_results(counts, False)

    return to_discard_after_resample, validation_results


def _counts_to_validation_results(
    counts: Dict[str, int], before_resample: bool
) -> List[datamodel.ValidationResult]:
    return [
        datamodel.ValidationResult(
            flag=k,
            count=int(v),
            before_resample=before_resample)
        for k, v in counts.items()
    ]


def _search_validation_results(val_results, key):
    for res in val_results:
        if res.flag == key:
            return res.count


def filter_resample(
    fx_obs: Union[datamodel.ForecastObservation, datamodel.ForecastAggregate],
    fx_data: Union[pd.Series, pd.DataFrame],
    obs_data: pd.DataFrame,
    quality_flags: Tuple[datamodel.QualityFlagFilter, ...],
    outages: Tuple[datamodel.TimePeriod, ...] = ()
) -> Tuple[
    Union[pd.Series, pd.DataFrame],
    pd.Series,
    List[datamodel.ValidationResult]
]:
    """Filter and resample the observation to the forecast interval length.

    Parameters
    ----------
    fx_obs : solarforecastarbiter.datamodel.ForecastObservation, solarforecastarbiter.datamodel.ForecastAggregate
        Pair of forecast and observation.
    fx_data : pandas.Series or pandas.DataFrame
        Timeseries data of the forecast.
    obs_data : pandas.DataFrame
        Timeseries of values and quality flags of the
        observation/aggregate data.
    quality_flags : tuple of solarforecastarbiter.datamodel.QualityFlagFilter
        Flags to process and apply as filters during resampling.
    outages: tuple of :py:class:`solarforecastarbiter.datamodel.TimePeriod`
        Time periods to drop from data prior to filtering or alignment.

    Returns
    -------
    forecast_values : pandas.Series or pandas.DataFrame
        Same as input data except may be coerced to a safer dtype.
    observation_values : pandas.Series
        Observation values filtered and resampled.
    validation_results : list
        Elements are
        :py:class:`solarforecastarbiter.datamodel.ValidationResult`.

    Notes
    -----
    The keep/exclude result of each element of the ``quality_flags``
    tuple is combined with the OR operation.

    For ``quality_flags`` tuple elements where
    ``QualityFlagFilter.discard_before_resample`` is ``False``, the
    ``QualityFlagFilter.quality_flags`` are considered during the
    resampling operation. The flags of the raw observations are combined
    with ``OR``, the total number of flagged points within a resample
    period is computed, and intervals are discarded where
    ``QualityFlagFilter.resample_threshold_percentage`` is exceeded.

    Therefore, the following examples can produce different results:

    >>> # separate flags. OR computed after resampling.
    >>> qflag_1 = QualityFlagFilter(('NIGHTTIME', ), discard_before_resample=False)
    >>> qflag_2 = QualityFlagFilter(('CLEARSKY', ), discard_before_resample=False)

    >>> # combined flags. OR computed during resampling.
    >>> qflag_combined = QualityFlagFilter(('NIGHTTIME', 'CLEARSKY'),
            discard_before_resample=False)

    Raises
    ------
    ValueError
        If fx_obs.forecast.interval_length is less than
        fx_obs.observation.interval_length
    ValueError
        If fx_obs.forecast is an EventForecast and
        fx_obs.forecast.interval_length is not equal to
        fx_obs.observation.interval_length
    """  # noqa: E501
    fx = fx_obs.forecast
    obs = fx_obs.data_object

    # Resample based on forecast type
    if isinstance(fx, datamodel.EventForecast):
        fx_data = _validate_event_dtype(fx_data)
        obs_data['value'] = _validate_event_dtype(obs_data['value'])
        obs_resampled, validation_results = _resample_event_obs(
            obs, fx, obs_data, quality_flags)
    else:
        obs_resampled, validation_results = _resample_obs(
            obs, fx, obs_data, quality_flags, outages)

    return fx_data, obs_resampled, validation_results


def align(fx_obs, fx_data, obs_data, ref_data, tz):
    """Align the observation data to the forecast data.

    Parameters
    ----------
    fx_obs : solarforecastarbiter.datamodel.ForecastObservation, solarforecastarbiter.datamodel.ForecastAggregate
        Pair of forecast and observation.
    fx_data : pandas.Series or pandas.DataFrame
        Timeseries data of the forecast.
    obs_data : pandas.Series
        Timeseries data of the observation/aggregate after processing
        the quality flag column and resampling to match
        fx_obs.forecast.interval_length.
    ref_data : pandas.Series or pandas.DataFrame or None
        Timeseries data of the reference forecast.
    tz : str
        Timezone to which processed data will be converted.

    Returns
    -------
    forecast_values : pandas.Series or pandas.DataFrame
    observation_values : pandas.Series
    reference_forecast_values : pandas.Series or pandas.DataFrame or None
    results : dict
        Keys are strings and values are typically integers that
        describe number of discarded and undefined data points.

    Notes
    -----
    This function does not currently account for mismatches in the
    `interval_label` of the `fx_obs.observation` and `fx_obs.forecast`.

    If ``obs_data`` will be subsampled if it is higher frequency than
    fx_data, but users should not rely on this behavior. Instead, use
    :py:func:`~.filter_resample` to match the input observations to the
    forecast data.
    """  # noqa: E501
    fx = fx_obs.forecast
    obs = fx_obs.data_object
    ref_fx = fx_obs.reference_forecast

    # Align (forecast is unchanged)
    # Remove non-corresponding observations and forecasts, and missing periods
    obs_data = obs_data.dropna(how="any")
    obs_aligned, fx_aligned = obs_data.align(
        fx_data.dropna(how="any"), 'inner')
    # another alignment step if reference forecast exists.
    # here we drop points that don't exist in all 3 series.
    # could set reference forecast to NaN where missing instead.
    # could set to 0 instead.
    # could build a DataFrame (implicit outer-join), then perform
    # alignment using ['forecast', 'observation'] or
    # ['forecast', 'observation', 'reference'] selections
    if ref_data is not None:
        obs_aligned, ref_fx_aligned = obs_aligned.align(
            ref_data.dropna(how="any"), 'inner')
        fx_aligned = fx_aligned.reindex(obs_aligned.index)
        ref_values = ref_fx_aligned.tz_convert(tz)
    else:
        ref_values = None

    # Determine series with timezone conversion
    forecast_values = fx_aligned.tz_convert(tz)
    observation_values = obs_aligned.tz_convert(tz)

    # Return dict summarizing results
    discarded_fx_intervals = len(fx_data.dropna(how="any")) - len(fx_aligned)
    discarded_obs_intervals = len(obs_data) - len(observation_values)
    obs_blurb = "Validated, Resampled " + obs.__blurb__
    results = {
        DISCARD_DATA_STRING.format(fx.__blurb__): discarded_fx_intervals,
        DISCARD_DATA_STRING.format(obs_blurb): discarded_obs_intervals
    }

    if ref_data is not None:
        k = DISCARD_DATA_STRING.format("Reference " + ref_fx.__blurb__)
        results[k] = len(ref_data.dropna(how='any')) - len(ref_fx_aligned)

    return forecast_values, observation_values, ref_values, results


def check_reference_forecast_consistency(fx_obs, ref_data):
    """Filter and resample the observation to the forecast interval length.

    Parameters
    ----------
    fx_obs : solarforecastarbiter.datamodel.ForecastObservation, solarforecastarbiter.datamodel.ForecastAggregate
        Pair of forecast and observation.
    ref_data : pandas.Series or pandas.DataFrame or None
        Timeseries data of the reference forecast.

    Raises
    ------
    ValueError
        If fx_obs.reference_forecast is not None but ref_data is None
        or vice versa
    ValueError
        If fx_obs.reference_forecast.interval_label or interval_length
        does not match fx_obs.forecast.interval_label or interval_length
    """  # noqa: E501
    fx = fx_obs.forecast
    ref_fx = fx_obs.reference_forecast

    if ref_fx is not None and ref_data is None:
        raise ValueError(
            'ref_data must be supplied if fx_obs.reference_forecast is not '
            'None')
    elif ref_fx is None and ref_data is not None:
        raise ValueError(
            'ref_data was supplied but fx_obs.reference_forecast is None')

    if ref_fx is not None:
        if fx.interval_length != ref_fx.interval_length:
            raise ValueError(
                f'forecast.interval_length "{fx.interval_length}" must match '
                'reference_forecast.interval_length '
                f'"{ref_fx.interval_length}"')
        if fx.interval_label != ref_fx.interval_label:
            raise ValueError(
                f'forecast.interval_label "{fx.interval_label}" must match '
                f'reference_forecast.interval_label "{ref_fx.interval_label}"')
        if isinstance(fx, datamodel.ProbabilisticForecast):
            if fx.axis != ref_fx.axis:
                raise ValueError(
                    f'forecast.axis "{fx.axis}" must match '
                    f'reference_forecast.axis "{ref_fx.axis}"')


def process_forecast_observations(forecast_observations, filters,
                                  forecast_fill_method, start, end,
                                  data, timezone, costs=tuple(),
                                  outages=tuple()):
    """
    Convert ForecastObservations into ProcessedForecastObservations
    applying any filters and resampling to align forecast and observation.

    Parameters
    ----------
    forecast_observations : list of solarforecastarbiter.datamodel.ForecastObservation, solarforecastarbiter.datamodel.ForecastAggregate
        Pairs to process
    filters : list of solarforecastarbiter.datamodel.BaseFilter
        Filters to apply to each pair.
    forecast_fill_method : str
        Indicates what process to use for handling missing forecasts.
        Currently supports : 'drop', 'forward', and bool or numeric value.
    start : pandas.Timestamp
        Start date and time for assessing forecast performance.
    end : pandas.Timestamp
        End date and time for assessing forecast performance.
    data : dict
        Dict with keys that are the Forecast/Observation/Aggregate object
        and values that are the corresponding pandas.Series/DataFrame for
        the object. Keys must also include all Forecast objects assigned
        to the ``reference_forecast`` attributes of the
        ``forecast_observations``.
    timezone : str
        Timezone that data should be converted to
    costs : tuple of :py:class:`solarforecastarbiter.datamodel.Cost`
        Costs that are referenced by any pairs. Pairs and costs are matched
        by the Cost name.
    outages : tuple of :py:class:`solarforecastarbiter.datamodel.TimePeriod`
        Tuple of time periods during which forecast submissions will be
        excluded from analysis.

    Returns
    -------
    tuple of ProcessedForecastObservation

    Notes
    -----
    In the case where the `interval_label` of the `obs` and `fx` do not
    match, this function currently returns a
    `ProcessedForecastObservation` object with a `interval_label` the
    same as the `fx`, regardless of whether the `interval_length` of the
    `fx` and `obs` are the same or different.

    The processing logic is as follows. For each forecast, observation
    pair in ``forecast_observations``:

      1. Fill missing forecast data points according to
         ``forecast_fill_method``.
      2. Remove any forecast points associated with an outage.
      3. Fill missing reference forecast data points according to
         ``forecast_fill_method``.
      4. Remove any reference forecast or observation points associated
         with an outage.
      5. Remove observation data points with ``quality_flag`` in
         filters. Remaining observation series is discontinuous.
      6. Resample observations to match forecast intervals. If at least
         10% of the observation intervals within a forecast interval are
         valid (not missing or matching ``filters``), the interval is
         value is computed from all subintervals. Otherwise the
         resampled observation is NaN.
      7. Drop NaN observation values.
      8. Align observations to match forecast times. Observation times
         for which there is not a matching forecast time are dropped on
         a forecast by forecast basis.
      9. Create
         :py:class:`~solarforecastarbiter.datamodel.ProcessedForecastObservation`
         with resampled, aligned data and metadata.
    """  # NOQA: E501
    if not all([isinstance(filter_, datamodel.QualityFlagFilter)
                for filter_ in filters]):
        logger.warning(
            'Only filtering on Quality Flag is currently implemented. '
            'Other filters will be discarded.')
        filters = tuple(
            f for f in filters if isinstance(f, datamodel.QualityFlagFilter))

    # create string for tracking forecast fill results.
    # this approach supports known methods or filling with contant values.
    forecast_fill_str = FORECAST_FILL_STRING_MAP.get(
        forecast_fill_method,
        FORECAST_FILL_CONST_STRING.format(forecast_fill_method)
    )
    costs_dict = {c.name: c for c in costs}
    # accumulate ProcessedForecastObservations in a dict.
    # use a dict so we can keep track of existing names and avoid repeats.
    processed_fxobs = {}
    for fxobs in forecast_observations:
        # accumulate PreprocessingResults from various stages in a list
        preproc_results = []

        # extract fx and obs data from data dict
        try:
            fx_data = data[fxobs.forecast]
        except KeyError as e:
            logger.error(
                'Failed to find data for forecast %s: %s',
                fxobs.forecast.name, e)
            continue

        try:
            obs_data = data[fxobs.data_object]
        except KeyError as e:
            logger.error(
                'Failed to find data for observation %s: %s',
                fxobs.data_object.name, e)
            continue

        # Get periods where data should be excluded from analysis due
        # to outages.
        forecast_outage_periods = outage_periods(
            fxobs.forecast,
            start,
            end,
            outages
        )

        # Apply fill to forecast and reference forecast
        fx_data, count = apply_fill(fx_data, fxobs.forecast,
                                    forecast_fill_method, start, end)
        preproc_results.append(datamodel.PreprocessingResult(
            name=FILL_RESULT_TOTAL_STRING.format('', forecast_fill_str),
            count=int(count)))

        outages_exist = len(outages) > 0
        if outages_exist:
            # Remove any forecast data that would have been submitted
            # during an outage
            fx_data, fx_outage_points = remove_outage_periods(
                forecast_outage_periods, fx_data, fxobs.forecast.interval_label
            )
            preproc_results.append(datamodel.PreprocessingResult(
                name=OUTAGE_DISCARD_STRING.format('Forecast'),
                count=int(fx_outage_points)))

        ref_data = data.get(fxobs.reference_forecast, None)

        try:
            check_reference_forecast_consistency(fxobs, ref_data)
        except ValueError as e:
            logger.error('Incompatible reference forecast and data: %s', e)
            continue

        if fxobs.reference_forecast is not None:
            ref_data, count = apply_fill(ref_data, fxobs.reference_forecast,
                                         forecast_fill_method, start, end)

            preproc_results.append(datamodel.PreprocessingResult(
                name=FILL_RESULT_TOTAL_STRING.format(
                    "Reference ", forecast_fill_str),
                count=int(count)))
            if outages_exist:
                ref_data, ref_outage_points = remove_outage_periods(
                    forecast_outage_periods, ref_data,
                    fxobs.reference_forecast.interval_label
                )
                preproc_results.append(datamodel.PreprocessingResult(
                    name=OUTAGE_DISCARD_STRING.format('Reference Forecast'),
                    count=int(ref_outage_points))
                )

        # filter and resample observation/aggregate data
        try:
            forecast_values, observation_values, val_results = filter_resample(
                fxobs, fx_data, obs_data, filters, forecast_outage_periods)
        except Exception as e:
            # should figure out the specific exception types to catch
            logger.error(
                'Failed to filter and resample data for pair (%s, %s): %s',
                fxobs.forecast.name, fxobs.data_object.name, e)
            continue

        if outages_exist:
            obs_outage_points_dropped = _search_validation_results(
                val_results, 'OUTAGE')
            if obs_outage_points_dropped is None:
                logger.warning(
                    'Observation Values Discarded Due To Outage Not Available '
                    'For Pair (%s, %s)', fxobs.forecast.name,
                    fxobs.data_object.name)
            else:
                preproc_results.append(datamodel.PreprocessingResult(
                    name=OUTAGE_DISCARD_STRING.format('Observation'),
                    count=int(obs_outage_points_dropped)))

        # the total count ultimately shows up in both the validation
        # results table and the preprocessing summary table.
        total_discard_before_resample = _search_validation_results(
            val_results, 'TOTAL DISCARD BEFORE RESAMPLE')
        if total_discard_before_resample is None:
            logger.warning(
                'TOTAL DISCARD BEFORE RESAMPLE not available for pair '
                '(%s, %s)', fxobs.forecast.name, fxobs.data_object.name)
        else:
            preproc_results.append(datamodel.PreprocessingResult(
                name='Observation Values Discarded Before Resampling',
                count=int(total_discard_before_resample)))

        total_discard_after_resample = _search_validation_results(
            val_results, 'TOTAL DISCARD AFTER RESAMPLE')
        if total_discard_after_resample is None:
            logger.warning(
                'TOTAL DISCARD AFTER RESAMPLE not available for pair (%s, %s)',
                fxobs.forecast.name, fxobs.data_object.name)
        else:
            preproc_results.append(datamodel.PreprocessingResult(
                name='Resampled Observation Values Discarded',
                count=int(total_discard_after_resample)))

        # Align and create processed pair
        try:
            forecast_values, observation_values, ref_fx_values, results = \
                align(fxobs, forecast_values, observation_values, ref_data,
                      timezone)
            preproc_results.extend(
                [datamodel.PreprocessingResult(name=k, count=int(v))
                 for k, v in results.items()])
        except Exception as e:
            logger.error(
                'Failed to align data for pair (%s, %s): %s',
                fxobs.forecast.name, fxobs.data_object.name, e)
            continue

        logger.info('Processed data successfully for pair (%s, %s)',
                    fxobs.forecast.name, fxobs.data_object.name)
        name = _name_pfxobs(processed_fxobs.keys(), fxobs.forecast)
        cost_name = fxobs.cost
        cost = costs_dict.get(cost_name)
        if cost_name is not None and cost is None:
            logger.warning(
                'Cannot calculate cost metrics for %s, cost parameters '
                'not supplied for cost: %s', name, cost_name)
        processed = datamodel.ProcessedForecastObservation(
            name=name,
            original=fxobs,
            interval_value_type=fxobs.forecast.interval_value_type,
            interval_length=fxobs.forecast.interval_length,
            interval_label=fxobs.forecast.interval_label,
            valid_point_count=len(forecast_values),
            validation_results=val_results,
            preprocessing_results=tuple(preproc_results),
            forecast_values=forecast_values,
            observation_values=observation_values,
            reference_forecast_values=ref_fx_values,
            normalization_factor=fxobs.normalization,
            uncertainty=fxobs.uncertainty,
            cost=cost
        )
        processed_fxobs[name] = processed
    return tuple(processed_fxobs.values())


def _name_pfxobs(current_names, forecast, i=1):
    """Create unique, descriptive name for forecast.

    Users should call this function with a ``Forecast`` object. This
    will augment the ``Forecast.name`` attribute with probabilistic
    descriptors (if needed). The function will then inspect the list of
    names in ``current_names``. If the name is not unique, a recursive
    call, using a string, will allow up to 99 variants of the name.

    Parameters
    ----------
    current_names : list of str
    forecast : datamodel.Forecast or str

    Returns
    -------
    str
    """
    if isinstance(forecast, str):
        # handle input when called recursively
        forecast_name = forecast
    else:
        # handle initial augmentation of forecast.name
        forecast_name = forecast.name
        if isinstance(forecast, datamodel.ProbabilisticForecastConstantValue):
            if forecast.axis == 'x':
                forecast_name += (
                    f' Prob(x <= {forecast.constant_value} '
                    f'{forecast.constant_value_units})')
            else:
                forecast_name += f' Prob(f <= x) = {forecast.constant_value}%'
    if i > 99:
        logger.warning(
            'Limit of unique names for identically named forecasts reached.'
            ' Aligned pairs may have duplicate names.')
        return forecast_name
    if forecast_name in current_names:
        if i == 1:
            new_name = f'{forecast_name}-{i:02d}'
        else:
            new_name = f'{forecast_name[:-3]}-{i:02d}'
        return _name_pfxobs(current_names, new_name, i + 1)
    else:
        return forecast_name


def forecast_report_issue_times(
    forecast: datamodel.Forecast,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DatetimeIndex:
    """Returns all of the issue times that contribute data
    to a report for this forecast. May include issue times that
    correspond with data before and after the report to ensure
    report coverage.

    Parameters
    ----------
    forecast: datamodel.Forecast
        Forecast to find issue times for.
    start: pd.Timestamp
        Start of the report.
    end: pd.Timestamp
        End of the report.

    Returns
    -------
    pandas.DatetimeIndex
        Pandas DatetimeIndex representing all of the issue times.
    """
    # Get total forecast horizon to get the time from issue to last value
    total_forecast_horizon = forecast.lead_time_to_start + forecast.run_length

    # Convert start to utc so we can align with forecast issue time. This
    # is necessary because the report start/end are not necessarily aligned
    # with forecast issue times or forecast start times.
    utc_start = start.tz_convert('UTC')

    # Get the last potential issue time that does not contribute
    # data to the report. An issue time here would contain data up
    # until report_start. We want the first issue time after this
    # time.
    lookback_start = utc_start - total_forecast_horizon

    # Realign to a forecast issue time near the lookback
    issue_search_start = lookback_start.replace(
        hour=forecast.issue_time_of_day.hour,
        minute=forecast.issue_time_of_day.minute
    )

    # Get the number of forecast runs between issue start and lookback
    lookback_diff = issue_search_start - lookback_start

    # Floor to round up if negative (issue_search_start before lookback)
    # or round down if positive (issue_search_start after lookback)
    runs_until_start = np.floor(lookback_diff / forecast.run_length)

    # Get the duration of runs between issue_search_start and lookback_start.
    run_durations = runs_until_start * forecast.run_length

    # Find the issue time immediately after lookback_start by subtracting the
    # run durations (which will be negative for issue_search start before
    # lookback_start).
    first_issue_time = issue_search_start - run_durations

    if first_issue_time == lookback_start:
        # if first issue time is lookback_start, adjust by one run to find
        # the issue time that contributes the first values within the report
        first_issue_time += forecast.run_length

    # Get all possible issue times that contribute to the report from the
    # first issue time, until the lead time of the forecast before the
    # end of the report.
    issue_times = pd.date_range(
        first_issue_time,
        end.tz_convert('UTC') - forecast.lead_time_to_start,
        freq=forecast.run_length,
        closed="left"
    )
    return issue_times


def outage_periods(
    forecast: datamodel.Forecast,
    start: pd.Timestamp,
    end: pd.Timestamp,
    outages: Tuple[datamodel.TimePeriod, ...]
) -> Tuple[datamodel.TimePeriod, ...]:
    """Converts report outage periods to forecast data periods to
    drop from analysis. The returned periods do not account for
    interval label.

    Parameters
    ----------
    forecast: solarforecastarbiter.datamodel.Forecast
    start: pandas.Timestamp
    end: pandas.Timestamp
    outages: tuple of solarforecastarbiter.datamodel.TimePeriod
        List of time ranges to check for forecast issue times.

    Returns
    -------
    tuple of solarforecastarbiter.datamodel.TimePeriod
        Times between these values should not be included in analysis.
    """
    # First, determine a list of forecast issue times that include data that
    # falls within the report
    issue_times = forecast_report_issue_times(forecast, start, end)

    outage_periods = []
    # For each outage, if a forecast submission/issue_time falls within
    # the outage, create start/end bounds for the forecast data to exclude.
    for outage in outages:
        outage_submissions = issue_times[
            (issue_times >= outage.start) & (issue_times <= outage.end)
        ]
        for issue_time in outage_submissions:
            fx_start = issue_time + forecast.lead_time_to_start
            fx_end = fx_start + forecast.run_length
            outage_periods.append(datamodel.TimePeriod(
                start=fx_start,
                end=fx_end
            ))
    return tuple(outage_periods)


def remove_outage_periods(
    outages: Tuple[datamodel.TimePeriod, ...],
    data: pd.DataFrame,
    interval_label: str
) -> Tuple[pd.DataFrame, int]:
    """Returns a copy of a dataframe with all values within an outage
    period dropped.

    Parameters
    ----------
    outages: tuple of :py:class:`solarforecastarbiter.datamodel.TimePeriod`
        Tuple of dictionaries with start and end keys. Values should be
        timestamps denoting the start and end of periods to remove.
    data: pandas.DataFrame
        The dataframe to drop outage data from.
    interval_label: str
        The interval label to drop.

    Returns
    -------
    pandas.DataFrame, int
        The data DataFrame with outage data dropped, and total
        number of points removed.
    """  # NOQA
    if len(outages) == 0:
        return data, 0

    # Set to the boolean series of outage data on first iteration
    # of loop below
    full_outage_index = pd.Series(False, index=data.index)

    for outage in outages:
        if interval_label == "ending":
            outage_index = (data.index > outage.start) & (
                data.index <= outage.end)
        else:
            outage_index = (data.index >= outage.start) & (
                data.index < outage.end)

        full_outage_index = full_outage_index | outage_index
    dropped_total = full_outage_index.sum()
    return data[~full_outage_index], dropped_total
