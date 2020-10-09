"""
Provides preprocessing steps to be performed on the timeseries data.
"""
import logging

import numpy as np
import pandas as pd


from solarforecastarbiter import datamodel
from solarforecastarbiter.validation import quality_mapping


logger = logging.getLogger(__name__)

# Titles to refer to counts of preprocessing results
VALIDATION_RESULT_TOTAL_STRING = "TOTAL FLAGGED VALUES DISCARDED"
FILL_RESULT_TOTAL_STRING = "Total {0}Forecast Values {1}"
DISCARD_DATA_STRING = "Values Discarded by Alignment"
UNDEFINED_DATA_STRING = "Undefined Values"
FORECAST_FILL_CONST_STRING = "Filled with {0}"
FORECAST_FILL_STRING_MAP = {'drop': "Dropped",
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


def _resample_event_obs(obs, fx, obs_data, quality_flags):
    """Resample the event observation.

    Parameters
    ----------
    obs : datamodel.Observation
        The Observation being resampled.
    fx : datamodel.EventForecast
        The corresponding Forecast.
    obs_data : pd.Series
        Timeseries data of the event observation.
    quality_flags : tuple of solarforecastarbiter.datamodel.QualityFlagFilter
        Flags to process and apply as filters during resampling.

    Returns
    -------
    obs_resampled : pandas.Series
        Timeseries data of the Observation resampled to match the Forecast.
    counts : dict
        Dict where keys are quality_flag.quality_flags and values
        are integers indicating the number of points filtered
        for the given flag.

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
    counts = discard_before_resample.astype(int).sum(axis=0).to_dict()
    to_discard_before_resample = discard_before_resample.any(axis=1)

    obs_resampled = obs_data[~to_discard_before_resample]

    return obs_resampled, counts


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


def _resample_obs(obs, fx, obs_data, quality_flags):
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

    Returns
    -------
    obs_resampled : pandas.Series
        The observation time series resampled to match the forecast
        interval_length. Time series will have missing labels where
        values failed validation.
    counts : dict
        Dict where keys are quality_flag.quality_flags and values
        are integers indicating the number of points filtered
        for the given flag.
    """
    if fx.interval_length < obs.interval_length:
        raise ValueError(
            'Cannot resample observation to match forecast because '
            'fx.interval_length < obs.interval_length.')

    if obs_data.empty:
        return obs_data

    # label convention when resampling
    closed = datamodel.CLOSED_MAPPING[obs.interval_label]

    # bools w/ has columns like NIGHTTIME, CLEARSKY EXCEEDED
    obs_flags = quality_mapping.convert_mask_into_dataframe(
        obs_data['quality_flag'])
    obs_flags['ISNAN'] = obs_data['value'].isna()

    # determine the points that should never contribute
    # combine unique elements of tuple of tuples
    discard_before_resample_flags = set(['ISNAN'])
    for f in filter(lambda x: x.discard_before_resample, quality_flags):
        discard_before_resample_flags |= set(f.quality_flags)
    discard_before_resample = obs_flags[discard_before_resample_flags]
    counts = discard_before_resample.astype(int).sum(axis=0).to_dict()
    to_discard_before_resample = discard_before_resample.any(axis=1)

    # track number of points discarded before resampling in each interval
    to_discard_before_resample_count = to_discard_before_resample.resample(
        fx.interval_length, closed=closed, label=closed).sum()

    # Series to track if a given resampled interval should be discarded
    to_discard = pd.Series(False, index=to_discard_before_resample_count.index)

    # will be used to determine threshold number of points
    interval_ratio = fx.interval_length / obs.interval_length

    for f in filter(lambda x: not x.discard_before_resample, quality_flags):
        # should we put ISNAN in both the before and during resample exclude?
        quality_flags_to_exclude = f.quality_flags + ('ISNAN', )
        filter_name = ' OR '.join(quality_flags_to_exclude)
        # Reduce DataFrame with relevant flags to bool series.
        # could add a QualityFlagFilter.logic key to control
        # OR (.any(axis=1)) vs. AND (.all(axis=1))
        obs_ser = obs_flags[quality_flags_to_exclude].any(axis=1)
        # Series describing number of points in each interval that are flagged
        resampled_flags_count = obs_ser.resample(
            fx.interval_length, closed=closed, label=closed).sum()
        threshold = f.resample_threshold_percentage * interval_ratio
        flagged = resampled_flags_count > threshold
        to_discard |= flagged
        counts[filter_name] = flagged.sum()

    # determine intervals with too many pre-resampling points removed
    flagged = to_discard_before_resample_count > (0.1 * interval_ratio)
    to_discard |= flagged
    counts['PRE-RESAMPLE EXCEEDED'] = flagged.sum()

    # resample using all of the data
    resampled_values = \
        obs_data.loc[~to_discard_before_resample, 'value'].resample(
            fx.interval_length, closed=closed, label=closed).mean()
    # discard the intervals with too many flagged sub-interval points.
    # resampled_values.index does not contain labels for intervals for
    # which all sub-intervals were discarded, so care is needed in the
    # next indexing operation.
    labels_to_keep = to_discard.index[~to_discard]
    obs_resampled = resampled_values.loc[
        resampled_values.index.intersection(labels_to_keep)]

    return obs_resampled, counts


def filter_resample(fx_obs, fx_data, obs_data, ref_data, quality_flags):
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
    ref_data : pandas.Series or pandas.DataFrame or None
        Timeseries data of the reference forecast.
    quality_flags : tuple of solarforecastarbiter.datamodel.QualityFlagFilter
        Flags to process and apply as filters during resampling.

    Returns
    -------
    forecast_values : pandas.Series or pandas.DataFrame
        Same as input data except may be coerced to a safer dtype.
    observation_values : pandas.Series
        Observation values filtered and resampled.
    counts : dict
        Dict where keys are quality_flag.quality_flags and values
        are integers indicating the number of points filtered
        for the given flag.

    Notes
    -----
    This function does not currently account for mismatches in the
    `interval_label` of the `fx_obs.observation` and `fx_obs.forecast`.

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
        If fx_obs.reference_forecast is not None but ref_data is None
        or vice versa
    ValueError
        If fx_obs.reference_forecast.interval_label or interval_length
        does not match fx_obs.forecast.interval_label or interval_length
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
    ref_fx = fx_obs.reference_forecast

    # raise ValueError if intervals don't match
    _check_ref_fx(fx, ref_fx, ref_data)

    # Resample based on forecast type
    if isinstance(fx, datamodel.EventForecast):
        fx_data = _validate_event_dtype(fx_data)
        obs_data = _validate_event_dtype(obs_data)
        obs_resampled, counts = _resample_event_obs(obs, fx, obs_data)
    else:
        obs_resampled, counts = _resample_obs(obs, fx, obs_data, quality_flags)

    return fx_data, obs_resampled, counts


def align(fx_obs, fx_data, obs_data, ref_data, tz):
    """Align the observation data to the forecast data.

    Parameters
    ----------
    fx_obs : solarforecastarbiter.datamodel.ForecastObservation, solarforecastarbiter.datamodel.ForecastAggregate
        Pair of forecast and observation.
    fx_data : pandas.Series or pandas.DataFrame
        Timeseries data of the forecast.
    obs_data : pandas.Series
        Timeseries data of the observation/aggregate after processing the
        quality flag column.
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
    obs = fx_obs.data_object
    ref_fx = fx_obs.reference_forecast

    # Align (forecast is unchanged)
    # Remove non-corresponding observations and
    # forecasts, and missing periods
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

    # prob fx DataFrame needs to be summed across both dimensions
    if isinstance(fx_data, pd.DataFrame):
        undefined_fx = fx_data.isna().sum().sum()
    else:
        undefined_fx = fx_data.isna().sum()

    # Return dict summarizing results
    results = {
        fx.__blurb__ + " " + DISCARD_DATA_STRING:
            len(fx_data.dropna(how="any")) - len(fx_aligned),
        obs.__blurb__ + " " + DISCARD_DATA_STRING:
            len(obs_data) - len(observation_values),
        fx.__blurb__ + " " + UNDEFINED_DATA_STRING:
            int(undefined_fx),
        obs.__blurb__ + " " + UNDEFINED_DATA_STRING:
            int(obs_data.isna().sum())
    }

    if ref_data is not None:
        k = type(ref_fx).__name__ + " " + UNDEFINED_DATA_STRING
        results[k] = len(ref_data.dropna(how='any')) - len(ref_fx_aligned)

    return forecast_values, observation_values, ref_values, results


def _check_ref_fx(fx, ref_fx, ref_data):
    if ref_fx is not None and ref_data is None:
        raise ValueError(
            'ref_data must be supplied if fx_obs.reference_forecast is not'
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


def _merge_quality_filters(filters):
    """Merge any quality flag filters into one single QualityFlagFilter"""
    combo = set()
    for filter_ in filters:
        if isinstance(filter_, datamodel.QualityFlagFilter):
            combo |= set(filter_.quality_flags)
    return datamodel.QualityFlagFilter(tuple(combo))


def process_forecast_observations(forecast_observations, filters,
                                  forecast_fill_method, start, end,
                                  data, timezone, costs=tuple()):
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
      2. Fill missing reference forecast data points according to
         ``forecast_fill_method``.
      3. Remove observation data points with ``quality_flag`` in
         filters. Remaining observation series is discontinuous.
      4. Resample observations to match forecast intervals. If at least
         10% of the observation intervals within a forecast interval are
         valid (not missing or matching ``filters``), the interval is
         value is computed from all subintervals. Otherwise the
         resampled observation is NaN.
      5. Drop NaN observation values.
      6. Align observations to match forecast times. Observation times
         for which there is not a matching forecast time are dropped.
      7. Create
         :py:class:`~solarforecastarbiter.datamodel.ProcessedForecastObservation`
         with resampled, aligned data and metadata.
    """  # NOQA: E501
    if not all([isinstance(filter_, datamodel.QualityFlagFilter)
                for filter_ in filters]):
        logger.warning(
            'Only filtering on Quality Flag is currently implemented. '
            'Other filters will be discarded.')
        filters = [
            f for f in filters if isinstance(f, datamodel.QualityFlagFilter)]
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

        # Apply fill to forecast and reference forecast
        fx_ser = data[fxobs.forecast]
        fx_ser, count = apply_fill(fx_ser, fxobs.forecast,
                                   forecast_fill_method, start, end)
        preproc_results.append(datamodel.PreprocessingResult(
            name=FILL_RESULT_TOTAL_STRING.format('', forecast_fill_str),
            count=int(count)))
        if fxobs.reference_forecast is not None:
            ref_ser = data[fxobs.reference_forecast]
            ref_ser, count = apply_fill(ref_ser, fxobs.reference_forecast,
                                        forecast_fill_method, start, end)
            preproc_results.append(datamodel.PreprocessingResult(
                name=FILL_RESULT_TOTAL_STRING.format(
                    "Reference ", forecast_fill_str),
                count=int(count)))
        else:
            ref_ser = None

        # filter and resample observation/aggregate data
        obs_data = data[fxobs.data_object]
        forecast_values, observation_values, counts = filter_resample(
            fxobs, fx_ser, obs_data, ref_ser, filters)

        # store validated data in validated_observations
        val_results = tuple(datamodel.ValidationResult(flag=k, count=v)
                            for k, v in counts.items())

        # this count value no longer makes sense because the first object
        # is at a different interval than the second.
        # might need to add a 'total' to counts, exclude from the
        # ValidationResult comprehension above, and use it here.
        preproc_obs_results = datamodel.PreprocessingResult(
            name=VALIDATION_RESULT_TOTAL_STRING,
            count=(len(data[fxobs.data_object]) - len(observation_values)))
        preproc_results.append(preproc_obs_results)

        # Align and create processed pair
        try:
            forecast_values, observation_values, ref_fx_values, results = \
                align(fxobs, forecast_values, observation_values, ref_ser,
                      timezone)
            preproc_results.extend(
                [datamodel.PreprocessingResult(name=k, count=int(v))
                 for k, v in results.items()])
        except Exception as e:
            logger.error(
                'Failed to align data for pair (%s, %s): %s',
                fxobs.forecast.name, fxobs.data_object.name, e)
        else:
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
    if isinstance(forecast, str):
        forecast_name = forecast
    else:
        forecast_name = forecast.name
        if isinstance(forecast, datamodel.ProbabilisticForecastConstantValue):
            if forecast.axis == 'x':
                forecast_name += \
                    f' Prob(x <= {forecast.constant_value} {forecast.units})'
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
