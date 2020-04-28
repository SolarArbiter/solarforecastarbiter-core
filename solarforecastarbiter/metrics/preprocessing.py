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
DISCARD_DATA_STRING = "Values Discarded by Alignment"
UNDEFINED_DATA_STRING = "Undefined Values"


def apply_validation(obs_df, qfilter, handle_func):
    """
    Apply validation steps based on provided filters to the data.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        The observation data with 'value' and 'quality_flag' columns
    qfilter : solarforecastarbiter.datamodel.QualityFlagFilter
    handle_func : function
        Function that handles how `quality_flags` will be used.
        See solarforecastarbiter.metrics.preprocessing.exclude as an
        example.

    Returns
    -------
    validated_obs : pandas.Series
        The validated timeseries data as pandas.Series.
    counts : dict
        Dict where keys are qfilter.quality_flags and values
        are integers indicating the number of points filtered
        for the given flag.
    """
    # List of flags from filter
    if not isinstance(qfilter, datamodel.QualityFlagFilter):
        raise TypeError(f"{qfilter} not a QualityFlagFilter")
    filters = qfilter.quality_flags

    if obs_df.empty:
        return obs_df.value, {f: 0 for f in filters}
    else:
        validation_df = quality_mapping.convert_mask_into_dataframe(
            obs_df['quality_flag'])
        validation_df = validation_df[list(filters)]
        validated_obs = handle_func(obs_df.value, validation_df)
        counts = validation_df.astype(int).sum(axis=0).to_dict()
        return validated_obs, counts


def _resample_event_obs(obs, fx, obs_series):
    """
    Resample the event observation.

    Parameters
    ----------
    obs : datamodel.Observation
        The Observation being resampled.
    fx : datamodel.EventForecast
        The corresponding Forecast.
    obs_series : pd.Series
        Timeseries data of the event observation.

    Returns
    -------
    obs_resampled : pandas.Series
        Timeseries data of the Observation resampled to match the Forecast.

    Raises
    ------
    RuntimeError
        If the Forecast and Observation do not have the same interval length.

    """

    if fx.interval_length != obs.interval_length:
        raise ValueError("Event observation and forecast time-series "
                         "must have matching interval length.")
    else:
        obs_resampled = obs_series

    return obs_resampled


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


def _resample_obs(obs, fx, obs_series):
    """

    Parameters
    ----------
    obs : datamodel.Observation
        The Observation being resampled.
    fx : datamodel.Forecast
        The corresponding Forecast.
    obs_series : pandas.Series
        Timeseries data of the observation/aggregate after processing the
        quality flag column.

    Returns
    -------
    obs_resampled : pandas.Series
        The observation time-series resampled to match the forecast
        time-series.

    """

    # Resample observation, checking for invalid interval_length and that
    # the Series has data:
    if fx.interval_length > obs.interval_length and not obs_series.empty:
        closed = datamodel.CLOSED_MAPPING[fx.interval_label]
        obs_resampled = obs_series.resample(
            fx.interval_length,
            label=closed,
            closed=closed
        ).agg(["mean", "count"])

        # Drop intervals if too many samples missing
        count_threshold = int(
            fx.interval_length / obs.interval_length * 0.1
        )
        obs_resampled = obs_resampled["mean"].where(
            obs_resampled["count"] >= count_threshold
        )
    else:
        obs_resampled = obs_series

    return obs_resampled


def resample_and_align(fx_obs, fx_series, obs_series, ref_series, tz):
    """
    Resample the observation to the forecast interval length and align to
    remove overlap.

    Parameters
    ----------
    fx_obs : solarforecastarbiter.datamodel.ForecastObservation, solarforecastarbiter.datamodel.ForecastAggregate
        Pair of forecast and observation.
    fx_series : pandas.Series
        Timeseries data of the forecast.
    obs_series : pandas.Series
        Timeseries data of the observation/aggregate after processing the quality
        flag column.
    ref_series : pandas.Series or None
        Timeseries data of the reference forecast.
    tz : str
        Timezone to which processed data will be converted.

    Returns
    -------
    forecast_values : pandas.Series
    observation_values : pandas.Series
    reference_forecast_values : pandas.Series or None
    results : dict

    Notes
    -----
    In the case where the `interval_label` of the `obs` and `fx` do not match,
    this function currently returns a `ProcessedForecastObservation` object
    with a `interval_label` the same as the `fx`, regardless of whether the
    `interval_length` of the `fx` and `obs` are the same or different.

    Raises
    ------
    ValueError
        If fx_obs.reference_forecast is not None but ref_series is None
        or vice versa
    ValueError
        If fx_obs.reference_forecast.interval_label or interval_length
        does not match fx_obs.forecast.interval_label or interval_length

    Todo
    ----
      * Add other resampling functions (besides mean like first, last, median)
    """  # noqa: E501
    fx = fx_obs.forecast
    obs = fx_obs.data_object
    ref_fx = fx_obs.reference_forecast

    # raise ValueError if intervals don't match
    _check_ref_fx(fx, ref_fx, ref_series)

    # Resample based on forecast type
    if isinstance(fx, datamodel.EventForecast):
        fx_series = _validate_event_dtype(fx_series)
        obs_series = _validate_event_dtype(obs_series)
        obs_resampled = _resample_event_obs(obs, fx, obs_series)
    else:
        obs_resampled = _resample_obs(obs, fx, obs_series)

    # Align (forecast is unchanged)
    # Remove non-corresponding observations and
    # forecasts, and missing periods
    obs_resampled = obs_resampled.dropna(how="any")
    obs_aligned, fx_aligned = obs_resampled.align(fx_series.dropna(how="any"),
                                                  'inner')
    # another alignment step if reference forecast exists.
    # here we drop points that don't exist in all 3 series.
    # could set reference forecast to NaN where missing instead.
    # could set to 0 instead.
    # could build a DataFrame (implicit outer-join), then perform
    # alignment using ['forecast', 'observation'] or
    # ['forecast', 'observation', 'reference'] selections
    if ref_series is not None:
        obs_aligned, ref_fx_aligned = obs_aligned.align(
            ref_series.dropna(how="any"), 'inner')
        fx_aligned = fx_aligned.reindex(obs_aligned.index)
        ref_values = ref_fx_aligned.tz_convert(tz)
    else:
        ref_values = None

    # Determine series with timezone conversion
    forecast_values = fx_aligned.tz_convert(tz)
    observation_values = obs_aligned.tz_convert(tz)

    if isinstance(fx_series, pd.DataFrame):
        undefined_fx = fx_series.isna().sum().sum()
    else:
        undefined_fx = fx_series.isna().sum()

    # Return dict summarizing results
    results = {
        type(fx).__name__ + " " + DISCARD_DATA_STRING:
            len(fx_series.dropna(how="any")) - len(fx_aligned),
        type(obs).__name__ + " " + DISCARD_DATA_STRING:
            len(obs_resampled) - len(observation_values),
        type(fx).__name__ + " " + UNDEFINED_DATA_STRING:
            int(undefined_fx),
        type(obs).__name__ + " " + UNDEFINED_DATA_STRING:
            int(obs_series.isna().sum())
    }

    if ref_series is not None:
        k = type(ref_fx).__name__ + " " + UNDEFINED_DATA_STRING
        results[k] = int(ref_values.isna().sum())

    return forecast_values, observation_values, ref_values, results


def _check_ref_fx(fx, ref_fx, ref_series):
    if ref_fx is not None and ref_series is None:
        raise ValueError(
            'ref_series must be supplied if fx_obs.reference_forecast is not'
            'None')
    elif ref_fx is None and ref_series is not None:
        raise ValueError(
            'ref_series was supplied but fx_obs.reference_forecast is None')

    if ref_fx is not None:
        if fx.interval_length != ref_fx.interval_length:
            raise ValueError(
                'forecast.interval_length %s must match '
                'reference_forecast.interval_length %s',
                fx.interval_length, ref_fx.interval_length)
        if fx.interval_label != ref_fx.interval_label:
            raise ValueError(
                'forecast.interval_label %s must match '
                'reference_forecast.interval_label %s',
                fx.interval_label, ref_fx.interval_label)
        if isinstance(fx, datamodel.ProbabilisticForecast):
            if fx.axis != ref_fx.axis:
                raise ValueError(
                    'forecast.axis %s must match reference_forecast.axis %s',
                    fx.axis, ref_fx.axis)


def exclude(values, quality_flags=None):
    """
    Return a timeseries with all questionable values removed.
    All NaN values will be removed first and then iff `quality_flag` is set
    (not 0) the corresponding values will also be removed.

    Parameters
    ----------
    values : pandas.Series
        Timeseries values.
    quality_flags : pandas.DataFrame
        Timeseries of quality flags. Default is None.

    Returns
    -------
    pandas.Series :
        Timeseries of values excluding non-quality values.
    """
    # Missing values
    bad_idx = values.isna()

    # Handle quality flags
    if quality_flags is not None:
        consolidated_flag = quality_flags.any(axis=1)
        bad_quality_idx = (consolidated_flag != 0)
        bad_idx = bad_idx | bad_quality_idx

    return values[~bad_idx]


def _merge_quality_filters(filters):
    """Merge any quality flag filters into one single QualityFlagFilter"""
    combo = set()
    for filter_ in filters:
        if isinstance(filter_, datamodel.QualityFlagFilter):
            combo |= set(filter_.quality_flags)
    return datamodel.QualityFlagFilter(tuple(combo))


def process_forecast_observations(forecast_observations, filters, data,
                                  timezone):
    """
    Convert ForecastObservations into ProcessedForecastObservations
    applying any filters and resampling to align forecast and observation.

    Parameters
    ----------
    forecast_observations : list of solarforecastarbiter.datamodel.ForecastObservation, solarforecastarbiter.datamodel.ForecastAggregate
        Pairs to process
    filters : list of solarforecastarbiter.datamodel.BaseFilter
        Filters to apply to each pair.
    data : dict
        Dict with keys that are the Forecast/Observation/Aggregate object
        and values that are the corresponding pandas.Series/DataFrame for
        the object. Keys must also include all Forecast objects assigned
        to the ``reference_forecast`` attributes of the
        ``forecast_observations``.
    timezone : str
        Timezone that data should be converted to

    Returns
    -------
    list of ProcessedForecastObservation
    """  # NOQA
    if not all([isinstance(filter_, datamodel.QualityFlagFilter)
                for filter_ in filters]):
        logger.warning(
            'Only filtering on Quality Flag is currently implemented')
    qfilter = _merge_quality_filters(filters)
    validated_observations = {}
    processed_fxobs = {}
    for fxobs in forecast_observations:
        # validate observation or aggregate data
        if fxobs.data_object not in validated_observations:
            try:
                obs_ser, counts = apply_validation(
                    data[fxobs.data_object],
                    qfilter,
                    exclude)
            except Exception as e:
                logger.error(
                    'Failed to validate data for %s. %s',
                    fxobs.data_object.name, e)
                # store empty data in validated_observations
                preproc_results = (datamodel.PreprocessingResult(
                    name=VALIDATION_RESULT_TOTAL_STRING,
                    count=-1), )
                validated_observations[fxobs.data_object] = (
                    pd.Series([], name='value', index=pd.DatetimeIndex(
                        [], name='timestamp', tz='UTC'), dtype=float),
                    (), preproc_results)
            else:
                # store validated data in validated_observations
                val_results = tuple(datamodel.ValidationResult(flag=k, count=v)
                                    for k, v in counts.items())
                preproc_results = (datamodel.PreprocessingResult(
                    name=VALIDATION_RESULT_TOTAL_STRING,
                    count=len(data[fxobs.data_object]) - len(obs_ser)), )
                validated_observations[fxobs.data_object] = (
                    obs_ser, val_results, preproc_results)

        obs_ser, val_results, preproc_results = (
            validated_observations[fxobs.data_object])

        # resample and align observations to forecast, create
        # ProcessedForecastObservation
        fx_ser = data[fxobs.forecast]
        if fxobs.reference_forecast is not None:
            ref_ser = data[fxobs.reference_forecast]
        else:
            ref_ser = None
        try:
            forecast_values, observation_values, ref_fx_values, results = \
                resample_and_align(fxobs, fx_ser, obs_ser, ref_ser, timezone)
            preproc_results += tuple(datamodel.PreprocessingResult(
                name=k, count=v) for k, v in results.items())
        except Exception as e:
            logger.error(
                'Failed to resample and align data for pair (%s, %s): %s',
                fxobs.forecast.name, fxobs.data_object.name, e)
        else:
            logger.info('Processed data successfully for pair (%s, %s)',
                        fxobs.forecast.name, fxobs.data_object.name)
            name = _name_pfxobs(processed_fxobs.keys(),
                                fxobs.forecast.name)
            processed = datamodel.ProcessedForecastObservation(
                name=name,
                original=fxobs,
                interval_value_type=fxobs.forecast.interval_value_type,
                interval_length=fxobs.forecast.interval_length,
                interval_label=fxobs.forecast.interval_label,
                valid_point_count=len(forecast_values),
                validation_results=val_results,
                preprocessing_results=preproc_results,
                forecast_values=forecast_values,
                observation_values=observation_values,
                reference_forecast_values=ref_fx_values,
                normalization_factor=fxobs.normalization,
                uncertainty=fxobs.uncertainty
                )
            processed_fxobs[name] = processed
    return tuple(processed_fxobs.values())


def _name_pfxobs(current_names, forecast_name, i=1):
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
