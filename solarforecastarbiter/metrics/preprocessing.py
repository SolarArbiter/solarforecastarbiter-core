"""
Provides preprocessing steps to be performed on the timeseries data.
"""


from solarforecastarbiter import datamodel
from solarforecastarbiter.validation import quality_mapping


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


def resample_and_align(fx_obs, fx_series, obs_series, tz):
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
        Timeseries data of the observation/aggregate after processing the quality flag column.
    tz : str
        Timezone to which processed data will be converted.

    Returns
    -------
    forecast_values : pandas.Series
    observation_values : pandas.Series

    Todo
    ----
      * Add other resampling functions (besides mean like first, last, median)
    """  # noqa: E501
    fx = fx_obs.forecast
    obs = fx_obs.data_object

    # Resample observation, checking for invalid interval_length and that the
    # Series has data:
    if fx.interval_length > obs.interval_length and not obs_series.empty:
        closed = datamodel.CLOSED_MAPPING[fx.interval_label]
        obs_resampled = obs_series.resample(
            fx.interval_length,
            label=closed,
            closed=closed
        ).agg(["mean", "count"])

        # Drop intervals if too many samples missing
        count_threshold = int(fx.interval_length / obs.interval_length * 0.1)
        obs_resampled = obs_resampled["mean"].where(
            obs_resampled["count"] >= count_threshold
        )
    elif fx.interval_length < obs.interval_length:
        raise ValueError('observation.interval_length cannot be greater than '
                         'forecast.interval_length.')
    else:
        obs_resampled = obs_series

    # Align (forecast is unchanged)
    # Remove non-corresponding observations and
    # forecasts, and missing periods
    obs_resampled = obs_resampled.dropna(how="any")
    obs_aligned, fx_aligned = obs_resampled.align(fx_series.dropna(how="any"),
                                                  'inner')

    # Determine series with timezone conversion
    forecast_values = fx_aligned.tz_convert(tz)
    observation_values = obs_aligned.tz_convert(tz)
    return forecast_values, observation_values


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
        the object.
    timezone : str
        Timezone that data should be converted to

    Returns
    -------
    list of ProcessedForecastObservation
    """  # NOQA
    if not all([isinstance(filter_, datamodel.QualityFlagFilter)
                for filter_ in filters]):
        # TODO: warn/raise with unused filter
        pass
    qfilter = _merge_quality_filters(filters)
    validated_observations = {}
    processed_fxobs = []
    for fxobs in forecast_observations:
        if fxobs.data_object not in validated_observations:
            obs_ser, counts = apply_validation(
                data[fxobs.data_object],
                qfilter,
                exclude)
            val_results = tuple(datamodel.ValidationResult(flag=k, count=v)
                                for k, v in counts.items())
            validated_observations[fxobs.data_object] = (obs_ser, val_results)

        obs_ser, val_results = validated_observations[fxobs.data_object]
        fx_ser = data[fxobs.forecast]
        forecast_values, observation_values = resample_and_align(
            fxobs, fx_ser, obs_ser, timezone)

        processed = datamodel.ProcessedForecastObservation(
            original=fxobs,
            interval_value_type=fxobs.forecast.interval_value_type,
            interval_length=fxobs.forecast.interval_length,
            interval_label=fxobs.forecast.interval_label,
            valid_point_count=len(forecast_values),
            validation_results=val_results,
            forecast_values=forecast_values,
            observation_values=observation_values)

        processed_fxobs.append(processed)
    return processed_fxobs
