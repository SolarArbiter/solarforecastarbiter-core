"""
Provides preprocessing steps to be performed on the timeseries data.
"""


from solarforecastarbiter import datamodel
from solarforecastarbiter.validation import quality_mapping


def apply_validation(data, qfilter, handle_func):
    """
    Apply validation steps based on provided filters to the data.

    Parameters
    ----------
    data : dict
        Keys are Observation and Forecast models and values are
        the timeseries data as pandas.Series.
    qfilter : solarforecastarbiter.datamodel.QualityFlagFilter
    handle_func : function
        Function that handles how `quality_flags` will be used.
        See solarforecastarbiter.metrics.preprocessing.exclude as an
        example.

    Returns
    -------
    dict :
        Keys are Observation and Forecast models and values
        the validated timeseries data as pandas.Series.
    """
    validated_data = {}

    # List of flags from filter
    if not isinstance(qfilter, datamodel.QualityFlagFilter):
        raise TypeError(f"{qfilter} not a QualityFlagFilter")
    filters = qfilter.quality_flags

    # Apply handling function to quality flags
    for model, values in data.items():

        # Apply only to Observations
        if isinstance(model, (datamodel.Observation, datamodel.Aggregate)):
            if values.empty:
                validated_data[model] = values.value
            else:
                validation_df = quality_mapping.convert_mask_into_dataframe(
                    values['quality_flag'])
                validation_df = validation_df[list(filters)]
                validated_data[model] = handle_func(values.value,
                                                    validation_df)
        elif isinstance(model, datamodel.Forecast):
            validated_data[model] = values
        else:
            raise TypeError(
                f"{model} not an Observation, Aggregate, or Forecast")

    return validated_data


def resample_and_align(fx_obs, data, tz):
    """
    Resample the observation to the forecast interval length and align to
    remove overlap.

    Parameters
    ----------
    fx_obs : solarforecastarbiter.datamodel.ForecastObservation, solarforecastarbiter.datamodel.ForecastAggregate
        Pair of forecast and observation.
    data : dict
        Keys are Observation and Forecast models and values
        the validated timeseries data as pandas.Series.
    tz : str
        Timezone to witch processed data will be converted.

    Returns
    -------
    solarforecastarbiter.datamodel.ProcessedForecastObservation

    Todo
    ----
      * Add other resampling functions (besides mean like first, last, median)
    """  # noqa: E501
    fx = fx_obs.forecast
    obs = fx_obs.data_object

    # Resample observation
    closed = datamodel.CLOSED_MAPPING[fx.interval_label]
    obs_resampled = data[obs].resample(fx.interval_length,
                                       label=closed,
                                       closed=closed).mean()

    # Align (forecast is unchanged)
    # Remove non-corresponding observations and
    # forecasts, and missing periods
    obs_resampled = obs_resampled.dropna(how="any")
    obs_aligned, fx_aligned = obs_resampled.align(data[fx].dropna(how="any"),
                                                  'inner')

    # Determine series with timezone conversion
    forecast_values = fx_aligned.tz_convert(tz)
    observation_values = obs_aligned.tz_convert(tz)

    # Create ProcessedForecastObservation
    processed_fx_obs = datamodel.ProcessedForecastObservation(
        original=fx_obs,
        interval_value_type=fx.interval_value_type,
        interval_length=fx.interval_length,
        interval_label=fx.interval_label,
        forecast_values=forecast_values,
        observation_values=observation_values)

    return processed_fx_obs


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
