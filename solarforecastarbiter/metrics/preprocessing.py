"""
Provides preprocessing steps to be performed on the timeseries data.
"""

import numpy as np
import pandas as pd

from solarforecastarbiter import datamodel
from solarforecastarbiter.validation import quality_mapping


def apply_validation(data, qfilter, handle_func):
    """
    Apply validation steps based on provided filters to the data.

    Parameters
    ----------
    qfilter : list of
        :class:`solarforecastarbiter.datamodel.QualityFlagFilter`
    data : :class:`pd.DataFrame`
        Pandas DataFrame of observations and forecasts.
    handle_func : function
        Function that handles how `quality_flags` will be used.

    Returns
    -------
    pd.DataFrame
    """
    validated_data = {}

    # List of flags from filter
    if not isinstance(qfilter, datamodel.QualityFlagFilter):
        return TypeError(f"{filters} not a QualityFlagFilter")
    filters = qfilter.quality_flags

    # Apply handling function to quality flags
    for model, values in data.items():

        # Skip if empty
        if isinstance(values, pd.DataFrame) and values.empty:
            validated_data[model] = values.value
            continue

        # Apply only to Observations
        if isinstance(model, datamodel.Observation):
            validation_df = quality_mapping.convert_mask_into_dataframe(
                values['quality_flag'])
            validation_df = validation_df[filters]
            validated_data[model] = handle_func(values.value,
                                                validation_df.any(axis=1))
        elif isinstance(model, datamodel.Forecast):
            validated_data[model] = values
        else:
            raise TypeError(f"{model} not an Observation or Forecast")

    return validated_data


def resample_and_align(fx_obs, data, tz):
    """
    Resample to the forecast and observation using the the larger interval
    length  and align them to overlap.

    Parameters
    ----------
    fx_obs : :class:`solarforecastarbiter.datamodel.ForecastObservation`
        Pair of forecast and observation.
    data : dict
        Dictionary that must include the timeseries of the pair.
    tz : str
        Timezone to witch processed data will be converted.

    Returns
    -------
    :class:`solarforecastarbiter.datamodel.ProcessedForecastObservation`

    Todo
    ----
      * Add ability to set use smaller interval length
      * Add other resampling functions (besides mean like first, last, median)
    """
    fx = fx_obs.forecast
    obs = fx_obs.observation

    # Check that the forecast and observation models exist in the data
    assert fx in data.keys(), f"Forecast not found in data."
    assert obs in data.keys(), f"Observation not found in data."

    # Get minimum and maximum datetimes???
    # min_dt = max(fx_data.index.min(), obs_data.index.min())
    # max_dt = min(fx_data.index.max(), obs_data.index.max())

    # Determine larger interval length
    if fx.interval_length >= obs.interval_length:
        inherit = fx
        downsample = obs
    else:
        inherit = obs
        downsample = fx

    # Resample
    label = datamodel.CLOSED_MAPPING[inherit.interval_label]
    down_resampled = data[downsample].resample(inherit.interval_length,
                                               label=label).mean()
    inherit_resampled = data[inherit].resample(inherit.interval_length,
                                               label=label).mean()

    # Determine series with timezone conversion
    if isinstance(inherit, datamodel.Forecast):
        forecast_values = inherit_resampled.tz_convert(tz)
        observation_values = down_resampled.tz_convert(tz)
    else:
        observation_values = inherit_resampled.tz_convert(tz)
        forecast_values = down_resampled.tz_convert(tz)

    # Create ProcessedForecastObservation
    processed_fx_obs = datamodel.ProcessedForecastObservation(
        original=fx_obs,
        interval_value_type=inherit.interval_value_type,
        interval_length=inherit.interval_length,
        interval_label=inherit.interval_label,
        forecast_values=forecast_values,
        observation_values=observation_values)

    return processed_fx_obs


def exclude(values, quality_flags=None):
    """
    Return a timeseries with all questionable values removed.
    All NaN values will be removed first and then iff quality_flag is set
    (not 0) the corresponding values will also be removed.

    Parameters
    ----------
    values : :class:`pd.Series`
        Timeseries values.
    quality_flags : :class:`pd.Series`
        Timeseries quality flag [0,1], default None.

    Returns
    -------
    :class:`pd.Series` :
        Timeseries of values excluding non-quality values.
    """

    # Missing values
    bad_idx = values.isna()

    # Handle quality flags
    if quality_flags is not None:
        bad_quality_idx = (quality_flags != 0)
        bad_idx = bad_idx | bad_quality_idx

    print(bad_idx)
    return values[~bad_idx]
