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


def resample():
    pass


def realign():
    pass


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
