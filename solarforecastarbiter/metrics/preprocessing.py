"""
Provides preprocessing steps to be performed on the timeseries data.
"""

import datetime
import numpy as np
import pandas as pd

from solarforecastarbiter.metrics import context, errors


def exclude(values, quality_flags):
    """
    Return a timeseries with all questionable values removed (quality-flag = 1).
    
    Parameters
    ----------
    values : pd.Series
        Timeseries values
    quality_flags : pd.Series
        Timeseries quality flags
    
    Returns
    -------
    pd.Series :
        Timeseries of values excluding questionable values.
    
    Raises
    ------
    SfaMetricsInputEror if the `values` and `quality_flags` do not match.
    """
    raise NotImplementedError


def fill_static_value(values, quality_flags, default_value):
    """
    Return a timeseries with fillin of the questionable values (quality-flag = 1) 
    with the given default value.
    
    Parameters
    ----------
    values : pd.Series
        Timeseries values
    quality_flags : pd.Series
        Timeseries quality flags
    default_value : float
        The value to use for all fillins.
    
    Returns
    -------
    pd.Series :
        Timeseries of values with fillin.
    
    Raises
    ------
    SfaMetricsInputEror if the `values` and `quality_flags` do not match.
    TypeError if default_value is not of type float.
    """
    raise NotImplementedError


def fill_by_interpolation(values, quality_flags, threshold):
    """
    Return a timeseries with fillin of the questionable values (quality-flag = 1) 
    by interpolation as long as the number of consective values is less than
    the threshold, otherwise they will be excluded.
    
    Parameters
    ----------
    values : pd.Series
        Timeseries values
    quality_flags : pd.Series
        Timeseries quality flags
    threshold : int
        The maximum number of consective values to interpolate over.
    
    Returns
    -------
    pd.Series :
        Timeseries of values with fillin.
    
    Raises
    ------
    SfaMetricsInputEror if the `values` and `quality_flags` do not match.
    TypeError if `threshold` is not of type int.
    """
    raise NotImplementedError
