"""
Provides preprocessing steps to be performed on the timeseries data.
"""

import datetime
import numpy as np
import pandas as pd

from solarforecastarbiter.metrics import context, errors

def preprocess(preprocess_func):
    """Preprocessing decorator."""
    # TODO implement this decorator and use it for the rest of 
    # the methods here. 
    raise NotImplementedError
    

def exclude(values, quality_flags=None, **kwargs):
    """
    Return a timeseries with all questionable values removed.
    
    All NaN values will be removed first then iff quality_flag is set the 
    corresponding values will also be removed.
    
    Parameters
    ----------
    values : pd.Series
        Timeseries values
    quality_flags : pd.Series
        Timeseries quality flags, Default None
    
    Returns
    -------
    pd.Series :
        Timeseries of values excluding questionable values.
    
    Raises
    ------
    SfaMetricsInputEror if the `values` and `quality_flags` do not match.
    """
    
    # Missing values
    bad_idx = values.isna()

    # Handle quality flags
    if quality_flags is not None:
        bad_quality_idx = quality_flags==1
        bad_idx = bad_idx | bad_quality_idx
    
    # print(bad_idx)
    return values[~bad_idx]


def fill_static_value(values, quality_flags, default_value, threshold):
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
    threshold : int
        The maximum number of consective values to fill.
    
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


def fill_forward(values, quality_flags, threshold):
    """
    Return a timeseries with fillin of the questionable values (quality-flag = 1) 
    with the given the last  value.
    
    Parameters
    ----------
    values : pd.Series
        Timeseries values
    quality_flags : pd.Series
        Timeseries quality flags
    threshold : int
        The maximum number of consective values to fill.
    
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
