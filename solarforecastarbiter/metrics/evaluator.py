"""
Provides evaluation of observations to forecasts. 

Provides tools for evaluation of forecast performance from a given metrics' 
context :py:mod:`solarforecastarbiter.metrics.context` and returns the results 
:py:mod:`solarforecastarbiter.metrics.results`.
"""

import datetime
import numpy as np
import pandas as pd

from solarforecastarbiter.metrics import context, results, deterministic, probabilistic, event, errors


def evaluate(observations, forecasts, context):
    """
    Evaluate the performance of the forecasts to the observations using the 
    context to define the preprocessing, metrics and results to returnself.
    
    Parameters
    ----------
    observations : pd.DataFrame
        observations are a DataFrame of values and quality flags with timestamp as index
    forecasts : pd.Series
        forecasts are a Series of values with timestamp as index
    context : dict
        a context dictionary as defined in :py:mod:`solarforecastarbiter.metrics.context`
    
    Returns
    -------
    dict
        a results dictionary as defined in :py:mod:`solarforecastarbiter.metrics.results`
    
    Raises
    ------
    
    """
    
    # Verify input - probably isn't necessary
    if not are_validate_observations(observations):
        raise errors.SfaMetricsInputError("Observations must be a pandas DataFrame \ 
                                          with value and quality_flag columns \
                                          and an index of datetimes.")
    
    if not are_valid_forecasts(forecasts):
        raise errors.SfaMetricsInputError("Forecasts must be a pandas Series \
                                          with value \
                                          and an index of datetimes")
    
    # Preprocessing
    obs_context = context['preprocessing']['observations']
    
    # TODO: replace this with mapping to preprocessing functions and add decorators
    if obs_context == 'exclude': 
        obs_values = method(observations.values, 
                            observations.quality_flags)
    
    for key,val context['metrics']:
        

def are_valid_observations(observations):
    """Validate observations in expected format."""
    if not isinstance(observations, pd.DataFrame):
        return False
    if not ['value','quality_flag'] in observations.columns:
        return False
    if not observations.index.is_all_dates:
        return False
    return True


def are_valid_forecasts(forecasts):
    """Validate forecasts are in expected format."""
    if not isinstace(forecasts, pd.Series)
        return False
    if not forecasts.index.is_all_dates:
        return False
    return True
