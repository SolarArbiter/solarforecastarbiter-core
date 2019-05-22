"""
Provides evaluation of observations to forecasts. 

Provides tools for evaluation of forecast performance from a given metrics' 
context :py:mod:`solarforecastarbiter.metrics.context` and returns the results 
:py:mod:`solarforecastarbiter.metrics.results`.
"""

import datetime
import numpy as np
import pandas as pd

from solarforecastarbiter.metrics import context, results, deterministic, probabilistic, event


def evaluate(observations, forecasts, context):
    """
    Evaluate the performance of the forecasts to the observations using the 
    context to define the preprocessing, metrics and results to returnself.
    
    Parameters
    ----------
    observations : tuple of pd.Series
    
    Returns
    -------
    dict
    
    Raises
    ------
    
    """
    pass

