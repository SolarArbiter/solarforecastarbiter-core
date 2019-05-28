"""
Provides the dictionary structure for the results returned by the metrics'
evaluator.

It can include:
- pandas.Series of the timeseries of observations, forecasts, errors
- any messages of the preprocessing of the timeseries data
- values of the desired metrics over the entire timespan
- values of the desired metrics by desired groupings

"""

import pandas as pd


# Result dictionary that can be associated with each observation and forecast
_PREPROCESSING_RESULT = {
    'fill_value' : None,
    'fill_method' : None,
    'max_fill_intervals' : None,
    'missing_timestamps' : None,
    'filled_timestamps' : None,
    'excluded_timestamps' : None
}

# Main result diction
EVALUATOR_RESULTS = {
    
    # Timeseries
    'timeseries' : {
        'observations' : None,
        'forecasts' : None
    },
    
    # Metrics
    'metrics' : {
        'total' : {}
    }

}


def create_empty_results(context):
    """Get an empty results dictionary from context."""
    raise NotImplementedError


def get_number_of_missing_timestamps(results, type='both'):
    """Get the number of missing timestamps."""
    raise NotImplementedError


def print_results_string(results):
    """Pretty print results to string."""
    raise NotImplementedError


def show_results(results):
    """Print results to screen."""
    print(print_results_string(results))