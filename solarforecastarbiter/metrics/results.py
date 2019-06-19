"""
Provides the dictionary structure for the results returned by the metrics'
evaluator.

It can include:
- pandas.Series of the timeseries of observations, forecasts, errors
- any messages of the preprocessing of the timeseries data
- values of the desired metrics over the entire timespan
- values of the desired metrics by desired groupings

"""

from dataclasses import dataclass
import pandas as pd

from solarforecastarbiter.metrics import context

PREF_METRIC_GROUP_ORDER = ['year', 'month', 'weekday', 'date', 'hour']


# Result dictionary that can be associated with each observation and forecast
_PREPROCESSING_RESULT = {
    'fill_value': None,
    'fill_method': None,
    'max_fill_intervals': None,
    'missing_timestamps': None,
    'filled_timestamps': None,
    'excluded_timestamps': None
}


@dataclass(order=False)
class MetricsResult:
    """
    A class that holds the results from a metrics' evaulation.
    
    Attributes
    ----------
    processed_observations : pd.DataFrame
        Observation timeseries after being processed.
    processed_forecasts : pd.DataFrame
        Forecast timeseris after being processed.
    metrics : pd.DataFrame
        Multindex DataFrame of all calculated metrics.
        
    Methods
    -------
    get_number_of_missing_timestamps()
        Get the number of missing timestamps over the processed timeseries.
    
    """
    processed_observations: pd.Series
    processed_forecasts: pd.Series
    metrics_context: dict
    metrics: pd.DataFrame = None

    def construct_empty_metrics(self, metrics_context):
        """
        Construct the pandas.DataFrame with MultiIndex 
        as specifid by the solarforecastarbiter.metrics:`MetricsContext`.
        
        Parameters
        ----------
        metrics_context : solarforecastarbiter.metrics:`MetricsContext`
        """
        dt_groups = []
        metric_groups = []
        
        # timestamp groups
        for groupby, use in metrics_context['results']['groupings'].items():
            if use:
                dt_groups += [groupby]

        dt_groups = [x for x in PREF_METRIC_GROUP_ORDER if x in dt_groups]

        # metrics
        for metric, use in metrics_context['metrics'].items():
            if use:
                metric_groups += [metric]

        metrics_df = pd.DataFrame(columns=dt_groups+metric_groups)
        metrics_df.set_index(dt_groups, inplace=True)

        self.metrics = metrics_df

    def get_number_of_missing_timestamps():
        """Get the number of missing timestamps."""
        raise NotImplementedError

