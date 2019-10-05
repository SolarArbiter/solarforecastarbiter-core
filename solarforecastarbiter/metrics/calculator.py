"""
Metric calculation functions.

Right now placeholder so we can delete report.metrics.py.
Needs to cleaned up and expanded.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import deterministic


def calculate_metrics_for_processed_pairs(processed_fxobs):
    """
    Loop through the forecast-observation pairs and calculate metrics.

    Parameters
    ----------
    processed_fxobs : list
        List of solarforecastarbiter.datamodel.ForecastObservation

    Returns
    -------
    dict
        Multi-level dictionary of metrics by category.

    """
    calc_metrics = []
    for fxobs_ in processed_fxobs:
        metrics_ = calculate_metrics(fxobs_.original,
                                     fxobs_.forecast_values,
                                     fxobs_.observation_values)
        calc_metrics.append(metrics_)
    return calc_metrics


def calculate_metrics(forecast_observation, fx_values, obs_values):
    """TODO: UPDATE"""
    metrics = defaultdict(dict)
    metrics['name'] = forecast_observation.forecast.name

    if fx_values.empty or obs_values.empty:
        for category in ["total", "month", "day", "hour"]:
            for metric in ["mae", "mbe", "rmse"]:
                metrics[category][metric] = np.nan
    else:
        df = pd.concat([obs_values, fx_values], axis=1)
        df.columns = ["obs_values", "fx_values"]

        metrics["total"]["mae"] = deterministic.mean_absolute(
            df.obs_values, df.fx_values
        )
        metrics["total"]["mbe"] = deterministic.mean_bias(
            df.obs_values, df.fx_values
        )
        metrics["total"]["rmse"] = deterministic.root_mean_square(
            df.obs_values, df.fx_values
        )

        # monthly metrics
        mae, mbe, rmse = [], [], []
        for _, group in df.groupby(df.index.month):
            mae.append(
                deterministic.mean_absolute(df.obs_values, df.fx_values)
            )
            mbe.append(
                deterministic.mean_bias(df.obs_values, df.fx_values)
            )
            rmse.append(
                deterministic.root_mean_square(df.obs_values, df.fx_values)
            )
        metrics["month"]["mae"] = mae
        metrics["month"]["mbe"] = mbe
        metrics["month"]["rmse"] = rmse

        # daily metrics
        mae, mbe, rmse = [], [], []
        for _, group in df.groupby(df.index.date):
            mae.append(
                deterministic.mean_absolute(df.obs_values, df.fx_values)
            )
            mbe.append(
                deterministic.mean_bias(df.obs_values, df.fx_values)
            )
            rmse.append(
                deterministic.root_mean_square(df.obs_values, df.fx_values)
            )
        metrics["day"]["mae"] = mae
        metrics["day"]["mbe"] = mbe
        metrics["day"]["rmse"] = rmse

        # hourly metrics
        mae, mbe, rmse = [], [], []
        for _, group in df.groupby(df.index.hour):
            mae.append(
                deterministic.mean_absolute(df.obs_values, df.fx_values)
            )
            mbe.append(
                deterministic.mean_bias(df.obs_values, df.fx_values)
            )
            rmse.append(
                deterministic.root_mean_square(df.obs_values, df.fx_values)
            )
        metrics["hour"]["mae"] = mae
        metrics["hour"]["mbe"] = mbe
        metrics["hour"]["rmse"] = rmse

    return metrics
