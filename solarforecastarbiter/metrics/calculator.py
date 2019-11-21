"""
Metric calculation functions.

Right now placeholder so we can delete report.metrics.py.
Needs to cleaned up and expanded.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from solarforecastarbiter.metrics import deterministic


def calculate_metrics_for_processed_pairs(processed_fxobs):
    """
    Loop through the forecast-observation pairs and calculate metrics.

    Parameters
    ----------
    processed_fxobs : list
        List of solarforecastarbiter.datamodel.ProcessedForecastObservation.

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
        mae, mbe, rmse = {}, {}, {}
        for idx, group in df.groupby(df.index.month):
            mae[idx] = deterministic.mean_absolute(
                group.obs_values, group.fx_values)
            mbe[idx] = deterministic.mean_bias(
                group.obs_values, group.fx_values)
            rmse[idx] = deterministic.root_mean_square(
                group.obs_values, group.fx_values)
        metrics["month"]["mae"] = pd.Series(mae)
        metrics["month"]["mbe"] = pd.Series(mbe)
        metrics["month"]["rmse"] = pd.Series(rmse)

        # daily metrics
        mae, mbe, rmse = {}, {}, {}
        for idx, group in df.groupby(df.index.date):
            mae[idx] = deterministic.mean_absolute(
                group.obs_values, group.fx_values)
            mbe[idx] = deterministic.mean_bias(
                group.obs_values, group.fx_values)
            rmse[idx] = deterministic.root_mean_square(
                group.obs_values, group.fx_values)
        metrics["day"]["mae"] = pd.Series(mae)
        metrics["day"]["mbe"] = pd.Series(mbe)
        metrics["day"]["rmse"] = pd.Series(rmse)

        # hourly metrics
        mae, mbe, rmse = {}, {}, {}
        for idx, group in df.groupby(df.index.hour):
            mae[idx] = deterministic.mean_absolute(
                group.obs_values, group.fx_values)
            mbe[idx] = deterministic.mean_bias(
                group.obs_values, group.fx_values)
            rmse[idx] = deterministic.root_mean_square(
                group.obs_values, group.fx_values)
        metrics["hour"]["mae"] = pd.Series(mae)
        metrics["hour"]["mbe"] = pd.Series(mbe)
        metrics["hour"]["rmse"] = pd.Series(rmse)

    return metrics
