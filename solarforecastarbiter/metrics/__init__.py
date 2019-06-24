"""
To be deleted and replaced by
https://github.com/SolarArbiter/solarforecastarbiter-core/pull/102
"""
from collections import defaultdict
import datetime as dt
import json

import numpy as np
import pandas as pd

from solarforecastarbiter import datamodel
from solarforecastarbiter.validation.quality_mapping import \
    convert_mask_into_dataframe


def validate_resample_align(report, data):
    data_validated = apply_validation(report, data)
    processed_fxobs = resample_realign(report, data_validated)
    return processed_fxobs


def apply_validation(report, data):
    data_validated = {}
    # datamodel doesn't restrict number of filters of given type
    qc_filters = [f.quality_flags for f in report.filters if
                  isinstance(f, datamodel.QualityFlagFilter)]
    for fx_or_ob, values in data.items():
        # if empty?
        if isinstance(fx_or_ob, datamodel.Observation):
            _data_validated = values
            # what if no filters? make sure values taken
            for flags in qc_filters:
                validation_df = convert_mask_into_dataframe(
                    values['quality_flag'])
                _data_validated = (_data_validated['value'].where(
                    ~validation_df[list(flags)].any(axis=1)))
            data_validated[fx_or_ob] = _data_validated
        else:
            # assume it's a forecast
            data_validated[fx_or_ob] = values

    return data_validated


def resample_realign(report, data):
    """Probably apply validation and other filters before this"""
    processed = []
    data_resampled = {}
    for fxobs in report.forecast_observations:
        # for now just resample to forecast
        interval_label = fxobs.forecast.interval_label
        interval_value_type = fxobs.forecast.interval_value_type
        interval_length = fxobs.forecast.interval_length
        # if this obs has not already been resampled, do it
        if fxobs.observation not in data_resampled:
            label = datamodel.CLOSED_MAPPING[interval_label]
            resampled = data[fxobs.observation].resample(
                interval_length, label=label).mean()
            data_resampled[fxobs.observation] = resampled
        # no resampling allowed for forecasts for now
        processed_fxobs = datamodel.ProcessedForecastObservation(
            original=fxobs, interval_value_type=interval_value_type,
            interval_length=interval_length, interval_label=interval_label,
            forecast_values=data[fxobs.forecast],
            observation_values=data_resampled[fxobs.observation])
        processed.append(processed_fxobs)
    return processed


def loop_forecasts_calculate_metrics(report, processed_fxobs):
    metrics = []
    for fxobs_ in processed_fxobs:
        metrics_ = calculate_metrics(fxobs_.original, fxobs_.forecast_values,
                                     fxobs_.observation_values)
        metrics.append(metrics_)
    return metrics


def calculate_metrics(forecast_observation, fx_values, obs_values):
    metrics = defaultdict(dict)
    metrics['name'] = forecast_observation.forecast.name
    diff = fx_values - obs_values
    metrics['total']['mae'] = diff.abs().mean()
    _rmse = diff.aggregate(rmse)
    metrics['total']['rmse'] = _rmse
    metrics['total']['mbe'] = diff.mean()
    metrics['day']['mae'] = diff.abs().groupby(lambda x: x.date).mean()
    _rmse = diff.groupby(lambda x: x.date).aggregate(rmse)
    metrics['day']['rmse'] = _rmse
    metrics['day']['mbe'] = diff.groupby(lambda x: x.date).mean()
    metrics['month']['mae'] = diff.abs().groupby(lambda x: x.month).mean()
    _rmse = diff.groupby(lambda x: x.month).aggregate(rmse)
    metrics['month']['rmse'] = _rmse
    metrics['month']['mbe'] = diff.groupby(lambda x: x.month).mean()
    metrics['hour']['mae'] = diff.abs().groupby(lambda x: x.hour).mean()
    _rmse = diff.groupby(lambda x: x.hour).aggregate(rmse)
    metrics['hour']['rmse'] = _rmse
    metrics['hour']['mbe'] = diff.groupby(lambda x: x.hour).mean()
    return metrics


# not ideal, but works for now
class MetricsEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (pd.Series, pd.DataFrame)):
            return json.loads(o.to_json(date_format='iso',
                                        date_unit='s',
                                        double_precision=3))
        return super().default(o)


def rmse(diff):
    return np.sqrt((diff * diff).mean())
