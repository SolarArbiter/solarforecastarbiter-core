"""
To be deleted and replaced by
https://github.com/SolarArbiter/solarforecastarbiter-core/pull/102
"""
import numpy as np

from collections import defaultdict

from solarforecastarbiter import datamodel
from solarforecastarbiter.validation.quality_mapping import \
    convert_mask_into_dataframe


def validate_resample_align(report, data):
    data_validated = apply_validation(report, data)
    fxobs_resampled, data_resampled = resample_realign(report, data_validated)
    return fxobs_resampled, data_resampled


def apply_validation(report, data):
    data_validated = {}
    # datamodel doesn't restrict number of filters of given type
    qc_filters = [f.quality_flags for f in report.filters if
                  isinstance(f, datamodel.QualityFlagFilter)]
    for fx_or_ob, values in data.items():
        if isinstance(fx_or_ob, datamodel.Observation):
            _data_validated = values
            for flags in qc_filters:
                validation_df = convert_mask_into_dataframe(
                    values['quality_flag'])
                _data_validated = (_data_validated['value'].where(
                    ~validation_df[flags].any(axis=1)))
            data_validated[fx_or_ob] = _data_validated
        else:
            # assume it's a forecast
            data_validated[fx_or_ob] = values

    return data_validated


def resample_realign(report, data):
    """Probably apply validation and other filters before this"""
    data_resampled = {}
    fxobs_resampled = []
    for fxobs in report.forecast_observations:
        obs_resampled = match_observation_to_forecast_intervals(
            fxobs.observation, fxobs.forecast)
        _fxobs_resampled = datamodel.ForecastObservation(
            fxobs.forecast, obs_resampled)
        fxobs_resampled.append(_fxobs_resampled)
        # if this obs has not already been resampled, do it
        if obs_resampled not in data_resampled:
            label = datamodel.CLOSED_MAPPING[obs_resampled.interval_label]
            resampled = data[fxobs.observation].resample(
                obs_resampled.interval_length, label=label).mean()
            data_resampled[obs_resampled] = resampled
        # no resampling allowed for forecasts for now
        data_resampled[fxobs.forecast] = data[fxobs.forecast]

    return fxobs_resampled, data_resampled


def match_observation_to_forecast_intervals(observation, forecast):
    # forecast intervals are 1h and labels are beginning
    # make new observation that matches forecasts and resample data
    obs_dict = observation.to_dict()
    obs_dict['interval_label'] = forecast.interval_label
    obs_dict['interval_length'] = forecast.interval_length
    # https://github.com/SolarArbiter/solarforecastarbiter-core/issues/125
    obs_dict['site'] = datamodel.Site.from_dict(obs_dict['site'])
    observation_resampled = datamodel.Observation.from_dict(obs_dict)
    return observation_resampled


def loop_forecasts_calculate_metrics(fxobs, data):
    metrics = []
    for fxobs_ in fxobs:
        metrics_ = calculate_metrics(fxobs_, data[fxobs_.forecast],
                                     data[fxobs_.observation])
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


def rmse(diff):
    return np.sqrt((diff * diff).sum() / (len(diff) - 1))
