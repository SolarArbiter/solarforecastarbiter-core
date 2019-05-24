from collections import defaultdict
import datetime
from functools import partial

import pandas as pd
import numpy as np

from solarforecastarbiter import datamodel, pvmodel

from solarforecastarbiter.reports import template, figures, main

dummy_metrics = {
    'total': {
        'mae': 5,
        'mbe': 3,
        'rmse': 10
    },
    'month': {
        '201901': {
            'mae': 5,
            'mbe': 3,
            'rmse': 10
        }
    },
    'day': {
        '20190101': {
            'mae': 1,
            'mbe': -1,
            'rmse': 2
        },
        '20190102': {
            'mae': 10,
            'mbe': -5,
            'rmse': 15
        },
        '20190103': {
            'mae': 5,
            'mbe': 1,
            'rmse': 7
        }
    },
    'hour': {
        1: {
            'mae': 1,
            'mbe': -1,
            'rmse': 2
        },
        2: {
            'mae': 1,
            'mbe': -1,
            'rmse': 2
        },
        3: {
            'mae': 1,
            'mbe': -1,
            'rmse': 2
        },
    }
}


def load_data_base(data, observation, data_start, data_end):
    # slice doesn't care about closed or interval label
    # so here we manually adjust start and end times
    if 'instant' in observation.interval_label:
        pass
    elif observation.interval_label == 'ending':
        data_start += pd.Timedelta('1s')
    elif observation.interval_label == 'beginning':
        data_end -= pd.Timedelta('1s')
    return data[data_start:data_end]


def _site_metadata():
    metadata = datamodel.Site(
        name='Albuquerque Baseline', latitude=35.05, longitude=-106.54,
        elevation=1657.0, timezone='America/Denver')
    return metadata


site = _site_metadata()
tz = 'America/Phoenix'
data_index = pd.date_range(
    start='20190101', end='20190401', freq='5min', tz=tz, closed='left')
solar_position = pvmodel.calculate_solar_position(
    site.latitude, site.longitude, site.elevation, data_index)
data_cs = pvmodel.calculate_clearsky(
    site.latitude, site.longitude, site.elevation,
    solar_position['apparent_zenith'])
data = data_cs['ghi'] * 0.8

data_cs_1h = data_cs.resample('1h').mean()
data_1h = data.resample('1h').mean()


interval_label = 'instant'


def ghi_observation_metadata(site_metadata):
    ghi_meta = datamodel.Observation(
        name='Albuquerque Baseline GHI', variable='ghi',
        interval_value_type='average', interval_length=pd.Timedelta('5min'),
        interval_label=interval_label, site=site_metadata, uncertainty=1)
    return ghi_meta


def ghi_forecast_metadata(site_metadata):
    ac_power_meta = datamodel.Forecast(
        name='Albuquerque Baseline GHI Fx', variable='ghi',
        issue_time_of_day=datetime.time(0, 0),  # issued on the hour
        lead_time_to_start=pd.Timedelta('0h'),
        run_length=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        interval_label='beginning',
        interval_value_type='mean',
        site=site_metadata)
    return ac_power_meta


def ghi_forecast_metadata2(site_metadata):
    ac_power_meta = datamodel.Forecast(
        name='Albuquerque Baseline GHI Fx2', variable='ghi',
        issue_time_of_day=datetime.time(0, 0),  # issued on the hour
        lead_time_to_start=pd.Timedelta('0h'),
        run_length=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        interval_label='beginning',
        interval_value_type='mean',
        site=site_metadata)
    return ac_power_meta


observation = ghi_observation_metadata(site)
forecast = ghi_forecast_metadata(site)
forecast2 = ghi_forecast_metadata2(site)
load_data = partial(load_data_base, data)

fxobs1 = datamodel.ForecastObservation(forecast, observation)
fxobs2 = datamodel.ForecastObservation(forecast2, observation)

report = datamodel.Report(
    name='Albuquerque GHI Forecast Analysis',
    start=pd.Timestamp('20190101 0000', tz=tz),
    end=pd.Timestamp('20190331 2359', tz=tz),
    forecast_observations=(fxobs1, fxobs2),
    metrics=('mae', 'rmse', 'mbe')
)


obs_values = load_data(observation, '20190101', '20200105')
fx_values = obs_values * (0.5 * np.random.randn(len(obs_values)) + 1)
fx_values = fx_values.clip(0)
data = pd.DataFrame({'observation': obs_values, 'forecast A': fx_values})
data['timestamp'] = data.index

obs_values = load_data(observation, '20190101', '20200105')
fx_values2 = obs_values * (0.5 * np.random.randn(len(obs_values)) + 0.75)
fx_values2 = fx_values2.clip(0)
data = pd.DataFrame({'observation': obs_values, 'forecast B': fx_values2})
data['timestamp'] = data.index

metadata = main.create_metadata(report)


def rmse(diff):
    return np.sqrt((diff * diff).sum() / (len(diff) - 1))

# breakpoint()
metrics_a = defaultdict(dict)
metrics_b = defaultdict(dict)
metrics_a['name'] = forecast.name
metrics_b['name'] = forecast2.name
metrics_a['total']['mae'] = (fx_values - obs_values).abs().mean()
_rmse = (fx_values - obs_values).aggregate(rmse)
metrics_a['total']['rmse'] = _rmse
metrics_a['total']['mbe'] = (fx_values - obs_values).mean()
metrics_b['total']['mae'] = (fx_values2 - obs_values).abs().mean()
_rmse = (fx_values2 - obs_values).aggregate(rmse)
metrics_b['total']['rmse'] = _rmse
metrics_b['total']['mbe'] = (fx_values2 - obs_values).mean()
metrics_a['day']['mae'] = (fx_values - obs_values).abs().groupby(lambda x: x.date).mean()
_rmse = (fx_values - obs_values).groupby(lambda x: x.date).aggregate(rmse)
print(_rmse)
metrics_a['day']['rmse'] = _rmse
metrics_a['day']['mbe'] = (fx_values - obs_values).groupby(lambda x: x.date).mean()
metrics_b['day']['mae'] = (fx_values2 - obs_values).abs().groupby(lambda x: x.date).mean()
_rmse = (fx_values2 - obs_values).groupby(lambda x: x.date).aggregate(rmse)
metrics_b['day']['rmse'] = _rmse
metrics_b['day']['mbe'] = (fx_values2 - obs_values).groupby(lambda x: x.date).mean()
metrics_a['month']['mae'] = (fx_values - obs_values).abs().groupby(lambda x: x.month).mean()
_rmse = (fx_values - obs_values).groupby(lambda x: x.month).aggregate(rmse)
metrics_a['month']['rmse'] = _rmse
metrics_a['month']['mbe'] = (fx_values - obs_values).groupby(lambda x: x.month).mean()
metrics_b['month']['mae'] = (fx_values2 - obs_values).abs().groupby(lambda x: x.month).mean()
_rmse = (fx_values2 - obs_values).groupby(lambda x: x.month).aggregate(rmse)
metrics_b['month']['rmse'] = _rmse
metrics_b['month']['mbe'] = (fx_values2 - obs_values).groupby(lambda x: x.month).mean()
metrics_a['hour']['mae'] = (fx_values - obs_values).abs().groupby(lambda x: x.hour).mean()
_rmse = (fx_values - obs_values).groupby(lambda x: x.hour).aggregate(rmse)
metrics_a['hour']['rmse'] = _rmse
metrics_a['hour']['mbe'] = (fx_values - obs_values).groupby(lambda x: x.hour).mean()
metrics_b['hour']['mae'] = (fx_values2 - obs_values).abs().groupby(lambda x: x.hour).mean()
_rmse = (fx_values2 - obs_values).groupby(lambda x: x.hour).aggregate(rmse)
metrics_b['hour']['rmse'] = _rmse
metrics_b['hour']['mbe'] = (fx_values2 - obs_values).groupby(lambda x: x.hour).mean()
metrics = (metrics_a, metrics_b)

prereport = template.prereport(report, metadata, metrics)

# optionally post prereport to API

with open('bokeh_prereport.md', 'w') as f:
    f.write(prereport)

fx_obs_cds = [
    (fxobs1, figures.construct_fx_obs_cds(fx_values, obs_values)),
    (fxobs2, figures.construct_fx_obs_cds(fx_values*.5, obs_values))
]

prereport_html = template.prereport_to_html(prereport)

body = template.add_figures_to_prereport(
    fx_obs_cds, report, metadata, prereport_html)

full_report = template.full_html(body)

with open('bokeh_report.html', 'w') as f:
    f.write(full_report)
