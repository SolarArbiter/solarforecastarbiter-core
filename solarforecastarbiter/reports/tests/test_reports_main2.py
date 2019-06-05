from collections import defaultdict
import datetime
from functools import partial
import logging

import pandas as pd
import numpy as np

from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.io.api import APISession, request_cli_access_token
from solarforecastarbiter.reports import template, figures, main


start = pd.Timestamp('20190401 0000Z')
end = pd.Timestamp('20190531 2359Z')

# don't store your real passwords or tokens in plain text like this!
# only for demonstration purposes!
token = request_cli_access_token('testing@solarforecastarbiter.org',
                                 'Thepassword123!')

session = APISession(token)

# NREL MIDC University of Arizona OASIS
site = session.get_site('9f61b880-7e49-11e9-9624-0a580a8003e9')
# GHI
observation = session.get_observation('9f657636-7e49-11e9-b77f-0a580a8003e9')
logging.info('getting observation values')
observation_values = session.get_observation_values(observation.observation_id,
                                                    start, end)
# current day (0) and day ahead (1) GHI forecasts derived from GFS
forecast_0 = session.get_forecast('da2bc386-8712-11e9-a1c7-0a580a8200ae')
forecast_1 = session.get_forecast('68a1c22c-87b5-11e9-bf88-0a580a8200ae')
logging.info('getting forecast_0 values')
forecast_values_0 = session.get_forecast_values(forecast_0.forecast_id,
                                                start, end)
logging.info('getting forecast_1 values')
forecast_values_1 = session.get_forecast_values(forecast_1.forecast_id,
                                                start, end)

# forecast intervals are 1h and labels are beginning
# make new observation that matches forecasts and resample data
obs_dict = observation.to_dict()
obs_dict['interval_label'] = 'beginning'
obs_dict['interval_length'] = 60
observation_resampled = datamodel.Observation.from_dict(obs_dict)
observation_values_1h = observation_values.resample('1h', label='left').mean()

fxobs1 = datamodel.ForecastObservation(forecast_0, observation_resampled)
fxobs2 = datamodel.ForecastObservation(forecast_1, observation_resampled)

report = datamodel.Report(
    name='NREL MIDC OASIS GHI Forecast Analysis',
    start=start,
    end=end,
    forecast_observations=(fxobs1, fxobs2),
    metrics=('mae', 'rmse', 'mbe')
)


metadata = main.create_metadata(report)


def rmse(diff):
    return np.sqrt((diff * diff).sum() / (len(diff) - 1))


df = pd.DataFrame({'obs': observation_values_1h['value'],
                   'fx0': forecast_values_0, 'fx1': forecast_values_1})
df = df.dropna()
# obs_values = observation_values_1h['value']
# fx_values = forecast_values_0
# fx_values2 = forecast_values_1
obs_values = df['obs']
fx_values = df['fx0']
fx_values2 = df['fx1']

metrics_a = defaultdict(dict)
metrics_b = defaultdict(dict)
metrics_a['name'] = forecast_0.name
metrics_b['name'] = forecast_1.name
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
    (fxobs2, figures.construct_fx_obs_cds(fx_values2, obs_values))
]

prereport_html = template.prereport_to_html(prereport)

body = template.add_figures_to_prereport(
    fx_obs_cds, report, metadata, prereport_html)

full_report = template.full_html(body)

with open('bokeh_report.html', 'w') as f:
    f.write(full_report)
