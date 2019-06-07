import sys
from solarforecastarbiter import datamodel
from solarforecastarbiter.io import api
import pandas as pd
import datetime


token = api.request_cli_access_token('reference@solarforecastarbiter.org', '9YwuVY@52D9s1Xs%%3JA')
session = api.APISession(token)
MIDC_SITE = '9f61b880-7e49-11e9-9624-0a580a8003e9'
site = session.get_site(MIDC_SITE)


extra_params = '{"model": "gfs_quarter_deg_to_hourly_mean","fetch_metadata": "GFS_0P25_1HR"}'
forecast1 = datamodel.Forecast(
    name='0 Day GFS GHI',
    issue_time_of_day=datetime.time(7),
    lead_time_to_start=pd.Timedelta('0min'),
    interval_length=pd.Timedelta('1h'),
    run_length=pd.Timedelta('24hr'),
    interval_label='beginning',
    interval_value_type='interval_mean',
    variable='ghi',
    site=site,
    extra_parameters=extra_params
)


forecast2 = datamodel.Forecast(
    name='0 Day GFS Temp',
    issue_time_of_day=datetime.time(7),
    lead_time_to_start=pd.Timedelta('0min'),
    interval_length=pd.Timedelta('1h'),
    run_length=pd.Timedelta('24hr'),
    interval_label='beginning',
    interval_value_type='interval_mean',
    variable='air_temperature',
    site=site,
    extra_parameters=extra_params
)


forecast3 = datamodel.Forecast(
    name='0 Day GFS DNI',
    issue_time_of_day=datetime.time(7),
    lead_time_to_start=pd.Timedelta('0min'),
    interval_length=pd.Timedelta('1h'),
    run_length=pd.Timedelta('24hr'),
    interval_label='beginning',
    interval_value_type='interval_mean',
    variable='dni',
    site=site,
    extra_parameters=extra_params
)

forecast4 = datamodel.Forecast(
    name='Day Ahead GFS GHI',
    issue_time_of_day=datetime.time(7),
    lead_time_to_start=pd.Timedelta('24hr'),
    interval_length=pd.Timedelta('1h'),
    run_length=pd.Timedelta('24hr'),
    interval_label='beginning',
    interval_value_type='interval_mean',
    variable='ghi',
    site=site,
    extra_parameters=extra_params
)



for forecast in [forecast4]:
    fx = session.create_forecast(forecast)
    print(fx.forecast_id)
