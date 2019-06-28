"""Defines the default Forecasts for reference sites"""
import datetime as dt
import json


import pandas as pd


from solarforecastarbiter.datamodel import Forecast, Site


_DUMMY_SITE = Site('dummy', 0, 0, 0, 'UTC')

# TODO: add poa globa, maybe relative humidity
CURRENT_NWP_VARIABLES = {'ac_power', 'ghi', 'dni', 'dhi', 'air_temperature',
                         'wind_speed'}


TEMPLATE_FORECASTS = [
    Forecast(
        name='Day Ahead GFS',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('1d'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_forecast': True,
             'model': 'gfs_quarter_deg_hourly_to_hourly_mean'
             })
        ),
    Forecast(
        name='Subhourly HRRR',
        issue_time_of_day=dt.time(5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('15min'),
        run_length=pd.Timedelta('6h'),
        interval_label='instant',
        interval_value_type='instantaneous',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_forecast': True,
             'model': 'hrrr_subhourly_to_subhourly_instantaneous'
             })
        ),
    Forecast(
        name='Current Day NAM',
        issue_time_of_day=dt.time(23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_forecast': True,
             'model': 'nam_12km_cloud_cover_to_hourly_mean'
             })
        ),
]
