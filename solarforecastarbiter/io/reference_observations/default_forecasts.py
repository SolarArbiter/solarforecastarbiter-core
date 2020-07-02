"""Defines the default Forecasts for reference sites"""
import datetime as dt
import json


import pandas as pd


from solarforecastarbiter.datamodel import (
    Forecast, Site, ProbabilisticForecast)
from solarforecastarbiter.io.fetch.nwp import DOMAIN


def is_in_nwp_domain(site):
    """
    Checks the location of the site and returns True if it is within the
    domain of the NWP forecasts.
    """
    return ((DOMAIN['leftlon'] <= site.longitude <= DOMAIN['rightlon']) and
            (DOMAIN['bottomlat'] <= site.latitude <= DOMAIN['toplat']))


_DUMMY_SITE = Site('dummy', 0, 0, 0, 'UTC')

# TODO: add poa globa, maybe relative humidity
CURRENT_NWP_VARIABLES = {'ac_power', 'ghi', 'dni', 'dhi', 'air_temperature',
                         'wind_speed'}

# issue time of day is in local standard time and will be
# adjusted to the appropriate UTC hour
TEMPLATE_DETERMINISTIC_NWP_FORECASTS = [
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
        name='Intraday HRRR',
        issue_time_of_day=dt.time(5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('6h'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_forecast': True,
             'model': 'hrrr_subhourly_to_hourly_mean'
             })
        ),
    Forecast(
        name='Intraday RAP',
        issue_time_of_day=dt.time(5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('6h'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_forecast': True,
             'model': 'rap_cloud_cover_to_hourly_mean'
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

TEMPLATE_PROBABILISTIC_NWP_FORECASTS = [
    ProbabilisticForecast(
        name='Day Ahead GEFS',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('1d'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        axis='y',
        constant_values=range(0, 101, 5),
        extra_parameters=json.dumps(
            {'is_reference_forecast': True,
             'model': 'gefs_half_deg_to_hourly_mean'
             })
    )
]


TEMPLATE_PROBABILISTIC_PERSISTENCE_FORECASTS = [
    ProbabilisticForecast(
        name='Hour Ahead Prob Persistence',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        axis='y',
        constant_values=range(0, 101, 5),
        extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True})
    )
]


TEMPLATE_DETERMINISTIC_PERSISTENCE_FORECASTS = [
    Forecast(
        name='Day Ahead Persistence',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('1d'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True})
        ),
    Forecast(
        name='Hour Ahead Persistence',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='beginning',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True})
        ),
    Forecast(
        name='Fifteen-minute Ahead Persistence',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('15min'),
        interval_length=pd.Timedelta('15min'),
        run_length=pd.Timedelta('15min'),
        interval_label='beginning',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True})
        ),
    Forecast(
        name='Five-minute Ahead Persistence',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('5min'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('5min'),
        interval_label='beginning',
        interval_value_type='interval_mean',
        variable='ghi',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True})
         )
]

TEMPLATE_NWP_FORECASTS = (
    TEMPLATE_DETERMINISTIC_NWP_FORECASTS +
    TEMPLATE_PROBABILISTIC_NWP_FORECASTS
)

TEMPLATE_FORECASTS = (
    TEMPLATE_NWP_FORECASTS +
    TEMPLATE_DETERMINISTIC_PERSISTENCE_FORECASTS +
    TEMPLATE_PROBABILISTIC_PERSISTENCE_FORECASTS
)

TEMPLATE_NETLOAD_PERSISTENCE_FORECASTS = [
    Forecast(
        name='Day of Week Persistence',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('1d'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='net_load',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True})
        ),
    Forecast(
        name='Hour Ahead Persistence',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='net_load',
        site=_DUMMY_SITE,
        extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True})
        )
]
