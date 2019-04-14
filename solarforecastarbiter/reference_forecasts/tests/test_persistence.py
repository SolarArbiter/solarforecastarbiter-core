
import pandas as pd
# from pandas.util.testing import assert_series_equal

from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.reference_forecasts import persistence

import pytest

def test():
    from solarforecastarbiter import conftest
    site = conftest._site_metadata()
    data_index = pd.DatetimeIndex(
        start='20190405', end='20190406', freq='5min', tz='America/Phoenix')
    solar_position = pvmodel.calculate_solar_position(
        site.latitude, site.longitude, site.elevation, data_index)
    data_cs = pvmodel.calculate_clearsky(
        site.latitude, site.longitude, site.elevation,
        solar_position['apparent_zenith'])
    data = data_cs['ghi'] * 0.8

    def load_data(observation, data_start, data_end):
        return data[data_start:data_end]

    def ghi_observation_metadata(site_metadata):
        ghi_meta = datamodel.Observation(
            name='Albuquerque Baseline GHI', variable='ghi',
            value_type='instantaneous', interval_length=pd.Timedelta('5min'),
            interval_label='instant', site=site_metadata, uncertainty=1)
        return ghi_meta

    observation = ghi_observation_metadata(site)
    window = pd.Timedelta('5min')
    tz = 'America/Phoenix'
    data_start = pd.Timestamp('20190405 1200', tz=tz)
    data_end = pd.Timestamp('20190405 1300', tz=tz)
    forecast_start = pd.Timestamp('20190405 1300', tz=tz)
    forecast_end = pd.Timestamp('20190405 1400', tz=tz)
    interval_length = pd.Timedelta('5min')
    persistence.persistence(
        observation, window, data_start, data_end, forecast_start,
        forecast_end, interval_length, load_data=load_data)
    persistence.index_persistence(
        observation, window, data_start, data_end, forecast_start,
        forecast_end, interval_length, load_data=load_data)
