"""
The current set of fixtures are primarily meant as examples of
what metadata, observations, and forecasts might look like
in terms of dataclass and pandas objects.
"""
import datetime as dt


import pandas as pd
import pytest


from solarforecastarbiter import datamodel


@pytest.fixture(scope='module', params=[
    pd.DataFrame.from_records(
        [(0, 0),
         (1.0, 0),
         (1.5, 0),
         (9.9, 1),
         (2.0, 0),
         (-999, 3)],
        index=pd.date_range(start='20190101T1200',
                            end='20190101T1225',
                            freq='5min',
                            tz='America/Denver'),
        columns=['value', 'quality'])
])
def short_test_observations(request):
    """
    Just a simple example of a DataFrame of observations.
    Also an example of parametrizing fixtures
    """
    return request.param


@pytest.fixture(scope='module')
def short_test_forecast():
    return pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                     index=pd.date_range(start='20190101T0600',
                                         end='20190101T1700',
                                         freq='1h',
                                         tz='America/Denver'))


@pytest.fixture(scope='module')
def site_metadata():
    """
    Simple example metadata for a fixed-tilt PV site
    """
    modeling_params = datamodel.FixedTiltModelingParameters(
        ac_capacity=.003, dc_capacity=.0035, temperature_coefficient=-0.003,
        surface_tilt=30, surface_azimuth=180)
    metadata = datamodel.SolarPowerPlant(
        name='Albuquerque Baseline', latitude=35.05, longitude=-106.54,
        elevation=1657.0, timezone='America/Denver', network='Sandia',
        modeling_parameters=modeling_params)
    return metadata


@pytest.fixture(scope='module')
def ac_power_observation_metadata(site_metadata):
    ac_power_meta = datamodel.Observation(
        name='Albuquerque Baseline AC Power', variable='ac_power',
        value_type='instantaneous', interval_label='instant',
        site=site_metadata, uncertainty=1)
    return ac_power_meta


@pytest.fixture(scope='module')
def ac_power_forecast_metadata(site_metadata):
    ac_power_fx_meta = datamodel.Forecast(
        name='Albuquerque Baseline AC Power forecast 1',
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('12h'),
        interval_label='beginning',
        value_type='mean',
        variable='ac_power',
        site=site_metadata
    )
    return ac_power_fx_meta
