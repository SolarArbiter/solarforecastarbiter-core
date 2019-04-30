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
    Simple example metadata for a site
    """
    return _site_metadata()


def _site_metadata():
    metadata = datamodel.Site(
        name='Albuquerque Baseline', latitude=35.05, longitude=-106.54,
        elevation=1657.0, timezone='America/Denver', network='Sandia')
    return metadata


@pytest.fixture(scope='module')
def powerplant_metadata():
    """
    Simple example metadata for a fixed-tilt PV site
    """
    return _powerplant_metadata()


def _powerplant_metadata():
    modeling_params = datamodel.FixedTiltModelingParameters(
        ac_capacity=.003, dc_capacity=.0035, temperature_coefficient=-0.003,
        dc_loss_factor=3, ac_loss_factor=0,
        surface_tilt=30, surface_azimuth=180)
    metadata = datamodel.SolarPowerPlant(
        name='Albuquerque Baseline', latitude=35.05, longitude=-106.54,
        elevation=1657.0, timezone='America/Denver', network='Sandia',
        modeling_parameters=modeling_params)
    return metadata


@pytest.fixture(scope='module',
                params=['site', 'powerplant'])
def site_powerplant_site_type(request):
    site_type = request.param
    modparams = globals()['_' + request.param + '_metadata']()
    return modparams, site_type


@pytest.fixture(scope='module')
def ac_power_observation_metadata(powerplant_metadata):
    ac_power_meta = datamodel.Observation(
        name='Albuquerque Baseline AC Power', variable='ac_power',
        value_type='instantaneous', interval_length=pd.Timedelta('5min'),
        interval_label='instant', site=powerplant_metadata, uncertainty=1)
    return ac_power_meta


@pytest.fixture(scope='module', params=['instant', 'beginning', 'ending'])
def ac_power_observation_metadata_label(request, powerplant_metadata):
    ac_power_meta = datamodel.Observation(
        name='Albuquerque Baseline AC Power', variable='ac_power',
        value_type='mean', interval_length=pd.Timedelta('5min'),
        interval_label=request.param, site=powerplant_metadata, uncertainty=1)
    return ac_power_meta


@pytest.fixture(scope='module')
def ghi_observation_metadata(site_metadata):
    ghi_meta = datamodel.Observation(
        name='Albuquerque Baseline GHI', variable='ghi',
        value_type='instantaneous', interval_length=pd.Timedelta('5min'),
        interval_label='instant', site=site_metadata, uncertainty=1)
    return ghi_meta


def default_observation(
        site_metadata,
        name='Albuquerque Baseline', variable='ghi',
        value_type='mean', uncertainty=1, interval_length='1h',
        interval_label='beginning'):
    """
    Helpful when you want to test a couple of parameters and don't
    need to worry about the rest.
    """
    obs = datamodel.Observation(
        name=name, variable=variable,
        value_type=value_type, site=site_metadata, uncertainty=uncertainty,
        interval_length=pd.Timedelta(interval_length),
        interval_label=interval_label
    )
    return obs


def default_forecast(
        site_metadata,
        name='Albuquerque Baseline', variable='ghi',
        value_type='mean', issue_time_of_day=dt.time(hour=5),
        lead_time_to_start='1h', interval_length='1h', run_length='12h',
        interval_label='beginning'):
    """
    Helpful when you want to test a couple of parameters and don't
    need to worry about the rest.
    """
    fx = datamodel.Forecast(
        site=site_metadata,
        name=name,
        value_type=value_type,
        variable=variable,
        issue_time_of_day=issue_time_of_day,
        lead_time_to_start=pd.Timedelta(lead_time_to_start),
        interval_length=pd.Timedelta(interval_length),
        run_length=pd.Timedelta(run_length),
        interval_label=interval_label
    )
    return fx


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


@pytest.fixture(scope='module', params=['instant', 'beginning', 'ending'])
def ac_power_forecast_metadata_label(request, site_metadata):
    ac_power_fx_meta = datamodel.Forecast(
        name='Albuquerque Baseline AC Power forecast 1',
        issue_time_of_day=dt.time(hour=0),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label=request.param,
        value_type='mean',
        variable='ac_power',
        site=site_metadata
    )
    return ac_power_fx_meta


def fixed_modeling_parameters():
    modeling_params = datamodel.FixedTiltModelingParameters(
        ac_capacity=.003, dc_capacity=.0035, temperature_coefficient=-0.003,
        dc_loss_factor=3, ac_loss_factor=0,
        surface_tilt=30, surface_azimuth=180)
    return modeling_params


def tracking_modeling_parameters():
    modeling_params = datamodel.SingleAxisModelingParameters(
        ac_capacity=.003, dc_capacity=.0035, temperature_coefficient=-0.003,
        dc_loss_factor=3, ac_loss_factor=0,
        axis_tilt=0, axis_azimuth=0, ground_coverage_ratio=2/7,
        backtrack=True, maximum_rotation_angle=45)
    return modeling_params


@pytest.fixture(scope='module',
                params=['fixed', 'tracking'])
def modeling_parameters_system_type(request):
    system_type = request.param
    modparams = globals()[request.param + '_modeling_parameters']()
    return modparams, system_type
