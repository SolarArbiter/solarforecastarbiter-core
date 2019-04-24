"""
The current set of fixtures are primarily meant as examples of
what metadata, observations, and forecasts might look like
in terms of dataclass and pandas objects.
"""
import datetime as dt
import json


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


@pytest.fixture()
def site_text():
    return b"""
{
    "elevation": 786.0,
    "extra_parameters": "",
    "latitude": 43.73403,
    "longitude": -96.62328,
    "modeling_parameters": {
        "ac_capacity": 0.015,
        "ac_loss_factor": 0.0,
        "axis_azimuth": null,
        "axis_tilt": null,
        "backtrack": null,
        "dc_capacity": 0.015,
        "dc_loss_factor": 0.0,
        "ground_coverage_ratio": null,
        "max_rotation_angle": null,
        "surface_azimuth": 180.0,
        "surface_tilt": 45.0,
        "temperature_coefficient": -0.002,
        "tracking_type": "fixed"
    },
    "name": "Fixed plant",
    "provider": "Organization 1",
    "timezone": "Etc/GMT+6",
    "site_id": "123e4567-e89b-12d3-a456-426655440002",
    "created_at": "2019-03-01T11:44:44Z",
    "modified_at": "2019-03-01T11:44:44Z"
}"""


@pytest.fixture()
def many_sites_text():
    return b"""[
    {
        "elevation": 786.0,
        "extra_parameters": "{\\"network\\": \\"NREL MIDC\\"}",
        "latitude": 32.22969,
        "longitude": -110.95534,
        "modeling_parameters": {
            "ac_capacity": null,
            "ac_loss_factor": null,
            "axis_azimuth": null,
            "axis_tilt": null,
            "backtrack": null,
            "dc_capacity": null,
            "dc_loss_factor": null,
            "ground_coverage_ratio": null,
            "max_rotation_angle": null,
            "surface_azimuth": null,
            "surface_tilt": null,
            "temperature_coefficient": null,
            "tracking_type": null
        },
        "name": "Weather Station 1",
        "provider": "Organization 1",
        "timezone": "America/Phoenix",
        "site_id": "d2018f1d-82b1-422a-8ec4-4e8b3fe92a4a",
        "created_at": "2019-03-01T11:44:44Z",
        "modified_at": "2019-03-01T11:44:44Z"
    },
    {
        "elevation": 786.0,
        "extra_parameters": "",
        "latitude": 43.73403,
        "longitude": -96.62328,
        "modeling_parameters": {
            "ac_capacity": 0.015,
            "ac_loss_factor": 0.0,
            "axis_azimuth": null,
            "axis_tilt": null,
            "backtrack": null,
            "dc_capacity": 0.015,
            "dc_loss_factor": 0.0,
            "ground_coverage_ratio": null,
            "max_rotation_angle": null,
            "surface_azimuth": 180.0,
            "surface_tilt": 45.0,
            "temperature_coefficient": -0.002,
            "tracking_type": "fixed"
        },
        "name": "Fixed plant",
        "provider": "Organization 1",
        "timezone": "Etc/GMT+6",
        "site_id": "123e4567-e89b-12d3-a456-426655440002",
        "created_at": "2019-03-01T11:44:44Z",
        "modified_at": "2019-03-01T11:44:44Z"
    },
    {
        "elevation": 786.0,
        "extra_parameters": "",
        "latitude": 43.73403,
        "longitude": -96.62328,
        "modeling_parameters": {
            "ac_capacity": 0.015,
            "ac_loss_factor": 0.0,
            "axis_azimuth": 180,
            "axis_tilt": 0,
            "backtrack": true,
            "dc_capacity": 0.015,
            "dc_loss_factor": 0.0,
            "ground_coverage_ratio": 0.233,
            "max_rotation_angle": 90.0,
            "surface_azimuth": null,
            "surface_tilt": null,
            "temperature_coefficient": -0.002,
            "tracking_type": "single_axis"
        },
        "name": "Tracking plant",
        "provider": "Organization 1",
        "timezone": "Etc/GMT+6",
        "site_id": "123e4567-e89b-12d3-a456-426655440002",
        "created_at": "2019-03-01T11:44:46Z",
        "modified_at": "2019-03-01T11:44:46Z"
    }
]"""



def _site_from_dict(site_dict):
    if 'modeling_parameters' in site_dict:
        return datamodel.SolarPowerPlant(
            name=site_dict['name'], latitude=site_dict['latitude'],
            longitude=site_dict['longitude'], elevation=site_dict['elevation'],
            timezone=site_dict['timezone'],
            provider=site_dict.get('provider', ''),
            well_known_text=site_dict.get('well_known_text', ''),
            extra_parameters=site_dict.get('extra_parameters', ''),
            modeling_parameters=site_dict['modeling_parameters'])
    else:
        return datamodel.Site(
            name=site_dict['name'], latitude=site_dict['latitude'],
            longitude=site_dict['longitude'], elevation=site_dict['elevation'],
            timezone=site_dict['timezone'],
            provider=site_dict.get('provider', ''),
            well_known_text=site_dict.get('well_known_text', ''),
            extra_parameters=site_dict.get('extra_parameters', ''))



@pytest.fixture()
def single_site(site_text):
    sited = json.loads(site_text)
    fixedmp = sited['modeling_parameters']
    fixed = datamodel.FixedTiltModelingParameters(
        ac_capacity=fixedmp['ac_capacity'],
        dc_capacity=fixedmp['dc_capacity'],
        temperature_coefficient=fixedmp['temperature_coefficient'],
        dc_loss_factor=fixedmp['dc_loss_factor'],
        ac_loss_factor=fixedmp['ac_loss_factor'],
        surface_tilt=fixedmp['surface_tilt'],
        surface_azimuth=fixedmp['surface_azimuth'])
    sited['modeling_parameters'] = fixed
    return _site_from_dict(sited)


@pytest.fixture()
def many_sites(many_sites_text):
    sites = json.loads(many_sites_text)
    del sites[0]['modeling_parameters']
    out = [_site_from_dict(sites[0])]
    fixedmp = sites[1]['modeling_parameters']
    fixed = datamodel.FixedTiltModelingParameters(
        ac_capacity=fixedmp['ac_capacity'],
        dc_capacity=fixedmp['dc_capacity'],
        temperature_coefficient=fixedmp['temperature_coefficient'],
        dc_loss_factor=fixedmp['dc_loss_factor'],
        ac_loss_factor=fixedmp['ac_loss_factor'],
        surface_tilt=fixedmp['surface_tilt'],
        surface_azimuth=fixedmp['surface_azimuth'])
    sites[1]['modeling_parameters'] = fixed
    out.append(_site_from_dict(sites[1]))
    singlemp = sites[2]['modeling_parameters']
    single = datamodel.SingleAxisModelingParameters(
        ac_capacity=singlemp['ac_capacity'],
        dc_capacity=singlemp['dc_capacity'],
        temperature_coefficient=singlemp['temperature_coefficient'],
        dc_loss_factor=singlemp['dc_loss_factor'],
        ac_loss_factor=singlemp['ac_loss_factor'],
        axis_tilt=singlemp['axis_tilt'],
        axis_azimuth=singlemp['axis_azimuth'],
        ground_coverage_ratio=singlemp['ground_coverage_ratio'],
        backtrack=singlemp['backtrack'],
        max_rotation_angle=singlemp['max_rotation_angle']
    )
    sites[2]['modeling_parameters'] = single
    out.append(_site_from_dict(sites[2]))
    return out


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
def ac_power_observation_metadata(site_metadata):
    ac_power_meta = datamodel.Observation(
        name='Albuquerque Baseline AC Power', variable='ac_power',
        value_type='instantaneous', interval_length=pd.Timedelta('5min'),
        interval_label='instant', site=site_metadata, uncertainty=1)
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


@pytest.fixture()
def single_observation_text():
    return b"""
{
  "_links": {
    "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440001"
  },
  "created_at": "2019-03-01T12:01:48+00:00",
  "extra_parameters": "{\\"instrument\\": \\"Ascension Technology Rotating Shadowband Pyranometer\\", \\"network\\": \\"UO SRML\\"}",
  "interval_label": "beginning",
  "interval_length": 5,
  "interval_value_type": "interval_mean",
  "modified_at": "2019-03-01T12:01:48+00:00",
  "name": "DNI Instrument 2",
  "observation_id": "9ce9715c-bd91-47b7-989f-50bb558f1eb9",
  "provider": "Organization 1",
  "site_id": "123e4567-e89b-12d3-a456-426655440001",
  "uncertainty": 0.1,
  "variable": "dni"
}
"""  # NOQA


@pytest.fixture()
def many_observations_text():
    return b"""[
  {
    "_links": {
      "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440001"
    },
    "created_at": "2019-03-01T12:01:48+00:00",
    "extra_parameters": "{\\"instrument\\": \\"Ascension Technology Rotating Shadowband Pyranometer\\", \\"network\\": \\"UO SRML\\"}",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "modified_at": "2019-03-01T12:01:48+00:00",
    "name": "DNI Instrument 2",
    "observation_id": "9ce9715c-bd91-47b7-989f-50bb558f1eb9",
    "provider": "Organization 1",
    "site_id": "123e4567-e89b-12d3-a456-426655440001",
    "uncertainty": 0.1,
    "variable": "dni"
  },
  {
    "_links": {
      "site": "http://127.0.0.1:5000/sites/d2018f1d-82b1-422a-8ec4-4e8b3fe92a4a"
    },
    "created_at": "2019-03-01T12:01:55+00:00",
    "extra_parameters": "{\\"instrument\\": \\"Kipp & Zonen CMP 22 Pyranometer\\", \\"network\\": \\"UO SRML\\"}",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "modified_at": "2019-03-01T12:01:55+00:00",
    "name": "GHI Instrument 2",
    "observation_id": "e0da0dea-9482-4073-84de-f1b12c304d23",
    "provider": "Organization 1",
    "site_id": "d2018f1d-82b1-422a-8ec4-4e8b3fe92a4a",
    "uncertainty": 0.1,
    "variable": "ghi"
  },
  {
    "_links": {
      "site": "http://127.0.0.1:5000/sites/d2018f1d-82b1-422a-8ec4-4e8b3fe92a4a"
    },
    "created_at": "2019-03-01T12:02:38+00:00",
    "extra_parameters": "{\\"instrument\\": \\"Kipp & Zonen CMP 22 Pyranometer\\", \\"network\\": \\"NOAA\\"}",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "modified_at": "2019-03-01T12:02:38+00:00",
    "name": "Sioux Falls, ghi",
    "observation_id": "b1dfe2cb-9c8e-43cd-afcf-c5a6feaf81e2",
    "provider": "Organization 1",
    "site_id": "d2018f1d-82b1-422a-8ec4-4e8b3fe92a4a",
    "uncertainty": 0.1,
    "variable": "ghi"
  }
]"""  # NOQA


@pytest.fixture()
def _observation_from_dict(single_site):
    def f(obs_dict):
        return datamodel.Observation(
            name=obs_dict['name'], variable=obs_dict['variable'],
            interval_value_type=obs_dict['interval_value_type'],
            interval_length=obs_dict['interval_length'],
            interval_label=obs_dict['interval_label'],
            site=single_site, uncertainty=obs_dict['uncertainty'],
            description=obs_dict.get('description', ''),
            extra_parameters=obs_dict.get('extra_parameters', ''))
    return f


@pytest.fixture()
def single_observation(single_observation_text, _observation_from_dict):
    return _observation_from_dict(json.loads(single_observation_text))


@pytest.fixture()
def many_observations(many_observations_text, _observation_from_dict):
    return [_observation_from_dict(obs) for obs
            in json.loads(many_observations_text)]
