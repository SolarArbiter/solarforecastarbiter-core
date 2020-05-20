"""
The current set of fixtures are primarily meant as examples of
what metadata, observations, and forecasts might look like
in terms of dataclass and pandas objects.
"""
import base64
import itertools
import datetime as dt
import json


import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pkg_resources import resource_filename, Requirement
import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import preprocessing


@pytest.fixture()
def observation_values_text():
    return b"""
{
  "_links": {
    "metadata": ""
  },
  "observation_id": "OBSID",
  "values": [
    {
      "quality_flag": 0,
      "timestamp": "2019-01-01T12:00:00-0700",
      "value": 0
    },
    {
      "quality_flag": 0,
      "timestamp": "2019-01-01T12:05:00-0700",
      "value": 1.0
    },
    {
      "quality_flag": 0,
      "timestamp": "2019-01-01T12:10:00-0700",
      "value": 1.5
    },
    {
      "quality_flag": 1,
      "timestamp": "2019-01-01T12:15:00-0700",
      "value": 9.9
    },
    {
      "quality_flag": 0,
      "timestamp": "2019-01-01T12:20:00-0700",
      "value": 2.0
    },
    {
      "quality_flag": 3,
      "timestamp": "2019-01-01T12:25:00-0700",
      "value": -999
    }
  ]
}
"""


@pytest.fixture()
def observation_values():
    return pd.DataFrame.from_records(
        [(0, 0),
         (1.0, 0),
         (1.5, 0),
         (9.9, 1),
         (2.0, 0),
         (-999, 3)],
        index=pd.date_range(start='20190101T1200',
                            end='20190101T1225',
                            freq='5min',
                            tz='America/Denver',
                            name='timestamp'),
        columns=['value', 'quality_flag']).tz_convert('UTC')


@pytest.fixture()
def validated_observation_values():
    return pd.DataFrame.from_records(
        [(0, 2),
         (1.0, 3),
         (1.5, 2),
         (9.9, 2),
         (2.0, 34),
         (-999, 1 << 8 | 2)],
        index=pd.date_range(start='20190101T1200',
                            end='20190101T1225',
                            freq='5min',
                            tz='America/Denver',
                            name='timestamp'),
        columns=['value', 'quality_flag']).tz_convert('UTC')


@pytest.fixture()
def forecast_values_text():
    return b"""
{
  "_links": {
    "metadata": ""
  },
  "forecast_id": "FXID",
  "values": [
    {
      "timestamp": "2019-01-01T06:00:00-0700",
      "value": 0.0
    },
    {
      "timestamp": "2019-01-01T07:00:00-0700",
      "value": 1.0
    },
    {
      "timestamp": "2019-01-01T08:00:00-0700",
      "value": 2.0
    },
    {
      "timestamp": "2019-01-01T09:00:00-0700",
      "value": 3.0
    },
    {
      "timestamp": "2019-01-01T10:00:00-0700",
      "value": 4.0
    },
    {
      "timestamp": "2019-01-01T11:00:00-0700",
      "value": 5.0
    }
  ]
}
"""


@pytest.fixture()
def forecast_values():
    return pd.Series([0.0, 1, 2, 3, 4, 5],
                     name='value',
                     index=pd.date_range(start='20190101T0600',
                                         end='20190101T1100',
                                         freq='1h',
                                         tz='America/Denver',
                                         name='timestamp')).tz_convert('UTC')


@pytest.fixture()
def prob_forecast_values_text_list():
    return [b"""
{
  "_links": {
    "metadata": ""
  },
  "forecast_id": "CV25",
  "values": [
    {
      "timestamp": "2019-01-01T06:00:00-0700",
      "value": 0.0
    },
    {
      "timestamp": "2019-01-01T07:00:00-0700",
      "value": 1.0
    },
    {
      "timestamp": "2019-01-01T08:00:00-0700",
      "value": 2.0
    },
    {
      "timestamp": "2019-01-01T09:00:00-0700",
      "value": 3.0
    },
    {
      "timestamp": "2019-01-01T10:00:00-0700",
      "value": 4.0
    },
    {
      "timestamp": "2019-01-01T11:00:00-0700",
      "value": 5.0
    }
  ]
}
""",  # NOQA
b"""
{
  "_links": {
    "metadata": ""
},
  "forecast_id": "CV50",
  "values": [
    {
    "timestamp": "2019-01-01T06:00:00-0700",
    "value": 1.0
    },
    {
    "timestamp": "2019-01-01T07:00:00-0700",
    "value": 2.0
    },
    {
    "timestamp": "2019-01-01T08:00:00-0700",
    "value": 3.0
    },
    {
    "timestamp": "2019-01-01T09:00:00-0700",
    "value": 4.0
    },
    {
    "timestamp": "2019-01-01T10:00:00-0700",
    "value": 5.0
    },
    {
    "timestamp": "2019-01-01T11:00:00-0700",
    "value": 6.0
    }
  ]
}
""",  # NOQA
b"""
{
  "_links": {
    "metadata": ""
},
  "forecast_id": "CV75",
  "values": [
    {
      "timestamp": "2019-01-01T06:00:00-0700",
      "value": 2.0
    },
    {
      "timestamp": "2019-01-01T07:00:00-0700",
      "value": 3.0
    },
    {
      "timestamp": "2019-01-01T08:00:00-0700",
      "value": 4.0
    },
    {
      "timestamp": "2019-01-01T09:00:00-0700",
      "value": 5.0
    },
    {
      "timestamp": "2019-01-01T10:00:00-0700",
      "value": 6.0
    },
    {
      "timestamp": "2019-01-01T11:00:00-0700",
      "value": 7.0
    }
  ]
}
"""
]


@pytest.fixture()
def prob_forecast_values():
    return pd.DataFrame(
        {'25': [0.0, 1, 2, 3, 4, 5],
         '50': [1.0, 2, 3, 4, 5, 6],
         '75': [2.0, 3, 4, 5, 6, 7]},
        index=pd.date_range(start='20190101T0600',
                            end='20190101T1100',
                            freq='1h',
                            tz='America/Denver',
                            name='timestamp')).tz_convert('UTC')


@pytest.fixture(scope='module')
def site_metadata():
    """
    Simple example metadata for a site
    """
    return _site_metadata()


def _site_metadata():
    metadata = datamodel.Site(
        name='Albuquerque Baseline', latitude=35.05, longitude=-106.54,
        elevation=1657.0, timezone='America/Denver', provider='Sandia')
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
        "temperature_coefficient": -0.2,
        "tracking_type": "fixed"
    },
    "name": "Power Plant 1",
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
            "temperature_coefficient": -0.2,
            "tracking_type": "fixed"
        },
        "name": "Power Plant 1",
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
            "temperature_coefficient": -0.2,
            "tracking_type": "single_axis"
        },
        "name": "Tracking plant",
        "provider": "Organization 1",
        "timezone": "Etc/GMT+6",
        "site_id": "123e4567-e89b-12d3-a456-426655440001",
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
            extra_parameters=site_dict.get('extra_parameters', ''),
            site_id=site_dict.get('site_id', ''),
            modeling_parameters=site_dict['modeling_parameters'])
    else:
        return datamodel.Site(
            name=site_dict['name'], latitude=site_dict['latitude'],
            longitude=site_dict['longitude'], elevation=site_dict['elevation'],
            timezone=site_dict['timezone'],
            site_id=site_dict.get('site_id', ''),
            provider=site_dict.get('provider', ''),
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


@pytest.fixture()
def get_site(many_sites):
    site_dict = {site.site_id: site for site in many_sites}

    def get(site_id):
        return site_dict.get(site_id, None)
    return get


@pytest.fixture()
def get_aggregate(aggregate):
    def get(agg_id):
        if agg_id is None:
            return None
        else:
            return aggregate
    return get


@pytest.fixture(scope='module')
def powerplant_metadata():
    """
    Simple example metadata for a fixed-tilt PV site
    """
    return _powerplant_metadata()


def _powerplant_metadata():
    modeling_params = datamodel.FixedTiltModelingParameters(
        ac_capacity=.003, dc_capacity=.0035, temperature_coefficient=-0.3,
        dc_loss_factor=3, ac_loss_factor=0,
        surface_tilt=30, surface_azimuth=180)
    metadata = datamodel.SolarPowerPlant(
        name='Albuquerque Baseline', latitude=35.05, longitude=-106.54,
        elevation=1657.0, timezone='America/Denver', provider='Sandia',
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
        interval_value_type='instantaneous',
        interval_length=pd.Timedelta('5min'),
        interval_label='instant', site=powerplant_metadata, uncertainty=1)
    return ac_power_meta


@pytest.fixture(scope='module')
def dc_power_observation_metadata(powerplant_metadata):
    dc_power_meta = datamodel.Observation(
        name='Albuquerque Baseline DC Power', variable='dc_power',
        interval_value_type='instantaneous',
        interval_length=pd.Timedelta('5min'),
        interval_label='instant', site=powerplant_metadata, uncertainty=1)
    return dc_power_meta


@pytest.fixture(scope='module')
def wind_speed_observation_metadata(powerplant_metadata):
    wind_speed_meta = datamodel.Observation(
        name='Albuquerque Baseline Wind Speed', variable='wind_speed',
        interval_value_type='instantaneous',
        interval_length=pd.Timedelta('5min'),
        interval_label='instant', site=powerplant_metadata, uncertainty=1)
    return wind_speed_meta


@pytest.fixture(scope='module', params=['instant', 'beginning', 'ending'])
def ac_power_observation_metadata_label(request, powerplant_metadata):
    ac_power_meta = datamodel.Observation(
        name='Albuquerque Baseline AC Power', variable='ac_power',
        interval_value_type='mean', interval_length=pd.Timedelta('5min'),
        interval_label=request.param, site=powerplant_metadata, uncertainty=1)
    return ac_power_meta


@pytest.fixture(scope='module')
def ghi_observation_metadata(site_metadata):
    ghi_meta = datamodel.Observation(
        name='Albuquerque Baseline GHI', variable='ghi',
        interval_value_type='instantaneous',
        interval_length=pd.Timedelta('5min'),
        interval_label='instant', site=site_metadata, uncertainty=1)
    return ghi_meta


def default_observation(
        site_metadata,
        name='Albuquerque Baseline', variable='ghi',
        interval_value_type='mean', uncertainty=1, interval_length='1h',
        interval_label='beginning'):
    """
    Helpful when you want to test a couple of parameters and don't
    need to worry about the rest.
    """
    obs = datamodel.Observation(
        site=site_metadata, name=name, variable=variable,
        interval_value_type=interval_value_type, uncertainty=uncertainty,
        interval_length=pd.Timedelta(interval_length),
        interval_label=interval_label
    )
    return obs


def default_forecast(
        site_metadata,
        name='Albuquerque Baseline', variable='ghi',
        interval_value_type='mean', issue_time_of_day=dt.time(hour=5),
        lead_time_to_start='1h', interval_length='1h', run_length='12h',
        interval_label='beginning'):
    """
    Helpful when you want to test a couple of parameters and don't
    need to worry about the rest.
    """
    fx = datamodel.Forecast(
        site=site_metadata,
        name=name,
        interval_value_type=interval_value_type,
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
        interval_value_type='mean',
        variable='ac_power',
        site=site_metadata
    )
    return ac_power_fx_meta


@pytest.fixture(scope='module')
def dc_power_forecast_metadata(site_metadata):
    dc_power_fx_meta = datamodel.Forecast(
        name='Albuquerque Baseline DC Power forecast 1',
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('12h'),
        interval_label='beginning',
        interval_value_type='mean',
        variable='dc_power',
        site=site_metadata
    )
    return dc_power_fx_meta


@pytest.fixture(scope='module')
def wind_speed_forecast_metadata(site_metadata):
    wind_speed_fx_meta = datamodel.Forecast(
        name='Albuquerque Baseline Wind Speed forecast 1',
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('12h'),
        interval_label='beginning',
        interval_value_type='mean',
        variable='wind_speed',
        site=site_metadata
    )
    return wind_speed_fx_meta


@pytest.fixture(scope='module', params=['instant', 'beginning', 'ending'])
def ac_power_forecast_metadata_label(request, site_metadata):
    ac_power_fx_meta = datamodel.Forecast(
        name='Albuquerque Baseline AC Power forecast 1',
        issue_time_of_day=dt.time(hour=0),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label=request.param,
        interval_value_type='mean',
        variable='ac_power',
        site=site_metadata
    )
    return ac_power_fx_meta


def fixed_modeling_parameters():
    modeling_params = datamodel.FixedTiltModelingParameters(
        ac_capacity=.003, dc_capacity=.0035, temperature_coefficient=-0.3,
        dc_loss_factor=3, ac_loss_factor=0,
        surface_tilt=30, surface_azimuth=180)
    return modeling_params


def tracking_modeling_parameters():
    modeling_params = datamodel.SingleAxisModelingParameters(
        ac_capacity=.003, dc_capacity=.0035, temperature_coefficient=-0.3,
        dc_loss_factor=3, ac_loss_factor=0,
        axis_tilt=0, axis_azimuth=0, ground_coverage_ratio=2 / 7,
        backtrack=True, max_rotation_angle=45)
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
    "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440002"
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
  "site_id": "123e4567-e89b-12d3-a456-426655440002",
  "uncertainty": 0.1,
  "variable": "dni"
}
"""  # NOQA


@pytest.fixture()
def many_observations_text():
    return b"""[
  {
    "_links": {
      "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440002"
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
    "site_id": "123e4567-e89b-12d3-a456-426655440002",
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
  },
  {
    "_links": {
      "site": "http://localhost:5000/sites/123e4567-e89b-12d3-a456-426655440001"
    },
    "created_at": "2019-03-01T12:01:39",
    "extra_parameters": "{\\"instrument\\": \\"Ascension Technology Rotating Shadowband Pyranometer\\", \\"network\\": \\"UO SRML\\"}",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "modified_at": "2019-03-01T12:01:39",
    "name": "GHI Instrument 1",
    "observation_id": "123e4567-e89b-12d3-a456-426655440000",
    "provider": "Organization 1",
    "site_id": "123e4567-e89b-12d3-a456-426655440001",
    "uncertainty": 0.1,
    "variable": "ghi"
  }
]"""  # NOQA


@pytest.fixture()
def _observation_from_dict(get_site):
    def f(obs_dict):
        return datamodel.Observation(
            name=obs_dict['name'], variable=obs_dict['variable'],
            interval_value_type=obs_dict['interval_value_type'],
            interval_length=pd.Timedelta(f'{obs_dict["interval_length"]}min'),
            interval_label=obs_dict['interval_label'],
            site=get_site(obs_dict['site_id']),
            uncertainty=obs_dict['uncertainty'],
            observation_id=obs_dict.get('observation_id', ''),
            provider=obs_dict.get('provider', ''),
            extra_parameters=obs_dict.get('extra_parameters', ''))
    return f


@pytest.fixture()
def single_observation(single_observation_text, _observation_from_dict):
    return _observation_from_dict(json.loads(single_observation_text))


@pytest.fixture()
def many_observations(many_observations_text, _observation_from_dict):
    return [_observation_from_dict(obs) for obs
            in json.loads(many_observations_text)]


@pytest.fixture()
def single_observation_text_with_site_text(site_text):
    return (b"""
{
  "_links": {
    "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440002"
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
  "site_id": "123e4567-e89b-12d3-a456-426655440002",
  "uncertainty": 0.1,
  "variable": "dni",
  "site": """  # NOQA
  + site_text + b"""
}
""")


@pytest.fixture()
def single_forecast_text():
    return b"""
{
  "_links": {
    "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440002",
    "aggregate": null
  },
  "created_at": "2019-03-01T11:55:37+00:00",
  "extra_parameters": "",
  "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3",
  "interval_label": "beginning",
  "interval_length": 5,
  "interval_value_type": "interval_mean",
  "issue_time_of_day": "06:00",
  "lead_time_to_start": 60,
  "modified_at": "2019-03-01T11:55:37+00:00",
  "name": "DA GHI",
  "provider": "Organization 1",
  "run_length": 1440,
  "site_id": "123e4567-e89b-12d3-a456-426655440002",
  "aggregate_id": null,
  "variable": "ghi"
}
"""


@pytest.fixture()
def many_forecasts_text():
    return b"""
[
  {
    "_links": {
      "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440001",
      "aggregate": null
    },
    "created_at": "2019-03-01T11:55:37+00:00",
    "extra_parameters": "",
    "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "issue_time_of_day": "06:00",
    "lead_time_to_start": 60,
    "modified_at": "2019-03-01T11:55:37+00:00",
    "name": "DA GHI",
    "provider": "Organization 1",
    "run_length": 1440,
    "site_id": "123e4567-e89b-12d3-a456-426655440001",
    "aggregate_id": null,
    "variable": "ghi"
  },
  {
    "_links": {
      "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440002",
      "aggregate": null
    },
    "created_at": "2019-03-01T11:55:38+00:00",
    "extra_parameters": "",
    "forecast_id": "f8dd49fa-23e2-48a0-862b-ba0af6dec276",
    "interval_label": "beginning",
    "interval_length": 1,
    "interval_value_type": "interval_mean",
    "issue_time_of_day": "12:00",
    "lead_time_to_start": 60,
    "modified_at": "2019-03-01T11:55:38+00:00",
    "name": "HA Power",
    "provider": "Organization 1",
    "run_length": 60,
    "site_id": "123e4567-e89b-12d3-a456-426655440002",
    "aggregate_id": null,
    "variable": "ac_power"
  },
  {
    "_links": {
      "site": null,
      "aggregate": "http://localhost:5000/aggregates/458ffc27-df0b-11e9-b622-62adb5fd6af0"
    },
    "aggregate_id": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
    "created_at": "2019-03-01T11:55:37+00:00",
    "extra_parameters": "",
    "forecast_id": "39220780-76ae-4b11-bef1-7a75bdc784e3",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "issue_time_of_day": "06:00",
    "lead_time_to_start": 60,
    "modified_at": "2019-03-01T11:55:37+00:00",
    "name": "GHI Aggregate FX",
    "provider": "Organization 1",
    "run_length": 1440,
    "site_id": null,
    "variable": "ghi"
  }
]
"""  # NOQA


@pytest.fixture()
def _forecast_from_dict(single_site, get_site, get_aggregate):
    def f(fx_dict):
        return datamodel.Forecast(
            name=fx_dict['name'], variable=fx_dict['variable'],
            interval_value_type=fx_dict['interval_value_type'],
            interval_length=pd.Timedelta(f"{fx_dict['interval_length']}min"),
            interval_label=fx_dict['interval_label'],
            site=get_site(fx_dict.get('site_id')),
            aggregate=get_aggregate(fx_dict.get('aggregate_id')),
            issue_time_of_day=dt.time(int(fx_dict['issue_time_of_day'][:2]),
                                      int(fx_dict['issue_time_of_day'][3:])),
            lead_time_to_start=pd.Timedelta(f"{fx_dict['lead_time_to_start']}min"),  # NOQA
            run_length=pd.Timedelta(f"{fx_dict['run_length']}min"),
            forecast_id=fx_dict.get('forecast_id', ''),
            provider=fx_dict.get('provider', ''),
            extra_parameters=fx_dict.get('extra_parameters', ''))
    return f


@pytest.fixture()
def single_forecast(single_forecast_text, _forecast_from_dict):
    return _forecast_from_dict(json.loads(single_forecast_text))


@pytest.fixture()
def single_reference_forecast(single_forecast, ref_forecast_id):
    return single_forecast.replace(forecast_id=ref_forecast_id)


@pytest.fixture()
def many_forecasts(many_forecasts_text, _forecast_from_dict):
    return [_forecast_from_dict(fx) for fx
            in json.loads(many_forecasts_text)]


@pytest.fixture()
def single_event_observation_text():
    return b"""
{
  "_links": {
    "site": "http://127.0.0.1:5000/sites/991d15ce-7f66-11ea-96ae-0242ac150002"
  },
  "name": "Weather Station Event Observation",
  "variable": "event",
  "interval_value_type": "instantaneous",
  "interval_length": 5,
  "interval_label": "event",
  "site_id": "123e4567-e89b-12d3-a456-426655440002",
  "observation_id": "991d15ce-7f66-11ea-96ae-0242ac150002",
  "provider": "Organization 1",
  "uncertainty": 1.0,
  "extra_parameters": "",
  "created_at": "2019-03-01T12:01:48+00:00",
  "modified_at": "2019-03-01T12:01:48+00:00"
}
"""


@pytest.fixture()
def single_event_observation(single_event_observation_text,
                             _observation_from_dict):
    return _observation_from_dict(json.loads(single_event_observation_text))


@pytest.fixture()
def _event_forecast_from_dict(single_site, get_site, get_aggregate):
    def f(fx_dict):
        return datamodel.EventForecast(
            name=fx_dict['name'], variable=fx_dict['variable'],
            interval_value_type=fx_dict['interval_value_type'],
            interval_length=pd.Timedelta(f"{fx_dict['interval_length']}min"),
            interval_label=fx_dict['interval_label'],
            site=get_site(fx_dict.get('site_id')),
            aggregate=get_aggregate(fx_dict.get('aggregate_id')),
            issue_time_of_day=dt.time(int(fx_dict['issue_time_of_day'][:2]),
                                      int(fx_dict['issue_time_of_day'][3:])),
            lead_time_to_start=pd.Timedelta(f"{fx_dict['lead_time_to_start']}min"),  # NOQA
            run_length=pd.Timedelta(f"{fx_dict['run_length']}min"),
            forecast_id=fx_dict.get('forecast_id', ''),
            provider=fx_dict.get('provider', ''),
            extra_parameters=fx_dict.get('extra_parameters', ''))
    return f


@pytest.fixture()
def single_event_forecast_text():
    return b"""
{
  "_links": {
    "site": "http://127.0.0.1:5000/sites/24cbae4e-7ea6-11ea-86b1-0242ac150002",
    "aggregate": null
  },
  "name": "Weather Station Event Forecast",
  "issue_time_of_day": "05:00",
  "lead_time_to_start": 60,
  "interval_length": 5,
  "run_length": 60,
  "interval_label": "event",
  "interval_value_type": "instantaneous",
  "variable": "event",
  "forecast_id": "24cbae4e-7ea6-11ea-86b1-0242ac150002",
  "site_id": "123e4567-e89b-12d3-a456-426655440002",
  "aggregate_id": null,
  "provider": "Organization 1",
  "extra_parameters": "",
  "created_at": "2020-04-17T11:55:37+00:00",
  "modified_at": "2020-04-17T11:55:37+00:00"
}
"""


@pytest.fixture()
def many_event_forecast_text():
    return b"""
[
    {
      "_links": {
        "site": "http://127.0.0.1:5000/sites/24cbae4e-7ea6-11ea-86b1-0242ac150002",
        "aggregate": null
      },
      "name": "Weather Station Event Forecast",
      "issue_time_of_day": "05:00",
      "lead_time_to_start": 60,
      "interval_length": 5,
      "run_length": 60,
      "interval_label": "event",
      "interval_value_type": "instantaneous",
      "variable": "event",
      "forecast_id": "24cbae4e-7ea6-11ea-86b1-0242ac150002",
      "site_id": "123e4567-e89b-12d3-a456-426655440002",
      "aggregate_id": null,
      "provider": "Organization 1",
      "extra_parameters": "",
      "created_at": "2020-04-17T11:55:37+00:00",
      "modified_at": "2020-04-17T11:55:37+00:00"
    },
    {
      "_links": {
        "site": "http://127.0.0.1:5000/sites/24cbae4e-7ea6-11ea-86b1-0242ac150002",
        "aggregate": null
      },
      "name": "Solar Power Plant Event Forecast",
      "issue_time_of_day": "05:00",
      "lead_time_to_start": 60,
      "interval_length": 5,
      "run_length": 60,
      "interval_label": "event",
      "interval_value_type": "instantaneous",
      "variable": "event",
      "forecast_id": "24cbae4e-7ea6-11ea-86b1-0242ac150002",
      "site_id": "123e4567-e89b-12d3-a456-426655440002",
      "aggregate_id": null,
      "provider": "Organization 2",
      "extra_parameters": "",
      "created_at": "2020-04-17T11:55:37+00:00",
      "modified_at": "2020-04-17T11:55:37+00:00"
    }
]
"""  # NOQA


@pytest.fixture()
def single_event_forecast(single_event_forecast_text,
                          _event_forecast_from_dict):
    return _event_forecast_from_dict(json.loads(single_event_forecast_text))


@pytest.fixture()
def many_event_forecasts(many_event_forecasts_text, _event_forecast_from_dict):
    return [_event_forecast_from_dict(fx) for fx
            in json.loads(many_event_forecasts_text)]


@pytest.fixture()
def prob_forecast_constant_value_text():
    return b"""
{
  "_links": {
    "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440002",
    "aggregate": null
  },
  "created_at": "2019-03-01T11:55:37+00:00",
  "extra_parameters": "",
  "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3",
  "interval_label": "beginning",
  "interval_length": 5,
  "interval_value_type": "interval_mean",
  "issue_time_of_day": "06:00",
  "lead_time_to_start": 60,
  "modified_at": "2019-03-01T11:55:37+00:00",
  "name": "DA GHI",
  "provider": "Organization 1",
  "run_length": 1440,
  "site_id": "123e4567-e89b-12d3-a456-426655440002",
  "aggregate_id": null,
  "variable": "ghi",
  "axis": "x",
  "constant_value": 0
}
"""


@pytest.fixture()
def prob_forecast_constant_value_y_text():
    return b"""
{
  "_links": {
    "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440002",
    "aggregate": null
  },
  "created_at": "2019-03-01T11:55:37+00:00",
  "extra_parameters": "",
  "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3",
  "interval_label": "beginning",
  "interval_length": 5,
  "interval_value_type": "interval_mean",
  "issue_time_of_day": "06:00",
  "lead_time_to_start": 60,
  "modified_at": "2019-03-01T11:55:37+00:00",
  "name": "DA GHI",
  "provider": "Organization 1",
  "run_length": 1440,
  "site_id": "123e4567-e89b-12d3-a456-426655440002",
  "aggregate_id": null,
  "variable": "ghi",
  "axis": "y",
  "constant_value": 50.0
}
"""


@pytest.fixture()
def prob_forecast_text():
    return b"""
{
    "_links": {
      "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440001",
      "aggregate": null
    },
    "created_at": "2019-03-01T11:55:37+00:00",
    "extra_parameters": "",
    "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "issue_time_of_day": "06:00",
    "lead_time_to_start": 60,
    "modified_at": "2019-03-01T11:55:37+00:00",
    "name": "DA GHI",
    "provider": "Organization 1",
    "run_length": 1440,
    "site_id": "123e4567-e89b-12d3-a456-426655440002",
    "aggregate_id": null,
    "variable": "ghi",
    "axis": "x",
    "constant_values": [
        {
            "_links": {},
            "constant_value": 0,
            "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3"
        }
    ]
}
"""  # NOQA


@pytest.fixture()
def prob_forecast_y_text():
    return b"""
{
    "_links": {
      "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440001",
      "aggregate": null
    },
    "created_at": "2019-03-01T11:55:37+00:00",
    "extra_parameters": "",
    "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "issue_time_of_day": "06:00",
    "lead_time_to_start": 60,
    "modified_at": "2019-03-01T11:55:37+00:00",
    "name": "DA GHI",
    "provider": "Organization 1",
    "run_length": 1440,
    "site_id": "123e4567-e89b-12d3-a456-426655440002",
    "aggregate_id": null,
    "variable": "ghi",
    "axis": "y",
    "constant_values": [
        {
            "_links": {},
            "constant_value": 0.50,
            "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3"
        }
    ]
}
"""  # NOQA


@pytest.fixture()
def many_prob_forecasts_text():
    return b"""
[
    {
        "_links": {
          "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440001",
          "aggregate": null
        },
        "created_at": "2019-03-01T11:55:37+00:00",
        "extra_parameters": "",
        "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3",
        "interval_label": "beginning",
        "interval_length": 5,
        "interval_value_type": "interval_mean",
        "issue_time_of_day": "06:00",
        "lead_time_to_start": 60,
        "modified_at": "2019-03-01T11:55:37+00:00",
        "name": "DA GHI",
        "provider": "Organization 1",
        "run_length": 1440,
        "site_id": "123e4567-e89b-12d3-a456-426655440002",
        "aggregate_id": null,
        "variable": "ghi",
        "axis": "x",
        "constant_values": [
            {
                "_links": {},
                "constant_value": 0,
                "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3"
            }
        ]
    },
    {
        "_links": {
          "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440001",
          "aggregate": null
        },
        "created_at": "2019-03-01T11:55:37+00:00",
        "extra_parameters": "",
        "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3",
        "interval_label": "beginning",
        "interval_length": 5,
        "interval_value_type": "interval_mean",
        "issue_time_of_day": "06:00",
        "lead_time_to_start": 60,
        "modified_at": "2019-03-01T11:55:37+00:00",
        "name": "DA GHI",
        "provider": "Organization 1",
        "run_length": 1440,
        "site_id": "123e4567-e89b-12d3-a456-426655440002",
        "aggregate_id": null,
        "variable": "ghi",
        "axis": "x",
        "constant_values": [
            {
                "_links": {},
                "constant_value": 0,
                "forecast_id": "11c20780-76ae-4b11-bef1-7a75bdc784e3"
            }
        ]
    },
    {
        "_links": {
          "site": null,
          "aggregate": "http://127.0.0.1:5000/aggregates/458ffc27-df0b-11e9-b622-62adb5fd6af0"
        },
        "created_at": "2019-03-02T14:55:38+00:00",
        "extra_parameters": "",
        "forecast_id": "f6b620ca-f743-11e9-a34f-f4939feddd82",
        "interval_label": "beginning",
        "interval_length": 5,
        "interval_value_type": "interval_mean",
        "issue_time_of_day": "06:00",
        "lead_time_to_start": 60,
        "modified_at": "2019-03-02T14:55:38+00:00",
        "name": "GHI Aggregate CDF FX",
        "provider": "Organization 1",
        "run_length": 1440,
        "site_id": null,
        "variable": "ghi",
        "aggregate_id": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
        "axis": "x",
        "constant_values": [
            {
                "_links": {},
                "constant_value": 0,
                "forecast_id": "12c20780-76ae-4b11-bef1-7a75bdc784e3"
            }
        ]
    }
]
"""  # NOQA


@pytest.fixture()
def _prob_forecast_constant_value_from_dict(get_site, get_aggregate):
    def f(fx_dict):
        return datamodel.ProbabilisticForecastConstantValue(
            name=fx_dict['name'], variable=fx_dict['variable'],
            interval_value_type=fx_dict['interval_value_type'],
            interval_length=pd.Timedelta(f"{fx_dict['interval_length']}min"),
            interval_label=fx_dict['interval_label'],
            site=get_site(fx_dict.get('site_id')),
            aggregate=get_aggregate(fx_dict.get('aggregate_id')),
            issue_time_of_day=dt.time(int(fx_dict['issue_time_of_day'][:2]),
                                      int(fx_dict['issue_time_of_day'][3:])),
            lead_time_to_start=pd.Timedelta(f"{fx_dict['lead_time_to_start']}min"),  # NOQA
            run_length=pd.Timedelta(f"{fx_dict['run_length']}min"),
            forecast_id=fx_dict.get('forecast_id', ''),
            provider=fx_dict.get('provider', ''),
            extra_parameters=fx_dict.get('extra_parameters', ''),
            axis=fx_dict['axis'],
            constant_value=fx_dict['constant_value'])
    return f


@pytest.fixture()
def _prob_forecast_from_dict(get_site, prob_forecast_constant_value,
                             prob_forecast_constant_value_y,
                             get_aggregate, agg_prob_forecast_constant_value):
    def f(fx_dict):
        axis = fx_dict['axis']
        if fx_dict.get('aggregate_id') is not None:
            cv = agg_prob_forecast_constant_value
        else:
            if axis == 'x':
                cv = prob_forecast_constant_value
            else:
                cv = prob_forecast_constant_value_y
        return datamodel.ProbabilisticForecast(
            name=fx_dict['name'], variable=fx_dict['variable'],
            interval_value_type=fx_dict['interval_value_type'],
            interval_length=pd.Timedelta(f"{fx_dict['interval_length']}min"),
            interval_label=fx_dict['interval_label'],
            site=get_site(fx_dict.get('site_id')),
            aggregate=get_aggregate(fx_dict.get('aggregate_id')),
            issue_time_of_day=dt.time(int(fx_dict['issue_time_of_day'][:2]),
                                      int(fx_dict['issue_time_of_day'][3:])),
            lead_time_to_start=pd.Timedelta(f"{fx_dict['lead_time_to_start']}min"),  # NOQA
            run_length=pd.Timedelta(f"{fx_dict['run_length']}min"),
            forecast_id=fx_dict.get('forecast_id', ''),
            provider=fx_dict.get('provider', ''),
            extra_parameters=fx_dict.get('extra_parameters', ''),
            axis=axis,
            constant_values=(cv,))
    return f


@pytest.fixture()
def prob_forecast_constant_value(prob_forecast_constant_value_text,
                                 _prob_forecast_constant_value_from_dict):
    return _prob_forecast_constant_value_from_dict(
        json.loads(prob_forecast_constant_value_text))


@pytest.fixture()
def prob_forecast_constant_value_y(prob_forecast_constant_value_y_text,
                                   _prob_forecast_constant_value_from_dict):
    return _prob_forecast_constant_value_from_dict(
        json.loads(prob_forecast_constant_value_y_text))


@pytest.fixture()
def prob_forecasts(prob_forecast_text, _prob_forecast_from_dict):
    return _prob_forecast_from_dict(json.loads(prob_forecast_text))


@pytest.fixture()
def prob_forecasts_y(prob_forecast_y_text, _prob_forecast_from_dict):
    return _prob_forecast_from_dict(json.loads(prob_forecast_y_text))


@pytest.fixture()
def many_prob_forecasts(many_prob_forecasts_text, _prob_forecast_from_dict):
    return [_prob_forecast_from_dict(fx) for fx
            in json.loads(many_prob_forecasts_text)]


@pytest.fixture()
def single_forecast_observation(single_forecast, single_observation):
    return datamodel.ForecastObservation(single_forecast, single_observation)


@pytest.fixture()
def single_event_forecast_observation(single_event_forecast,
                                      single_event_observation):
    return datamodel.ForecastObservation(single_event_forecast,
                                         single_event_observation)


@pytest.fixture()
def single_prob_forecast_observation(prob_forecasts, single_observation):
    return datamodel.ForecastObservation(prob_forecasts, single_observation)


@pytest.fixture()
def single_prob_forecast_observation_y(prob_forecasts_y, single_observation):
    return datamodel.ForecastObservation(prob_forecasts_y, single_observation)


@pytest.fixture()
def single_prob_forecast_observation_reffx(prob_forecasts, single_observation):
    return datamodel.ForecastObservation(
        prob_forecasts,
        single_observation,
        reference_forecast=prob_forecasts)


@pytest.fixture()
def many_forecast_observation(many_forecasts, many_observations):
    many_ghi_forecasts = [fx for fx in many_forecasts
                          if fx.variable == 'ghi']
    many_ghi_observations = [obs for obs in many_observations
                             if obs.variable == 'ghi']
    cart_prod = itertools.product(many_ghi_forecasts, many_ghi_observations)
    return [datamodel.ForecastObservation(*c) for c in cart_prod]


@pytest.fixture()
def many_prob_forecasts_observation(many_prob_forecasts, many_observations):
    many_ghi_prob_forecasts = [pfx for pfx in many_prob_forecasts
                               if pfx.variable == 'ghi' and pfx.axis == 'x']
    many_ghi_observations = [obs for obs in many_observations
                             if obs.variable == 'ghi']
    cart_prod = itertools.product(many_ghi_prob_forecasts,
                                  many_ghi_observations)
    return [datamodel.ForecastObservation(*c) for c in cart_prod]


@pytest.fixture(params=[None, 1000])
def single_forecast_observation_norm(
        request, single_forecast, single_observation):
    return datamodel.ForecastObservation(
        single_forecast,
        single_observation,
        normalization=request.param)


@pytest.fixture(params=[None, 100, 'observation_uncertainty'])
def single_forecast_observation_uncert(
        request, single_forecast, single_observation):
    return datamodel.ForecastObservation(
        single_forecast,
        single_observation,
        uncertainty=request.param)


@pytest.fixture()
def single_forecast_observation_reffx(
        single_forecast, single_reference_forecast, single_observation):
    return datamodel.ForecastObservation(
        single_forecast,
        single_observation,
        reference_forecast=single_reference_forecast)


@pytest.fixture()
def single_forecast_ac_observation(
        ac_power_forecast_metadata, ac_power_observation_metadata):
    return datamodel.ForecastObservation(
        ac_power_forecast_metadata, ac_power_observation_metadata)


@pytest.fixture()
def single_forecast_dc_observation(
        dc_power_forecast_metadata, dc_power_observation_metadata):
    return datamodel.ForecastObservation(
        dc_power_forecast_metadata, dc_power_observation_metadata)


@pytest.fixture()
def single_forecast_wind_speed_observation(
        wind_speed_forecast_metadata, wind_speed_observation_metadata):
    return datamodel.ForecastObservation(
        wind_speed_forecast_metadata, wind_speed_observation_metadata)


@pytest.fixture()
def single_aggregate_forecast(single_site):
    forecast_agg = datamodel.Forecast(
        name="GHI Aggregate FX 60",
        issue_time_of_day=dt.time(0, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=single_site,
        forecast_id="49220780-76ae-4b11-bef1-7a75bdc784e3",
        extra_parameters='',
    )
    return forecast_agg


@pytest.fixture()
def single_forecast_aggregate(aggregate, single_aggregate_forecast):
    return datamodel.ForecastAggregate(single_aggregate_forecast, aggregate)


@pytest.fixture()
def single_forecast_aggregate_reffx(aggregate, single_aggregate_forecast):
    return datamodel.ForecastAggregate(
        single_aggregate_forecast,
        aggregate,
        reference_forecast=single_aggregate_forecast)


@pytest.fixture()
def single_prob_aggregate_forecast(single_site,
                                   agg_prob_forecast_constant_value):
    forecast_agg = datamodel.ProbabilisticForecast(
        name="GHI Aggregate FX 60",
        issue_time_of_day=dt.time(0, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=single_site,
        axis='x',
        constant_values=[agg_prob_forecast_constant_value],
        forecast_id="49220780-76ae-4b11-bef1-7a75bdc784e3",
        extra_parameters='',
    )
    return forecast_agg


@pytest.fixture()
def single_prob_forecast_aggregate(single_prob_aggregate_forecast,
                                   aggregate):
    return datamodel.ForecastAggregate(
        single_prob_aggregate_forecast, aggregate)


@pytest.fixture()
def single_prob_forecast_aggregate_reffx(single_prob_aggregate_forecast,
                                         aggregate):
    return datamodel.ForecastAggregate(
        single_prob_aggregate_forecast, aggregate,
        reference_forecast=single_prob_aggregate_forecast)


@pytest.fixture()
def report_objects(aggregate, ref_forecast_id):
    tz = 'America/Phoenix'
    start = pd.Timestamp('20190401 0000', tz=tz)
    end = pd.Timestamp('20190404 2359', tz=tz)
    site = datamodel.Site(
        name="NREL MIDC University of Arizona OASIS",
        latitude=32.22969,
        longitude=-110.95534,
        elevation=786.0,
        timezone="Etc/GMT+7",
        site_id="9f61b880-7e49-11e9-9624-0a580a8003e9",
        provider="Reference",
        extra_parameters='{"network": "NREL MIDC", "network_api_id": "UAT", "network_api_abbreviation": "UA OASIS", "observation_interval_length": 1}',  # NOQA
    )
    observation = datamodel.Observation(
        name="University of Arizona OASIS ghi",
        variable="ghi",
        interval_value_type="interval_mean",
        interval_length=pd.Timedelta("15min"),
        interval_label="ending",
        site=site,
        uncertainty=1.0,
        observation_id="9f657636-7e49-11e9-b77f-0a580a8003e9",
        extra_parameters='{"network": "NREL MIDC", "network_api_id": "UAT", "network_api_abbreviation": "UA OASIS", "observation_interval_length": 1, "network_data_label": "Global Horiz (platform) [W/m^2]"}',  # NOQA
    )
    forecast_0 = datamodel.Forecast(
        name="0 Day GFS GHI",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi", site=site,
        forecast_id="da2bc386-8712-11e9-a1c7-0a580a8200ae",
        extra_parameters='{"model": "gfs_quarter_deg_to_hourly_mean"}',
    )
    forecast_1 = datamodel.Forecast(
        name="Day Ahead GFS GHI",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("1 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=site,
        forecast_id="68a1c22c-87b5-11e9-bf88-0a580a8200ae",
        extra_parameters='{"model": "gfs_quarter_deg_to_hourly_mean"}',
    )
    forecast_agg = datamodel.Forecast(
        name="GHI Aggregate FX 60",
        issue_time_of_day=dt.time(0, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=site,
        forecast_id="49220780-76ae-4b11-bef1-7a75bdc784e3",
        extra_parameters='',
    )
    fxobs0 = datamodel.ForecastObservation(
        forecast_0,
        observation,
        # report_text parsing will ensure unc can be done dynamically too
        uncertainty=observation.uncertainty)
    forecast_ref = forecast_0.replace(forecast_id=ref_forecast_id)
    fxobs1 = datamodel.ForecastObservation(
        forecast_1,
        observation,
        normalization=1000.,
        uncertainty=15.,
        reference_forecast=forecast_ref)
    fxagg0 = datamodel.ForecastAggregate(
        forecast_agg,
        aggregate,
        uncertainty=5.)
    quality_flag_filter = datamodel.QualityFlagFilter(
        (
            "USER FLAGGED",
            "NIGHTTIME",
            "LIMITS EXCEEDED",
            "STALE VALUES",
            "INTERPOLATED VALUES",
            "INCONSISTENT IRRADIANCE COMPONENTS",
        )
    )
    timeofdayfilter = datamodel.TimeOfDayFilter((dt.time(12, 0),
                                                 dt.time(14, 0)))
    report_params = datamodel.ReportParameters(
        name="NREL MIDC OASIS GHI Forecast Analysis",
        start=start,
        end=end,
        object_pairs=(fxobs0, fxobs1, fxagg0),
        metrics=("mae", "rmse", "mbe", "s"),
        categories=("total", "date", "hour"),
        filters=(quality_flag_filter, timeofdayfilter)
    )
    report = datamodel.Report(
        report_id="56c67770-9832-11e9-a535-f4939feddd82",
        report_parameters=report_params
    )
    return report, observation, forecast_0, forecast_1, aggregate, forecast_agg


@pytest.fixture()
def event_report_objects():
    tz = 'America/Phoenix'
    start = pd.Timestamp('20190401T0000', tz=tz)
    end = pd.Timestamp('20190404T2359', tz=tz)
    site = datamodel.Site(
        name="NREL MIDC University of Arizona OASIS",
        latitude=32.22969,
        longitude=-110.95534,
        elevation=786.0,
        timezone="Etc/GMT+7",
        site_id="9f61b880-7e49-11e9-9624-0a580a8003e9",
        provider="Reference",
        extra_parameters=''
    )
    obs = datamodel.Observation(
        name="Example Event Observation",
        variable="event",
        interval_value_type="instantaneous",
        interval_length=pd.Timedelta("15min"),
        interval_label="event",
        site=site,
        uncertainty=1.0,
        observation_id="9f657636-7e49-11e9-b77f-0a580a8003e9",
        extra_parameters='',
    )
    fx0 = datamodel.EventForecast(
        name="Example Event Forecast",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="event",
        interval_value_type="instantaneous",
        variable="event",
        site=site,
        forecast_id="da2bc386-8712-11e9-a1c7-0a580a8200ae",
        extra_parameters='',
    )
    fx1 = datamodel.EventForecast(
        name="Alternative Example Event Forecast",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("1 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="event",
        interval_value_type="instantaneous",
        variable="event",
        site=site,
        forecast_id="68a1c22c-87b5-11e9-bf88-0a580a8200ae",
        extra_parameters='',
    )

    fxobs0 = datamodel.ForecastObservation(fx0, obs)
    fxobs1 = datamodel.ForecastObservation(fx1, obs)

    report_params = datamodel.ReportParameters(
        name="Example Event Report",
        start=start,
        end=end,
        object_pairs=(fxobs0, fxobs1),
        metrics=("pod", "far", "pofd", "csi", "ebias", "ea"),
        categories=("total", "date", "hour"),
    )

    report = datamodel.Report(
        report_id="56c67770-9832-11e9-a535-f4939feddd82",
        report_parameters=report_params
    )

    return report, obs, fx0, fx1


@pytest.fixture()
def cdf_and_cv_report_objects(aggregate, ref_forecast_id):
    tz = 'America/Denver'
    start = pd.Timestamp('20200401T0000', tz=tz)
    end = pd.Timestamp('20200407T2359', tz=tz)
    axis = 'y'
    site = datamodel.Site(
        name="NOAA SOLRAD Albuquerque New Mexico",
        latitude=35.03796,
        longitude=-106.62211,
        elevation=1617.0,
        timezone="Etc/GMT+7",
        site_id="c26ba076-7e49-11e9-bd90-0a580a8003e9",
        provider="Reference",
        extra_parameters=''
    )
    obs = datamodel.Observation(
        name="Albuquerque New Mexico ghi",
        variable="ghi",
        interval_value_type="interval mean",
        interval_length=pd.Timedelta("1min"),
        interval_label="ending",
        site=site,
        uncertainty=1.0,
        observation_id="c26ff506-7e49-11e9-beae-0a580a8003e9",
        extra_parameters='',
    )
    cv0 = datamodel.ProbabilisticForecastConstantValue(
        name="Day Ahead GEFS ghi",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=site,
        forecast_id="26a449ca-080b-11ea-aabd-0a580a8200e0",
        axis=axis,
        constant_value=0.0,
        extra_parameters='',
    )
    cv0_1 = cv0.replace(constant_value=25.0,
                        forecast_id='26a6684c-080b-11ea-bab3-0a580a8200e0')
    cv0_2 = cv0.replace(constant_value=50.0,
                        forecast_id='26a82ba8-080b-11ea-a36b-0a580a8200e0')
    cv0_3 = cv0.replace(constant_value=75.0,
                        forecast_id='26a9e678-080b-11ea-8989-0a580a8200e0')
    pfx0 = datamodel.ProbabilisticForecast(
        name="Day Ahead GEFS ghi",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=site,
        forecast_id="269d757a-080b-11ea-8107-0a580a8200e0",
        axis=axis,
        constant_values=(cv0_1, cv0_2, cv0_3),
        extra_parameters='',
    )
    cv1 = datamodel.ProbabilisticForecastConstantValue(
        name="Hour Ahead GEFS ghi (example only)",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("0 days 01:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=site,
        forecast_id="71ee6aff-7011-4610-bbae-2c9709ab1999",
        constant_value=0.0,
        axis=axis,
    )
    cv1_1 = cv1.replace(constant_value=25.0,
                        forecast_id='71ee6aff-7011-4610-1111-2c9709ab1999')
    cv1_2 = cv1.replace(constant_value=50.0,
                        forecast_id='71ee6aff-7011-4610-2222-2c9709ab1999')
    cv1_3 = cv1.replace(constant_value=75.0,
                        forecast_id='71ee6aff-7011-4610-3333-2c9709ab1999')
    pfx1 = datamodel.ProbabilisticForecast(
        name="Hour Ahead GEFS ghi (example only)",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("0 days 01:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=site,
        forecast_id="26b514ec-080b-11ea-bbea-0a580a8200e0",
        axis=axis,
        constant_values=(cv1_1, cv1_2, cv1_3),
        extra_parameters='',
    )
    cvagg_1 = cv1.replace(constant_value=25.0,
                          forecast_id='1a6cc8c0-c433-432b-1111-31a1fac500d5')
    cvagg_2 = cv1.replace(constant_value=50.0,
                          forecast_id='1a6cc8c0-c433-432b-2222-31a1fac500d5')
    cvagg_3 = cv1.replace(constant_value=75.0,
                          forecast_id='1a6cc8c0-c433-432b-3333-31a1fac500d5')
    pfx2 = datamodel.ProbabilisticForecast(
        name="GHI Aggregate FX 60 (example only)",
        issue_time_of_day=dt.time(7, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("0 days 01:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=site,
        forecast_id="49220780-76ae-4b11-bef1-7a75bdc784e3",
        axis=axis,
        constant_values=(cvagg_1, cvagg_2, cvagg_3),
        extra_parameters='',
    )

    # CDFs
    pfxobs0 = datamodel.ForecastObservation(
        pfx0,
        obs,
        uncertainty=obs.uncertainty
    )
    pfx_ref = pfx0.replace(forecast_id=ref_forecast_id)
    pfxobs1 = datamodel.ForecastObservation(
        pfx1,
        obs,
        normalization=1000,
        uncertainty=15.,
        reference_forecast=pfx_ref
    )
    pfx_agg = datamodel.ForecastAggregate(
        pfx2,
        aggregate
    )

    # single, constant values
    pfxcvobs0_1 = datamodel.ForecastObservation(
        cv0_1,
        obs,
        uncertainty=obs.uncertainty
    )
    pfxcvobs0_2 = datamodel.ForecastObservation(
        cv0_2,
        obs,
        normalization=1000,
        reference_forecast=cv0_3.replace(forecast_id=ref_forecast_id)
    )
    pfxcvagg0_3 = datamodel.ForecastAggregate(
        cv0_3,
        aggregate,
        uncertainty=15.,
    )

    quality_flag_filter = datamodel.QualityFlagFilter(
        (
            "USER FLAGGED",
            "NIGHTTIME",
            "LIMITS EXCEEDED",
            "STALE VALUES",
            "INTERPOLATED VALUES"
        )
    )
    timeofdayfilter = datamodel.TimeOfDayFilter((dt.time(6, 0),
                                                 dt.time(8, 0)))
    report_params = datamodel.ReportParameters(
        name="Albuquerque GHI Forecast Analysis",
        start=start,
        end=end,
        object_pairs=(pfxobs0, pfxobs1, pfxcvobs0_1, pfxcvobs0_2,
                      pfx_agg, pfxcvagg0_3),
        metrics=("crps", "bs", "bss", "unc"),
        categories=("total", "date", "hour"),
        filters=(quality_flag_filter, timeofdayfilter)
    )
    report = datamodel.Report(
        report_id="1179f8c1-4d76-479c-a826-053d6b6e5116",
        report_parameters=report_params
    )
    return report, obs, pfx0, pfx1, cv0_1, cv0_2, aggregate, pfx2, cv0_3


@pytest.fixture()
def event_report_text():
    return b"""
    {"created_at": "2019-06-26T16:49:18+00:00",
     "modified_at": "2019-06-26T16:49:18+00:00",
     "report_id": "56c67770-9832-11e9-a535-f4939feddd82",
     "report_parameters": {
         "name": "Example Event Report",
         "start": "2019-04-01T00:00:00-07:00",
         "end": "2019-04-04T23:59:00-07:00",
         "metrics": ["pod", "far", "pofd", "csi", "ebias", "ea"],
         "categories": ["total", "date", "hour"],
         "object_pairs": [
             {"forecast": "da2bc386-8712-11e9-a1c7-0a580a8200ae",
              "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9"},
             {"forecast": "68a1c22c-87b5-11e9-bf88-0a580a8200ae",
              "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9"}
         ]
     },
     "raw_report": null,
     "values": [],
     "status": "pending"}
    """


@pytest.fixture()
def quality_filter():
    return datamodel.QualityFlagFilter(
        (
            "USER FLAGGED",
            "NIGHTTIME",
            "LIMITS EXCEEDED",
            "STALE VALUES",
            "INTERPOLATED VALUES",
            "INCONSISTENT IRRADIANCE COMPONENTS",
        )
    )


@pytest.fixture()
def quality_filter_dict():
    return {'quality_flags': (
        "USER FLAGGED",
        "NIGHTTIME",
        "LIMITS EXCEEDED",
        "STALE VALUES",
        "INTERPOLATED VALUES",
        "INCONSISTENT IRRADIANCE COMPONENTS",
    )}


@pytest.fixture()
def timeofdayfilter():
    return datamodel.TimeOfDayFilter(
        time_of_day_range=(dt.time(12, 0), dt.time(14, 0))
    )


@pytest.fixture()
def timeofdayfilter_dict():
    return {'time_of_day_range': ("12:00", "14:00")}


@pytest.fixture()
def valuefilter(single_forecast):
    return datamodel.ValueFilter(
        metadata=single_forecast,
        value_range=(100.0, 900.0)
    )


@pytest.fixture()
def valuefilter_dict(single_forecast):
    return {
        'metadata': single_forecast.to_dict(),
        'value_range': (100.0, 900.0)
    }


@pytest.fixture
def ref_forecast_id():
    return "refbc386-8712-11e9-a1c7-0a580a8200ae"


@pytest.fixture()
def report_params_dict(report_objects, quality_filter_dict,
                       timeofdayfilter_dict, ref_forecast_id):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = \
        report_objects
    report_params = report.report_parameters
    ref_dict = forecast_0.to_dict()
    ref_dict.update(forecast_id=ref_forecast_id)
    return {
        'name': report_params.name,
        'start': report_params.start,
        'end': report_params.end,
        'object_pairs': (
            {'forecast': forecast_0.to_dict(),
             'observation': observation.to_dict(),
             'uncertainty': observation.uncertainty},
            {'forecast': forecast_1.to_dict(),
             'observation': observation.to_dict(),
             'normalization': 1000.,
             'uncertainty': 15.,
             'reference_forecast': ref_dict},
            {'forecast': forecast_agg.to_dict(),
             'aggregate': aggregate.to_dict(),
             'uncertainty': 5.},
        ),
        'metrics': ('mae', 'rmse', 'mbe', 's'),
        'filters': (quality_filter_dict, timeofdayfilter_dict),
    }


@pytest.fixture()
def report_params(report_objects):
    return report_objects[0].report_parameters


@pytest.fixture()
def report_dict(report_params_dict, report_objects):
    report = report_objects[0]
    return {
        'report_parameters': report_params_dict,
        'status': report.status,
        'report_id': report.report_id,
        '__version__': report.__version__
    }


@pytest.fixture()
def report_text():
    return b"""
    {"created_at": "2019-06-26T16:49:18+00:00",
     "modified_at": "2019-06-26T16:49:18+00:00",
     "report_id": "56c67770-9832-11e9-a535-f4939feddd82",
     "report_parameters": {
         "name": "NREL MIDC OASIS GHI Forecast Analysis",
         "start": "2019-04-01T00:00:00-07:00",
         "end": "2019-04-04T23:59:00-07:00",
         "filters": [
             {"quality_flags": [
                 "USER FLAGGED",
                 "NIGHTTIME",
                 "LIMITS EXCEEDED",
                 "STALE VALUES",
                 "INTERPOLATED VALUES",
                 "INCONSISTENT IRRADIANCE COMPONENTS"
             ]},
             {"time_of_day_range": ["12:00", "14:00"]}
         ],
         "metrics": ["mae", "rmse", "mbe", "s"],
         "categories": ["total", "date", "hour"],
         "object_pairs": [
             {"forecast": "da2bc386-8712-11e9-a1c7-0a580a8200ae",
              "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9",
              "uncertainty": "observation_uncertainty"},
             {"forecast": "68a1c22c-87b5-11e9-bf88-0a580a8200ae",
              "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9",
              "normalization": "1000",
              "uncertainty": "15",
              "reference_forecast": "refbc386-8712-11e9-a1c7-0a580a8200ae"},
             {"forecast": "49220780-76ae-4b11-bef1-7a75bdc784e3",
              "aggregate": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
              "uncertainty": "5"}
         ]
     },
     "raw_report": null,
     "values": [],
     "status": "pending"}
    """


@pytest.fixture
def metric_index():
    def index(category):
        if category == 'date':
            return '2019-01-01'
        else:
            return 1
    return index


@pytest.fixture
def preprocessing_result_types():
    return (preprocessing.VALIDATION_RESULT_TOTAL_STRING,
            'Forecast ' + preprocessing.DISCARD_DATA_STRING,
            'Forecast ' + preprocessing.UNDEFINED_DATA_STRING,
            'Observation ' + preprocessing.DISCARD_DATA_STRING,
            'Observation ' + preprocessing.UNDEFINED_DATA_STRING)


@pytest.fixture
def report_metrics(metric_index):
    """Produces dummy MetricResult list for a RawReport"""
    def gen(report):
        metrics = ()
        for fxobs in report.report_parameters.object_pairs:
            values = []
            if hasattr(fxobs, 'observation'):
                obsid = fxobs.observation.observation_id
            else:
                obsid = fxobs.aggregate.aggregate_id
            metrics_dict = {
                'name': f'{fxobs.forecast.name}',
                'forecast_id': fxobs.forecast.forecast_id,
                'observation_id': obsid
            }
            for metric, category in itertools.product(
                report.report_parameters.metrics,
                report.report_parameters.categories
            ):
                values.append(datamodel.MetricValue.from_dict(
                    {
                        'category': category,
                        'metric': metric,
                        'value': 2,
                        'index': metric_index(category),
                    }
                ))
            metrics_dict['values'] = values
            metrics = metrics + (
                datamodel.MetricResult.from_dict(metrics_dict),)
        return metrics
    return gen


@pytest.fixture()
def fail_pdf():
    with open(resource_filename(
        Requirement.parse('solarforecastarbiter'),
            'solarforecastarbiter/reports/figures/fail.pdf'),
              'rb'
    ) as f:
        return base64.a85encode(f.read()).decode()


@pytest.fixture()
def raw_report(report_objects, report_metrics, preprocessing_result_types,
               ref_forecast_id, fail_pdf):
    report, obs, fx0, fx1, agg, fxagg = report_objects

    def gen(with_series):
        def ser(interval_length):
            ser_index = pd.date_range(
                report.report_parameters.start,
                report.report_parameters.end,
                freq=to_offset(interval_length),
                name='timestamp')
            ser_value = pd.Series(
                np.repeat(100, len(ser_index)), name='value',
                index=ser_index)
            return ser_value
        il0 = fx0.interval_length
        qflags = list(
            f.quality_flags for f in report.report_parameters.filters if
            isinstance(f, datamodel.QualityFlagFilter)
        )
        qflags = list(qflags[0])
        fxobs0 = datamodel.ProcessedForecastObservation(
            fx0.name,
            datamodel.ForecastObservation(
                fx0,
                obs,
                uncertainty=obs.uncertainty),
            fx0.interval_value_type,
            il0,
            fx0.interval_label,
            valid_point_count=len(ser(il0)),
            validation_results=tuple(datamodel.ValidationResult(
                flag=f, count=0) for f in qflags),
            preprocessing_results=tuple(datamodel.PreprocessingResult(
                name=t, count=0) for t in preprocessing_result_types),
            forecast_values=ser(il0) if with_series else fx0.forecast_id,
            observation_values=ser(il0) if with_series else obs.observation_id,
            uncertainty=1.
        )
        il1 = fx1.interval_length
        fxobs1 = datamodel.ProcessedForecastObservation(
            fx1.name,
            datamodel.ForecastObservation(
                fx1,
                obs,
                normalization=1000.,
                uncertainty=15.,
                reference_forecast=fx0.replace(forecast_id=ref_forecast_id)),
            fx1.interval_value_type,
            il1,
            fx1.interval_label,
            valid_point_count=len(ser(il1)),
            validation_results=tuple(datamodel.ValidationResult(
                flag=f, count=0) for f in qflags),
            preprocessing_results=tuple(datamodel.PreprocessingResult(
                name=t, count=0) for t in preprocessing_result_types),
            forecast_values=ser(il1) if with_series else fx1.forecast_id,
            observation_values=ser(il1) if with_series else obs.observation_id,
            reference_forecast_values=(
                ser(il0) if with_series else ref_forecast_id),
            normalization_factor=1000.,
            uncertainty=15.,
        )
        ilagg = fxagg.interval_length
        fxagg_ = datamodel.ProcessedForecastObservation(
            fxagg.name,
            datamodel.ForecastAggregate(
                fxagg,
                agg,
                uncertainty=5.),
            fxagg.interval_value_type,
            ilagg,
            fxagg.interval_label,
            valid_point_count=len(ser(ilagg)),
            validation_results=tuple(datamodel.ValidationResult(
                flag=f, count=0) for f in qflags),
            preprocessing_results=tuple(datamodel.PreprocessingResult(
                name=t, count=0) for t in preprocessing_result_types),
            forecast_values=ser(ilagg) if with_series else fxagg.forecast_id,
            observation_values=ser(ilagg) if with_series else agg.aggregate_id,
            uncertainty=5.
        )
        figs = datamodel.RawReportPlots(
            (
                datamodel.PlotlyReportFigure.from_dict(
                    {
                        'name': 'mae tucson ghi',
                        'spec': '{"data":[{"x":[1],"y":[1],"type":"bar"}]}',
                        'pdf': fail_pdf,
                        'figure_type': 'bar',
                        'category': 'total',
                        'metric': 'mae',
                        'figure_class': 'plotly',
                    }
                ),), '4.5.3',
        )
        raw = datamodel.RawReport(
            generated_at=report.report_parameters.end,
            versions=(),
            timezone=obs.site.timezone,
            plots=figs,
            metrics=report_metrics(report),
            processed_forecasts_observations=(fxobs0, fxobs1, fxagg_))
        return raw
    return gen


@pytest.fixture
def report_with_raw(report_dict, raw_report):
    report_dict['raw_report'] = raw_report(True)
    report_dict['status'] = 'complete'
    report = datamodel.Report.from_dict(report_dict)
    return report


@pytest.fixture()
def failed_report(report_dict):
    report_dict['raw_report'] = datamodel.RawReport(
        generated_at=pd.Timestamp.utcnow(), timezone='UTC',
        versions=(), plots=None, metrics=(),
        processed_forecasts_observations=(),
        messages=(datamodel.ReportMessage(message='Report Failed',
                                          step='make',
                                          level='CRITICAL',
                                          function='fn'))
    )
    report_dict['status'] = 'failed'
    return datamodel.Report.from_dict(report_dict)


@pytest.fixture()
def pending_report(report_dict):
    report_dict['status'] = 'pending'
    report_dict['raw_report'] = None
    return datamodel.Report.from_dict(report_dict)


@pytest.fixture()
def aggregate_text():
    return b"""
{
  "aggregate_id": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
  "aggregate_type": "mean",
  "created_at": "2019-09-24T12:00:00+00:00",
  "description": "ghi agg",
  "extra_parameters": "extra",
  "interval_label": "ending",
  "interval_length": 60,
  "interval_value_type": "interval_mean",
  "modified_at": "2019-09-24T12:00:00+00:00",
  "name": "Test Aggregate ghi",
  "observations": [
    {
      "_links": {
        "observation": "http://localhost:5000/observations/123e4567-e89b-12d3-a456-426655440000/metadata"
      },
      "created_at": "2019-09-25T00:00:00+00:00",
      "effective_from": "2019-01-01T00:00:00+00:00",
      "effective_until": "2020-01-01T00:00:00+00:00",
      "observation_deleted_at": null,
      "observation_id": "123e4567-e89b-12d3-a456-426655440000"
    },
    {
      "_links": {
        "observation": "http://localhost:5000/observations/e0da0dea-9482-4073-84de-f1b12c304d23/metadata"
      },
      "created_at": "2019-09-25T00:00:00+00:00",
      "effective_from": "2019-01-01T00:00:00+00:00",
      "effective_until": null,
      "observation_deleted_at": null,
      "observation_id": "e0da0dea-9482-4073-84de-f1b12c304d23"
    },
    {
      "_links": {
        "observation": "http://localhost:5000/observations/b1dfe2cb-9c8e-43cd-afcf-c5a6feaf81e2/metadata"
      },
      "created_at": "2019-09-25T00:00:00+00:00",
      "effective_from": "2019-01-01T00:00:00+00:00",
      "effective_until": null,
      "observation_deleted_at": null,
      "observation_id": "b1dfe2cb-9c8e-43cd-afcf-c5a6feaf81e2"
    }
  ],
  "provider": "Organization 1",
  "timezone": "America/Denver",
  "variable": "ghi"
}
"""  # NOQA


@pytest.fixture()
def aggregate_observations(aggregate_text, many_observations):
    obsd = {o.observation_id: o for o in many_observations}
    aggd = json.loads(aggregate_text)

    def _tstamp(val):
        if val is None:
            return val
        else:
            return pd.Timestamp(val)

    aggobs = tuple([datamodel.AggregateObservation(
        observation=obsd[o['observation_id']],
        effective_from=_tstamp(o['effective_from']),
        effective_until=_tstamp(o['effective_until']),
        observation_deleted_at=_tstamp(o['observation_deleted_at']))
        for o in aggd['observations']])
    return aggobs


@pytest.fixture()
def single_aggregate_observation_text(single_observation_text_with_site_text):
    return (b'{"observation": ' + single_observation_text_with_site_text +
            b', "effective_from": "2019-01-03T13:00:00Z"}')


@pytest.fixture()
def single_aggregate_observation(single_observation):
    return datamodel.AggregateObservation(
        observation=single_observation,
        effective_from=pd.Timestamp('2019-01-03T13:00:00Z')
    )


@pytest.fixture()
def aggregate(aggregate_text, aggregate_observations):
    aggd = json.loads(aggregate_text)
    return datamodel.Aggregate(
        name=aggd['name'], description=aggd['description'],
        variable=aggd['variable'], aggregate_type=aggd['aggregate_type'],
        interval_length=pd.Timedelta(f"{aggd['interval_length']}min"),
        interval_label=aggd['interval_label'],
        timezone=aggd['timezone'], aggregate_id=aggd['aggregate_id'],
        provider=aggd['provider'], extra_parameters=aggd['extra_parameters'],
        observations=aggregate_observations)


@pytest.fixture()
def aggregate_forecast_text():
    return b"""
{
  "_links": {
    "site": null,
    "aggregate": "http://localhost:5000/aggregates/458ffc27-df0b-11e9-b622-62adb5fd6af0"
  },
  "aggregate_id": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
  "created_at": "2019-03-01T11:55:37+00:00",
  "extra_parameters": "",
  "forecast_id": "39220780-76ae-4b11-bef1-7a75bdc784e3",
  "interval_label": "beginning",
  "interval_length": 60,
  "interval_value_type": "interval_mean",
  "issue_time_of_day": "06:00",
  "lead_time_to_start": 60,
  "modified_at": "2019-03-01T11:55:37+00:00",
  "name": "GHI Aggregate FX",
  "provider": "Organization 1",
  "run_length": 1440,
  "site_id": null,
  "variable": "ghi"
}
"""  # NOQA


@pytest.fixture()
def aggregateforecast(aggregate_forecast_text, aggregate):
    fx_dict = json.loads(aggregate_forecast_text)
    return datamodel.Forecast(
        name=fx_dict['name'], variable=fx_dict['variable'],
        interval_value_type=fx_dict['interval_value_type'],
        interval_length=pd.Timedelta(f"{fx_dict['interval_length']}min"),
        interval_label=fx_dict['interval_label'],
        aggregate=aggregate,
        issue_time_of_day=dt.time(int(fx_dict['issue_time_of_day'][:2]),
                                  int(fx_dict['issue_time_of_day'][3:])),
        lead_time_to_start=pd.Timedelta(f"{fx_dict['lead_time_to_start']}min"),  # NOQA
        run_length=pd.Timedelta(f"{fx_dict['run_length']}min"),
        forecast_id=fx_dict.get('forecast_id', ''),
        provider=fx_dict.get('provider', ''),
        extra_parameters=fx_dict.get('extra_parameters', ''))


@pytest.fixture()
def aggregate_prob_forecast_text():
    return b"""
{
    "_links": {
      "site": null,
      "aggregate": "http://127.0.0.1:5000/aggregates/458ffc27-df0b-11e9-b622-62adb5fd6af0"
    },
    "created_at": "2019-03-02T14:55:38+00:00",
    "extra_parameters": "",
    "forecast_id": "f6b620ca-f743-11e9-a34f-f4939feddd82",
    "interval_label": "beginning",
    "interval_length": 5,
    "interval_value_type": "interval_mean",
    "issue_time_of_day": "06:00",
    "lead_time_to_start": 60,
    "modified_at": "2019-03-02T14:55:38+00:00",
    "name": "GHI Aggregate CDF FX",
    "provider": "Organization 1",
    "run_length": 1440,
    "site_id": null,
    "variable": "ghi",
    "aggregate_id": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
    "axis": "x",
    "constant_values": [
        {
            "_links": {},
            "constant_value": 0,
            "forecast_id": "12c20780-76ae-4b11-bef1-7a75bdc784e3"
        }
    ]
}
"""  # NOQA


@pytest.fixture()
def agg_prob_forecast_constant_value_text():
    return b"""
{
  "_links": {
    "site": "http://127.0.0.1:5000/sites/123e4567-e89b-12d3-a456-426655440002",
    "aggregate": null
  },
  "created_at": "2019-03-01T11:55:37+00:00",
  "extra_parameters": "",
  "forecast_id": "12c20780-76ae-4b11-bef1-7a75bdc784e3",
  "interval_label": "beginning",
  "interval_length": 5,
  "interval_value_type": "interval_mean",
  "issue_time_of_day": "06:00",
  "lead_time_to_start": 60,
  "name": "GHI Aggregate CDF FX",
  "provider": "Organization 1",
  "run_length": 1440,
  "aggregate_id": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
  "site_id": null,
  "variable": "ghi",
  "axis": "x",
  "constant_value": 0
}
"""


@pytest.fixture()
def agg_prob_forecast_constant_value(_prob_forecast_constant_value_from_dict,
                                     agg_prob_forecast_constant_value_text):
    return _prob_forecast_constant_value_from_dict(
        json.loads(agg_prob_forecast_constant_value_text))


@pytest.fixture()
def aggregate_prob_forecast(aggregate_prob_forecast_text,
                            agg_prob_forecast_constant_value,
                            aggregate):
    fx_dict = json.loads(aggregate_prob_forecast_text)
    fx_dict['constant_values'] = agg_prob_forecast_constant_value
    return datamodel.ProbabilisticForecast(
        name=fx_dict['name'], variable=fx_dict['variable'],
        interval_value_type=fx_dict['interval_value_type'],
        interval_length=pd.Timedelta(f"{fx_dict['interval_length']}min"),
        interval_label=fx_dict['interval_label'],
        site=None,
        aggregate=aggregate,
        issue_time_of_day=dt.time(int(fx_dict['issue_time_of_day'][:2]),
                                  int(fx_dict['issue_time_of_day'][3:])),
        lead_time_to_start=pd.Timedelta(f"{fx_dict['lead_time_to_start']}min"),
        run_length=pd.Timedelta(f"{fx_dict['run_length']}min"),
        forecast_id=fx_dict.get('forecast_id', ''),
        extra_parameters=fx_dict.get('extra_parameters', ''),
        provider=fx_dict.get('provider', ''),
        axis=fx_dict['axis'],
        constant_values=(agg_prob_forecast_constant_value, ))


@pytest.fixture
def metric_value_dict():
    return {
        'category': 'total',
        'metric': 'mae',
        'index': 1,
        'value': 1,
    }


@pytest.fixture
def metric_value(metric_value_dict):
    return datamodel.MetricValue.from_dict(metric_value_dict)


@pytest.fixture
def metric_result_dict(metric_value_dict):
    return {
        'name': 'total tucson ghi mae',
        'forecast_id': 'fxid',
        'values': [metric_value_dict],
        'observation_id': 'obsid',
        'aggregate_id': None,
    }


@pytest.fixture
def metric_result(metric_result_dict):
    return datamodel.MetricResult.from_dict(metric_result_dict)


@pytest.fixture
def validation_result_dict():
    return {
        'flag': 2,
        'count': 1,
    }


@pytest.fixture
def validation_result(validation_result_dict):
    return datamodel.ValidationResult.from_dict(validation_result_dict)


@pytest.fixture
def preprocessing_result_dict():
    return {
        'name': "Undefined Values",
        'count': 1
    }


@pytest.fixture
def preprocessing_result(preprocessing_result_dict):
    return datamodel.PreprocessingResult.from_dict(preprocessing_result_dict)


@pytest.fixture
def report_metadata_dict():
    return {
        'name': 'Report 1',
        'start': '2019-01-01T00:00Z',
        'end': '2019-01-02T00:00Z',
        'now': '2019-01-03T00:00Z',
        'timezone': 'America/Pheonix',
        'versions': (),
    }


@pytest.fixture
def plotly_report_figure_dict(fail_pdf):
    return {
        'name': 'mae tucson ghi',
        'spec': '{"data":[{"x":[1],"y":[1],"type":"bar"}]}',
        'pdf': fail_pdf,
        'figure_type': 'bar',
        'category': 'total',
        'metric': 'mae',
        'figure_class': 'plotly',
    }


@pytest.fixture
def bokeh_report_figure_dict():
    return{
        'name': 'mae tucson ghi',
        'div': '<div></div>',
        'svg': '<svg></svg>',
        'figure_type': 'bar',
        'category': 'total',
        'metric': 'mae',
        'figure_class': 'bokeh',
    }


@pytest.fixture
def plotly_report_figure(plotly_report_figure_dict):
    return datamodel.PlotlyReportFigure.from_dict(plotly_report_figure_dict)


@pytest.fixture
def bokeh_report_figure(bokeh_report_figure_dict):
    return datamodel.BokehReportFigure.from_dict(bokeh_report_figure_dict)


@pytest.fixture
def report_message_dict():
    return {
        'message': 'Report was bad',
        'step': 'calculating metrics',
        'level': 'exception',
        'function': 'calculate_deterministic_metrics',
    }


@pytest.fixture
def report_message(report_message_dict):
    return datamodel.ReportMessage.from_dict(report_message_dict)


@pytest.fixture
def raw_report_dict_with_event(fail_pdf):
    return {
        'data_checksum': None,
        'generated_at': '2020-04-22T20:02:40+00:00',
        'messages': [],
        'metrics': [{
            'aggregate_id': None,
            'forecast_id': '24cbae4e-7ea6-11ea-86b1-0242ac150002',
            'name': 'Weather Station Event Forecast',
            'observation_id': '991d15ce-7f66-11ea-96ae-0242ac150002',
            'values': [{
                'category': 'total',
                'index': '0',
                'metric': 'pod',
                'value': 0.3333333333333333}],
        }],
        'plots': {
            'bokeh_version': None,
            'figures': [{
                'category': 'total',
                'figure_class': 'plotly',
                'figure_type': 'bar',
                'metric': 'pod',
                'name': 'all',
                'spec': "{}",
                'pdf': fail_pdf}],
            'plotly_version': '4.5.3',
            'script': None},
        'processed_forecasts_observations': [{
            'cost_per_unit_error': 0.0,
            'forecast_values': '3907ab36-84d4-11ea-8bc4-54bf64606445',
            'interval_label': 'event',
            'interval_length': 5.0,
            'interval_value_type': 'instantaneous',
            'name': 'Weather Station Event Forecast',
            'normalization_factor': np.nan,
            'observation_values': '391820f6-84d4-11ea-8b57-54bf64606445',
            'original': {
                'cost_per_unit_error': 0.0,
                'forecast': {
                    'aggregate': None,
                    'extra_parameters': '',
                    'forecast_id': '24cbae4e-7ea6-11ea-86b1-0242ac150002',
                    'interval_label': 'event',
                    'interval_length': 5.0,
                    'interval_value_type': 'instantaneous',
                    'issue_time_of_day': '05:00',
                    'lead_time_to_start': 60.0,
                    'name': 'Weather Station Event Forecast',
                    'provider': 'Organization 1',
                    'run_length': 60.0,
                    'site': {
                        'elevation': 595.0,
                        'extra_parameters': "",
                        'latitude': 42.19,
                        'longitude': -122.7,
                        'name': 'Weather Station',
                        'provider': 'Organization 1',
                        'site_id': '123e4567-e89b-12d3-a456-426655440001',
                        'timezone': 'Etc/GMT+8'},
                    'variable': 'event'},
                'normalization': np.nan,
                'observation': {
                    'extra_parameters': '',
                    'interval_label': 'event',
                    'interval_length': 5.0,
                    'interval_value_type': 'instantaneous',
                    'name': 'Weather Station Event Observation',
                    'observation_id': '991d15ce-7f66-11ea-96ae-0242ac150002',
                    'provider': 'Organization 1',
                    'site': {
                        'elevation': 595.0,
                        'extra_parameters': "",
                        'latitude': 42.19,
                        'longitude': -122.7,
                        'name': 'Weather Station',
                        'provider': 'Organization 1',
                        'site_id': '123e4567-e89b-12d3-a456-426655440001',
                        'timezone': 'Etc/GMT+8'},
                    'uncertainty': 1.0,
                    'variable': 'event'},
                'reference_forecast': None,
                'uncertainty': None},
            'preprocessing_results': [
                {'count': 0,
                 'name': 'TOTAL FLAGGED VALUES DISCARDED'},
                {'count': 0,
                 'name': 'EventForecast Values Discarded by Alignment'},
                {'count': 0,
                 'name': 'Observation Values Discarded by Alignment'},
                {'count': 0,
                 'name': 'EventForecast Undefined Values'},
                {'count': 0,
                 'name': 'Observation Undefined Values'}],
            'reference_forecast_values': None,
            'uncertainty': None,
            'valid_point_count': 7,
            'validation_results': []}],
        'timezone': 'Etc/GMT+8',
        'versions': [['solarforecastarbiter', '1.0b4+32.gc77b43d'],
                     ['pvlib', '0.7.1'],
                     ['pandas', '1.0.3'],
                     ['numpy', '1.18.1'],
                     ['bokeh', '1.4.0'],
                     ['netcdf4', '1.5.3'],
                     ['xarray', '0.15.0'],
                     ['tables', '3.6.1'],
                     ['numexpr', '2.6.9'],
                     ['bottleneck', 'None'],
                     ['jinja2', '2.10.3'],
                     ['statsmodels', '0.11.0'],
                     ['python', '3.7.1'],
                     ['platform', 'A-Computer']]}


@pytest.fixture(scope='function')
def remove_orca():
    # otherwise generating all pdfs for tests can take ages
    import plotly.io as pio
    pio.orca.config.executable = '/dev/null'
