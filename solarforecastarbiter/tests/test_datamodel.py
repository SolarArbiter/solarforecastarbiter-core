from dataclasses import asdict
import datetime as dt


import pytest


from solarforecastarbiter import datamodel


@pytest.mark.parametrize('extra', [
    {},
    {'extra': 'thing'},
])
@pytest.mark.parametrize('model', [
    datamodel.Site, datamodel.Observation, datamodel.Forecast
])
def test_process_dict_into_datamodel_site(extra, model, site_metadata):
    dict_ = asdict(site_metadata)
    dict_.update(extra)
    out = datamodel.process_dict_into_datamodel(dict_, datamodel.Site)
    assert out == site_metadata

# and modeling params, and obs, and forecasts, and pp


@pytest.mark.parametrize('site_dict', [
    {
        "elevation": 786.0,
        "extra_parameters": '{"network": "NREL MIDC"}',
        "latitude": 32.22969,
        "longitude": -110.95534,
        "modeling_parameters": {
            "ac_capacity": None,
            "ac_loss_factor": None,
            "axis_azimuth": None,
            "axis_tilt": None,
            "backtrack": None,
            "dc_capacity": None,
            "dc_loss_factor": None,
            "ground_coverage_ratio": None,
            "max_rotation_angle": None,
            "surface_azimuth": None,
            "surface_tilt": None,
            "temperature_coefficient": None,
            "tracking_type": None
        },
        "name": "Weather Station 1",
        "provider": "Organization 1",
        "timezone": "America/Phoenix",
        "site_id": 'd2018f1d-82b1-422a-8ec4-4e8b3fe92a4a',
        "created_at": dt.datetime(2019, 3, 1, 11, 44, 44),
        "modified_at": dt.datetime(2019, 3, 1, 11, 44, 44)
    },
    {
        "elevation": 786.0,
        "extra_parameters": '{"network": "NREL MIDC"}',
        "latitude": 32.22969,
        "longitude": -110.95534,
        "modeling_parameters": {},
        "name": "no modeling params",
        "provider": "Organization 1",
        "timezone": "America/Phoenix",
        "site_id": 'd2018f1d-82b1-422a-8ec4-4e8b3fe92a4a',
        "created_at": dt.datetime(2019, 3, 1, 11, 44, 44),
        "modified_at": dt.datetime(2019, 3, 1, 11, 44, 44)
    },
    {
        "elevation": 786.0,
        "extra_parameters": "",
        "latitude": 43.73403,
        "longitude": -96.62328,
        "modeling_parameters": {
            "ac_capacity": 0.015,
            "ac_loss_factor": 0.0,
            "axis_azimuth": None,
            "axis_tilt": None,
            "backtrack": None,
            "dc_capacity": 0.015,
            "dc_loss_factor": 0.0,
            "ground_coverage_ratio": None,
            "max_rotation_angle": None,
            "surface_azimuth": 180.0,
            "surface_tilt": 45.0,
            "temperature_coefficient": -.002,
            "tracking_type": "fixed"
        },
        "name": "Fixed plant",
        "provider": "Organization 1",
        "timezone": "Etc/GMT+6",
        "site_id": '123e4567-e89b-12d3-a456-426655440002',
        "created_at": dt.datetime(2019, 3, 1, 11, 44, 46),
        "modified_at": dt.datetime(2019, 3, 1, 11, 44, 46)
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
            "backtrack": True,
            "dc_capacity": 0.015,
            "dc_loss_factor": 0.0,
            "ground_coverage_ratio": .233,
            "max_rotation_angle": 90.0,
            "surface_azimuth": None,
            "surface_tilt": None,
            "temperature_coefficient": -.002,
            "tracking_type": "single_axis"
        },
        "name": "Tracking plant",
        "provider": "Organization 1",
        "timezone": "Etc/GMT+6",
        "site_id": '123e4567-e89b-12d3-a456-426655440002',
        "created_at": dt.datetime(2019, 3, 1, 11, 44, 46),
        "modified_at": dt.datetime(2019, 3, 1, 11, 44, 46)
    }
])
def test_process_site_json(site_dict):
    out = datamodel.process_site_json(site_dict)
    assert isinstance(out, datamodel.Site)
    for param, val in site_dict.items():
        if param in ['modeling_parameters', 'provider', 'created_at',
                     'modified_at', 'site_id']:
            continue
        assert getattr(out, param) == val
    for param, val in site_dict['modeling_parameters'].items():
        if hasattr(out, 'modeling_parameters') and val is not None:
            assert getattr(out.modeling_parameters, param) == val
