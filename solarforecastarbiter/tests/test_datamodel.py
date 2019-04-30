from dataclasses import fields, MISSING
import json


import pandas as pd
import pytest


from solarforecastarbiter import datamodel


@pytest.fixture(params=['site', 'fixed', 'single', 'observation',
                        'forecast'])
def pdid_params(request, many_sites, many_sites_text, single_observation,
                single_observation_text, single_site,
                single_forecast_text, single_forecast):
    if request.param == 'site':
        return (many_sites[0], json.loads(many_sites_text)[0],
                datamodel.Site)
    elif request.param == 'fixed':
        return (many_sites[1].modeling_parameters,
                json.loads(many_sites_text)[1]['modeling_parameters'],
                datamodel.FixedTiltModelingParameters)
    elif request.param == 'single':
        return (many_sites[2].modeling_parameters,
                json.loads(many_sites_text)[2]['modeling_parameters'],
                datamodel.SingleAxisModelingParameters)
    elif request.param == 'observation':
        obs_dict = json.loads(single_observation_text)
        obs_dict['site'] = single_site
        return (single_observation, obs_dict,
                datamodel.Observation)
    elif request.param == 'forecast':
        fx_dict = json.loads(single_forecast_text)
        fx_dict['site'] = single_site
        return (single_forecast, fx_dict, datamodel.Forecast)


@pytest.mark.parametrize('extra', [
    {},
    {'extra': 'thing'},
])
def test_from_dict_into_datamodel(extra, pdid_params):
    expected, obj_dict, model = pdid_params
    obj_dict.update(extra)
    out = model.from_dict(obj_dict)
    assert out == expected


def test_from_dict_into_datamodel_missing_field(pdid_params):
    _, obj_dict, model = pdid_params
    for field in fields(model):
        if field.default is MISSING and field.default_factory is MISSING:
            break
    del obj_dict[field.name]
    with pytest.raises(KeyError):
        model.from_dict(obj_dict)


def test_from_dict_into_datamodel_no_extra(pdid_params):
    expected, obj_dict, model = pdid_params
    obj_dict.pop('extra_parameters', '')
    out = model.from_dict(obj_dict)
    for field in fields(model):
        if field.name == 'extra_parameters':
            continue
        assert getattr(out, field.name) == getattr(expected, field.name)


def test_from_dict_no_extra(pdid_params):
    expected, obj_dict, model = pdid_params
    names = [f.name for f in fields(model)]
    for key in list(obj_dict.keys()):
        if key not in names:
            del obj_dict[key]
    assert model.from_dict(obj_dict, raise_on_extra=True) == expected

def test_from_dict_extra_params_raise(pdid_params):
    _, obj_dict, model = pdid_params
    obj_dict['superextra'] = 'thing'
    with pytest.raises(KeyError):
        model.from_dict(obj_dict, raise_on_extra=True)


def test_invalid_variable(single_site):
    with pytest.raises(ValueError):
        datamodel.Observation(
            name='test', variable='noway',
            interval_value_type='mean',
            interval_length=pd.Timedelta('1min'),
            interval_label='beginning',
            site=single_site,
            uncertainty=0.1,
        )


@pytest.fixture(params=[0, 1, 2])
def _sites(many_sites_text, many_sites, request):
    models = [datamodel.Site, datamodel.SolarPowerPlant,
              datamodel.SolarPowerPlant]
    param = request.param
    site_dict_list = json.loads(many_sites_text)
    return site_dict_list[param], many_sites[param], models[param]


def test_process_site_dict(_sites):
    site_dict, expected, model= _sites
    out = model.from_dict(site_dict)
    assert out == expected
