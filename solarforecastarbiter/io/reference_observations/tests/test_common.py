import pytest

from solarforecastarbiter.datamodel import Site, Observation
from solarforecastarbiter.io.reference_observations import common
from solarforecastarbiter.io.reference_observations.tests.conftest import (
    sites_dicts,
    site_objects)


@pytest.fixture(scope='module',
                params=site_objects)
def test_observations(site):
    return Observation.from_dict({
        'name': 'site ghi',
        'variable': 'ghi',
        'interval_label': 'beginning',
        'interval_value_type': 'interval_mean',
        'interval_length': 1,
        'site': site_objects[1],
        'uncertainty': 0,
        'extra_parameters': "{}"
    })

invalid_params = {
    'name': 'site-invalid-jsonparams',
    'latitude': 3,
    'longitude': -3,
    'elevation': 6,
    'timezone': 'Etc/GMT+8',
    'extra_parameters': '{{ mon$stertruck',
}


def test_decode_extra_parameters():
    metadata = Site.from_dict(sites_dicts[0])
    params = common.decode_extra_parameters(metadata)
    assert params['network'] == 'ARM'
    assert params['observation_interval_length'] == 1


def test_decode_extra_parameters_error(mocker):
    log = mocker.patch('solarforecastarbiter.io.reference_observations.common.logger')
    ret = common.decode_extra_parameters(Site.from_dict(invalid_params))
    assert ret == None
    assert log.warning.called


@pytest.mark.parametrize('networks,site,expected', [
    (['ARM'], sites_dicts[0], True),
    ('ARM', sites_dicts[0], True),
    (['ARM', 'NREL MIDC'], sites_dicts[1], False),
    ('NREL MIDC', sites_dicts[3], False),
])
def test_check_network(networks, site, expected):
    metadata = Site.from_dict(site)
    assert common.check_network(networks, metadata) == expected


@pytest.mark.parametrize('networks,expected', [
    (['ARM'], site_objects[:1]),
    ('ARM', site_objects[:1]),
    (['NOAA SURFRAD', 'NOAA SOLRAD'], site_objects[1:3])
])
def test_filter_by_network( networks, expected):
    assert common.filter_by_networks(site_objects, networks) == expected


site_test_observation = Observation.from_dict({
    'name': 'site ghi',
    'variable': 'ghi',
    'interval_label': 'ending',
    'interval_value_type': 'interval_mean',
    'interval_length': 1,
    'site': site_objects[0],
    'uncertainty': 0,
    'extra_parameters': '{"network": "ARM", "observation_interval_length": 1}'
})

@pytest.mark.parametrize('site,variable', [
    (site_objects[0], 'ghi'),
])
def test_create_observation(mock_api, site, variable):
    common.create_observation(mock_api, site, variable)
    mock_api.create_observation.assert_called()
    mock_api.create_observation.assert_called_with(site_test_observation)

long_site_name = Site.from_dict({
    'name': 'ARM site with just abouts sixty four characters in its name oops',
    'latitude': 1,
    'longitude': 1,
    'elevation': 5,
    'timezone': 'Etc/GMT+8',
    'extra_parameters': ('{"network": "ARM", "network_api_abbreviation": '
                         '"site_abbrev", "observation_interval_length": 1}'),
})


observation_long_site_name = Observation.from_dict({
    'name': 'site_abbrev air_temperature',
    'variable': 'air_temperature',
    'interval_label': 'ending',
    'interval_value_type': 'interval_mean',
    'interval_length': 1,
    'site': long_site_name,
    'uncertainty': 0,
    'extra_parameters': ('{"network": "ARM", "network_api_abbreviation": '
                         '"site_abbrev", "observation_interval_length": 1}'),
})
@pytest.mark.parametrize('site,variable,expected', [
    (long_site_name, 'air_temperature', observation_long_site_name),
])
def test_create_observation_long_site(mock_api, site, variable, expected):
    common.create_observation(mock_api, site, variable)
    mock_api.create_observation.assert_called()
    mock_api.create_observation.assert_called_with(expected)


observation_with_extra_params = Observation.from_dict({
    'name': 'site ghi',
    'variable': 'ghi',
    'interval_label': 'ending',
    'interval_value_type': 'interval_mean',
    'interval_length': 5,
    'site': site_objects[0],
    'uncertainty': 0,
    'extra_parameters': '{"network": "", "observation_interval_length": 5}'
})
observation_params = {
    'network': '',
    'observation_interval_length': 5,
}
@pytest.mark.parametrize('site,variable,expected,extra_params', [
    (site_objects[0], 'ghi', observation_with_extra_params, observation_params),
])
def test_create_observation_extra_parameters(
        mock_api, site, variable, expected, extra_params):
    common.create_observation(mock_api, site, variable, extra_params)
    mock_api.create_observation.assert_called()
    mock_api.create_observation.assert_called_with(expected)


test_kwarg_observation = Observation.from_dict({
    'name': 'just observation',
    'variable': 'ghi',
    'interval_label': 'beginning',
    'interval_value_type': 'instantaneous',
    'interval_length': 1,
    'site': site_objects[0],
    'uncertainty': 2,
    'extra_parameters': '{"network": "ARM", "observation_interval_length": 1}'
})
obs_kwargs = {
    'interval_label': 'beginning',
    'name': 'just observation',
    'interval_value_type': 'instantaneous',
    'uncertainty': 2,
}
@pytest.mark.parametrize('site,variable,expected,kwargs', [
    (site_objects[0], 'ghi', test_kwarg_observation, obs_kwargs),
])
def test_create_observation_with_kwargs(mock_api, site, variable, expected, kwargs):
    common.create_observation(mock_api, site, variable, **kwargs)
    mock_api.create_observation.assert_called()
    mock_api.create_observation.assert_called_with(expected)
