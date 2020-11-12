import io
import pytest


import pandas as pd


from solarforecastarbiter.io.reference_observations import reference_data
from solarforecastarbiter.io.reference_observations.tests.conftest import (
    site_dicts,
    site_objects
)


@pytest.fixture
def log(mocker):
    logger = mocker.patch('solarforecastarbiter.io.reference_observations.'
                          'reference_data.logger')
    return logger


site_object_pairs = list(zip(site_dicts(), site_objects))[:-1]


def test_getapisession(mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.io.reference_observations.reference_data.APISession')  # NOQA
    reference_data.get_apisession('TEST', 'url')
    assert mocked.called_with('TEST', 'url')


@pytest.fixture()
def mock_creates(mocker):
    initobs = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.create_observation')  # NOQA
    initfx = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.create_forecasts')  # NOQA
    return initobs, initfx


@pytest.mark.parametrize('site_dict,expected', site_object_pairs)
def test_create_site(mock_api, site_dict, expected, mocker, mock_creates):
    mock_api.list_sites.return_value = {}
    mock_api.create_site.return_value = expected
    reference_data.create_site(mock_api, site_dict.copy())
    mock_api.create_site.assert_called_with(expected)


@pytest.mark.parametrize('site_dict,expected', site_object_pairs)
def test_create_site_exists(mock_api, site_dict, expected, mocker,
                            mock_creates):
    mock_api.create_site.return_value = expected
    reference_data.create_site(mock_api, site_dict.copy())
    mock_api.create_site.assert_not_called()


def test_update_reference_observations(
        mocker, mock_api, log, start, end, networks):
    api = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'reference_data.get_apisession')
    api.return_value = mock_api
    surfrad = mocker.patch(
        'solarforecastarbiter.io.reference_observations.reference_data.'
        'surfrad.update_observation_data')
    solrad = mocker.patch(
        'solarforecastarbiter.io.reference_observations.solrad.'
        'update_observation_data')
    arm = mocker.patch(
        'solarforecastarbiter.io.reference_observations.arm.'
        'update_observation_data')
    reference_data.update_reference_observations('TOKEN', start, end, networks)
    surfrad.assert_called()
    solrad.assert_called()
    arm.assert_called()
    assert log.info.call_count == 1


def test_update_reference_observations_gaps(mocker, mock_api, log, start, end,
                                            networks):
    api = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'reference_data.get_apisession')
    mocker.patch.dict('os.environ', {
        'DOE_ARM_API_KEY': 'key', 'DOE_ARM_USER_ID': 'id'})
    api.return_value = mock_api
    common_update = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common'
        '.update_site_observations')
    reference_data.update_reference_observations(
        'TOKEN', start, end, networks, gaps_only=True)
    assert common_update.call_count == 3
    assert common_update.call_args[1]['gaps_only'] is True


site_csv = """interval_length,name,latitude,longitude,elevation,network_api_id,network_api_abbreviation,timezone,attribution,network
1,Seattle UW WA,47.653999999999996,-122.309,70,94291.0,ST,Etc/GMT+8,,UO SRML
1,UO Solar Awning Eugene OR,44.05,-123.07,150,94255.0,AW,Etc/GMT+8,,UO SRML"""
csv_dicts = [
    {'name': 'Seattle UW WA',
     'latitude': 47.654,
     'longitude': -122.309,
     'elevation': 70,
     'timezone': 'Etc/GMT+8',
     'extra_parameters': {
         'network': 'UO SRML',
         'network_api_id': 94291.0,
         'network_api_abbreviation': 'ST',
         'attribution': '',
         'observation_interval_length': 1}
     },
    {'name': 'UO Solar Awning Eugene OR',
     'latitude': 44.05,
     'longitude': -123.07,
     'elevation': 150,
     'timezone': 'Etc/GMT+8',
     'extra_parameters': {
         'network': 'UO SRML',
         'network_api_id': 94255.0,
         'network_api_abbreviation': 'AW',
         'attribution': '',
         'observation_interval_length': 1}
     }
]


def test_site_df_to_dicts():
    site_df = pd.read_csv(io.StringIO(site_csv))
    dicts = reference_data.site_df_to_dicts(site_df)
    assert csv_dicts[0] in dicts
    assert csv_dicts[1] in dicts


def test_initialize_reference_metadata_objects(
        log, mocker, mock_api, site_objects_param, mock_creates):
    api = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'reference_data.get_apisession')
    mock_api.list_sites.return_value = {}
    api.return_value = mock_api
    reference_data.initialize_reference_metadata_objects('TOKEN', site_dicts())
    api.assert_called_once()
    assert mock_api.create_site.call_count == 4
    for site in site_objects_param[:-1]:
        mock_api.create_site.assert_any_call(site)
