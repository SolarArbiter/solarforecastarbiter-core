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


site_object_pairs = list(zip(site_dicts(), site_objects))[1:]


@pytest.mark.parametrize('site_dict,expected', site_object_pairs)
def test_create_site(mock_api, site_dict, expected, mocker):
    mock_api.create_site.return_value = expected
    reference_data.create_site(mock_api, site_dict)
    mock_api.create_site.assert_called_with(expected)


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
    reference_data.update_reference_observations('TOKEN', start, end, networks)
    mock_api.list_sites.assert_called_once()
    mock_api.list_observations.assert_called_once()
    surfrad.assert_called()
    solrad.assert_called()
    assert log.info.call_count == 1


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
         'observation_interval_length': 1}
     }
]


def test_site_df_to_dicts():
    site_df = pd.read_csv(io.StringIO(site_csv))
    dicts = reference_data.site_df_to_dicts(site_df)
    assert csv_dicts[0] in dicts
    assert csv_dicts[1] in dicts


def test_initialize_reference_metadata_objects(
        log, mocker, mock_api, site_objects_param):
    api = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'reference_data.get_apisession')
    api.return_value = mock_api
    reference_data.initialize_reference_metadata_objects('TOKEN', site_dicts())
    api.assert_called_once()
    assert mock_api.create_site.call_count == 3
    for site in site_objects_param[1:]:
        mock_api.create_site.assert_any_call(site)
