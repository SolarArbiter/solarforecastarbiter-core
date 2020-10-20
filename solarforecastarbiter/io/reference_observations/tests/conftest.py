import copy
import inspect
import json
import os


import numpy as np
import pandas as pd
import pytest


from solarforecastarbiter.datamodel import Site, Observation


TEST_DATA_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


def site_dicts():
    return [copy.deepcopy(site) for site in [
        {
            'name': 'site',
            'latitude': 1,
            'longitude': 1,
            'elevation': 5,
            'timezone': 'Etc/GMT+8',
            'extra_parameters': {
                "network": "DOE ARM",
                "network_api_id": 'B13',
                "network_api_abbreviation": 'sgp',
                "observation_interval_length": 1,
                "datastreams": {
                    'qcrad': 'sgpqcradlong1E13.c1',
                    'met': {
                        'sgpmetE13.b1': '2019-01-01/2020-01-01',  # noqa
                        'sgpmetE13.b2': '2020-01-01/2021-01-01',  # noqa
                    },
                },
            },
        },
        {
            'name': 'site2',
            'latitude': 2,
            'longitude': 2,
            'elevation': 5,
            'timezone': 'Etc/GMT+8',
            'extra_parameters': {"network": "NOAA SURFRAD",
                                 "network_api_id": 'some_id',
                                 "network_api_abbreviation": 'abbrv',
                                 "observation_interval_length": 5},
        },
        {
            'name': 'site3',
            'latitude': 3,
            'longitude': -3,
            'elevation': 6,
            'timezone': 'Etc/GMT+8',
            'extra_parameters': {"network": "NOAA SOLRAD",
                                 "network_api_id": 'some_id',
                                 "network_api_abbreviation": 'abbrv',
                                 "observation_interval_length": 1},
        },
        {
            'name': 'site4',
            'latitude': 4,
            'longitude': -5,
            'elevation': 12,
            'timezone': 'Etc/GMT+8',
            'extra_parameters': {"observation_interval_length": 1,
                                 "network": 'NREL MIDC',
                                 "network_api_id": 'BMS',
                                 "network_api_abbreviation": 'abbrv'},
        },
        {
            'name': 'site4',
            'latitude': 4,
            'longitude': -5,
            'elevation': 12,
            'timezone': 'Etc/GMT+8',
            'extra_parameters': {"observation_interval_length": 1,
                                 "network": 'Unincorporated',
                                 "network_api_id": 'BMS',
                                 "network_api_abbreviation": 'abbrv'},
        }
    ]]


def expected_site(site):
    new_site = site.copy()
    network = site['extra_parameters'].get('network', '')
    new_site['name'] = f"{network} {site['name']}"
    new_site.update({'extra_parameters': json.dumps(site['extra_parameters'])})
    return new_site


site_string_dicts = [expected_site(site) for site in site_dicts()]
site_objects = [Site.from_dict(site) for site in site_string_dicts]


@pytest.fixture
def site_dicts_param():
    return site_string_dicts


@pytest.fixture
def site_objects_param():
    return site_objects


def site_to_obs(site):
    ep = json.loads(site.extra_parameters)
    interval_length = ep['observation_interval_length']
    return Observation.from_dict({
        'name': 'site ghi',
        'variable': 'ghi',
        'interval_label': 'ending',
        'interval_value_type': 'interval_mean',
        'interval_length': interval_length,
        'site': site,
        'uncertainty': None,
        'extra_parameters': site.extra_parameters
    })


@pytest.fixture
def observation_objects_param(site_objects_param):
    return [site_to_obs(site) for site in site_objects_param]


@pytest.fixture
def networks():
    return ['DOE ARM', 'NOAA SURFRAD', 'NOAA SOLRAD', 'Unincorporated']


@pytest.fixture
def mock_api(mocker, site_objects_param, observation_objects_param):
    api = mocker.MagicMock()
    api.list_sites.return_value = site_objects_param
    api.list_observations.return_value = observation_objects_param
    return api


index = pd.date_range('20190101T1200Z', '20190101T1229Z',
                      freq='min', tz='UTC')
values = np.arange(100, 130)


@pytest.fixture
def start():
    return pd.Timestamp('20190101T1200Z')


@pytest.fixture
def end():
    return pd.Timestamp('20190101T1229Z')


@pytest.fixture
def fake_ghi_data():
    df = pd.DataFrame(index=index, data={'ghi': values})
    df['quality_flag'] = 0
    return df


@pytest.fixture
def mock_fetch(mocker, fake_ghi_data):
    fetch = mocker.MagicMock()
    fetch.return_value = fake_ghi_data
    return fetch


@pytest.fixture
def test_json_site_file():
    return os.path.join(TEST_DATA_DIR, 'data', 'test_site.json')
