import pandas as pd
import pytest


from solarforecastarbiter.datamodel import Site


sites_dicts = [
    {
        'name': 'ARM site',
        'latitude': 1,
        'longitude': 1,
        'elevation': 5,
        'timezone': 'Etc/GMT+8',
        'extra_parameters': ('{"network": "ARM",'
                             '"observation_interval_length": 1}'),
    },
    {
        'name': 'NOAA SURFRAD site2',
        'latitude': 2,
        'longitude': 2,
        'elevation': 5,
        'timezone': 'Etc/GMT+8',
        'extra_parameters': ('{"network": "NOAA SURFRAD",'
                             '"observation_interval_length": 5}'),
    },
    {
        'name': 'NOAA SOLRAD site3',
        'latitude': 3,
        'longitude': -3,
        'elevation': 6,
        'timezone': 'Etc/GMT+8',
        'extra_parameters': ('{"network": "NOAA SOLRAD",'
                             '"observation_interval_length": 1}'),
    },
    {
        'name': 'site4',
        'latitude': 4,
        'longitude': -5,
        'elevation': 12,
        'timezone': 'Etc/GMT+8',
        'extra_parameters': ('{ "observation_interval_length": 1}'),
    }
]
site_objects = [Site.from_dict(site) for site in sites_dicts]


@pytest.fixture
def sites_dicts_param():
    return sites_dicts


@pytest.fixture
def site_objects_pram():
    return site_objects


@pytest.fixture
def mock_api(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mock_fetch(mocker):
    fetch = mocker.MagicMock()
    fetch.return_value = pd.DataFrame({})
    return fetch
