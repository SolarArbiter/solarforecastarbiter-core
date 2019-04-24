import re


import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.io import api


def test_apisession_init(requests_mock):
    session = api.APISession('TOKEN')
    requests_mock.register_uri('GET', session.base_url)
    res = session.get('')
    assert res.request.headers['Authentication'] == 'Bearer TOKEN'


@pytest.mark.parametrize('method,endpoint,expected', [
    ('GET', '/', 'https://api.solarforecastarbiter.org/'),
    ('GET', '', 'https://api.solarforecastarbiter.org/'),
    ('GET', '/sites/', 'https://api.solarforecastarbiter.org/sites/'),
    ('GET', 'sites/', 'https://api.solarforecastarbiter.org/sites/'),
    ('POST', '/observations/obsid/values',
     'https://api.solarforecastarbiter.org/observations/obsid/values'),
])
def test_apisession_request(endpoint, method, expected, requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri(method, matcher)
    res = session.request(method, endpoint)
    assert res.request.url == expected


@pytest.fixture()
def mock_get_site(requests_mock, site_text):
    matcher = re.compile(f'https://api.solarforecastarbiter.org/sites/.*')
    requests_mock.register_uri('GET', matcher, content=site_text)


def test_apisession_get_site(mock_get_site, single_site):
    session = api.APISession('')
    site = session.get_site('')
    assert site == single_site


def test_apisession_list_sites(requests_mock, many_sites_text, many_sites):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri('GET', matcher, content=many_sites_text)
    site_list = session.list_sites()
    assert site_list == many_sites


def test_apisession_get_observation(requests_mock, single_observation,
                                    single_observation_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*')
    requests_mock.register_uri('GET', matcher, content=single_observation_text)
    obs = session.get_observation('')
    assert obs == single_observation


def test_apisession_list_observations(requests_mock, many_observations,
                                      many_observations_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*')
    requests_mock.register_uri('GET', matcher, content=many_observations_text)
    obs_list = session.list_observations()
    assert obs_list == many_observations


def test_apisession_get_forecast(requests_mock, single_forecast,
                                 single_forecast_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('GET', matcher, content=single_forecast_text)
    fx = session.get_forecast('')
    assert fx == single_forecast


def test_apisession_list_forecasts(requests_mock, many_forecasts,
                                      many_forecasts_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('GET', matcher, content=many_forecasts_text)
    fx_list = session.list_forecasts()
    assert fx_list == many_forecasts
