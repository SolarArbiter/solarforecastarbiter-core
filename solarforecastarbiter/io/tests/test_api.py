import json
import re


import pandas as pd
import pandas.testing as pdt
import pytest


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


def test_apisession_get_observation_values(requests_mock, observation_values,
                                           observation_values_text):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*/values')
    requests_mock.register_uri('GET', matcher, content=observation_values_text)
    out = session.get_observation_values(
        'obsid', pd.Timestamp('2019-01-01T12:00:00-0600'),
        pd.Timestamp('2019-01-01T12:25:00-0600'))
    pdt.assert_frame_equal(out, observation_values)


def test_apisession_get_forecast_values(requests_mock, forecast_values,
                                        forecast_values_text):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*/values')
    requests_mock.register_uri('GET', matcher, content=forecast_values_text)
    out = session.get_forecast_values(
        'fxid', pd.Timestamp('2019-01-01T06:00:00-0600'),
        pd.Timestamp('2019-01-01T11:00:00-0600'))
    pdt.assert_series_equal(out, forecast_values)


def test_apisession_post_observation_values(requests_mock, observation_values):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*/values')
    mocked = requests_mock.register_uri('POST', matcher)
    session.post_observation_values('obsid', observation_values)
    # observation_values_text has a different timestamp format
    assert mocked.request_history[0].text == '{"values":[{"timestamp":"2019-01-01T19:00:00Z","value":0.0,"quality_flag":0},{"timestamp":"2019-01-01T19:05:00Z","value":1.0,"quality_flag":0},{"timestamp":"2019-01-01T19:10:00Z","value":1.5,"quality_flag":0},{"timestamp":"2019-01-01T19:15:00Z","value":9.9,"quality_flag":1},{"timestamp":"2019-01-01T19:20:00Z","value":2.0,"quality_flag":0},{"timestamp":"2019-01-01T19:25:00Z","value":-999.0,"quality_flag":3}]}'  # NOQA


def test_apisession_post_forecast_values(requests_mock, forecast_values):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*/values')
    mocked = requests_mock.register_uri('POST', matcher)
    session.post_forecast_values('fxid', forecast_values)
    assert mocked.request_history[0].text == '{"values":[{"timestamp":"2019-01-01T13:00:00Z","value":0.0},{"timestamp":"2019-01-01T14:00:00Z","value":1.0},{"timestamp":"2019-01-01T15:00:00Z","value":2.0},{"timestamp":"2019-01-01T16:00:00Z","value":3.0},{"timestamp":"2019-01-01T17:00:00Z","value":4.0},{"timestamp":"2019-01-01T18:00:00Z","value":5.0}]}'  # NOQA
