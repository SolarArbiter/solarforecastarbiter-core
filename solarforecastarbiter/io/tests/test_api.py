import json
from random import randint
import re


import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import requests


from solarforecastarbiter.io import api, utils
from solarforecastarbiter import datamodel


def test_request_cli_access_token_mocked(requests_mock):
    requests_mock.register_uri(
        'POST', 'https://solarforecastarbiter.auth0.com/oauth/token',
        content=b'{"access_token": "token"}')
    assert api.request_cli_access_token('test', 'pass') == 'token'


def test_request_cli_access_token_real():
    try:
        requests.get('https://solarforecastarbiter.auth0.com')
    except Exception:  # pragma: no cover
        return pytest.skip('Cannot connect to Auth0')
    else:
        assert api.request_cli_access_token('testing@solarforecastarbiter.org',
                                            'Thepassword123!') is not None


def test_apisession_init(requests_mock):
    session = api.APISession('TOKEN')
    requests_mock.register_uri('GET', session.base_url)
    res = session.get('')
    assert res.request.headers['Authorization'] == 'Bearer TOKEN'


def test_apisession_init_hidden(requests_mock):
    ht = utils.HiddenToken('TOKEN')
    session = api.APISession(ht)
    requests_mock.register_uri('GET', session.base_url)
    res = session.get('')
    assert res.request.headers['Authorization'] == 'Bearer TOKEN'


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
def mock_get_site(requests_mock, site_text, many_sites_text):
    def get_site_from_text(request, context):
        site_id = request.url.split('/')[-1]
        if site_id == '':
            return many_sites_text
        else:
            sites = json.loads(many_sites_text)
            for site in sites:
                if site['site_id'] == site_id:
                    return json.dumps(site).encode('utf-8')

    matcher = re.compile(f'https://api.solarforecastarbiter.org/sites/.*')
    requests_mock.register_uri('GET', matcher, content=get_site_from_text)


@pytest.fixture()
def mock_list_sites(mocker, many_sites):
    mocker.patch('solarforecastarbiter.io.api.APISession.list_sites',
                 return_value=many_sites)


def test_apisession_get_site(mock_get_site, get_site):
    session = api.APISession('')
    site = session.get_site('123e4567-e89b-12d3-a456-426655440002')
    assert site == get_site('123e4567-e89b-12d3-a456-426655440002')


def test_apisession_get_site_dne(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri('GET', matcher, status_code=404)
    with pytest.raises(requests.exceptions.HTTPError):
        session.get_site('123e4567-e89b-12d3-a456-426655440002')


def test_apisession_list_sites(requests_mock, many_sites_text, many_sites):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri('GET', matcher, content=many_sites_text)
    site_list = session.list_sites()
    assert site_list == many_sites


def test_apisession_list_sites_empty(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri('GET', matcher, content=b"[]")
    site_list = session.list_sites()
    assert site_list == []


def test_apisession_create_site(requests_mock, single_site, site_text):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/sites/.*')
    requests_mock.register_uri('POST', matcher,
                               text=single_site.site_id)
    requests_mock.register_uri('GET', matcher, content=site_text)
    site_dict = single_site.to_dict()
    del site_dict['site_id']
    del site_dict['provider']
    del site_dict['extra_parameters']
    ss = type(single_site).from_dict(site_dict)
    new_site = session.create_site(ss)
    assert new_site == single_site


def test_apisession_get_observation(requests_mock, single_observation,
                                    single_observation_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*')
    requests_mock.register_uri('GET', matcher, content=single_observation_text)
    obs = session.get_observation('')
    assert obs == single_observation


def test_apisession_list_observations(requests_mock, many_observations,
                                      many_observations_text, mock_list_sites):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*')
    requests_mock.register_uri('GET', matcher, content=many_observations_text)
    obs_list = session.list_observations()
    assert obs_list == many_observations


def test_apisession_list_observations_empty(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*')
    requests_mock.register_uri('GET', matcher, content=b"[]")
    obs_list = session.list_observations()
    assert obs_list == []


def test_apisession_create_observation(requests_mock, single_observation,
                                       single_observation_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*')
    requests_mock.register_uri('POST', matcher,
                               text=single_observation.observation_id)
    requests_mock.register_uri('GET', matcher, content=single_observation_text)
    observation_dict = single_observation.to_dict()
    del observation_dict['observation_id']
    del observation_dict['extra_parameters']
    ss = type(single_observation).from_dict(observation_dict)
    new_observation = session.create_observation(ss)
    assert new_observation == single_observation


def test_apisession_get_forecast(requests_mock, single_forecast,
                                 single_forecast_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('GET', matcher, content=single_forecast_text)
    fx = session.get_forecast('')
    assert fx == single_forecast


def test_apisession_list_forecasts(requests_mock, many_forecasts,
                                   many_forecasts_text, mock_list_sites):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('GET', matcher, content=many_forecasts_text)
    fx_list = session.list_forecasts()
    assert fx_list == many_forecasts


def test_apisession_create_forecast(requests_mock, single_forecast,
                                    single_forecast_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*')
    requests_mock.register_uri('POST', matcher,
                               text=single_forecast.forecast_id)
    requests_mock.register_uri('GET', matcher, content=single_forecast_text)
    forecast_dict = single_forecast.to_dict()
    del forecast_dict['forecast_id']
    del forecast_dict['extra_parameters']
    ss = type(single_forecast).from_dict(forecast_dict)
    new_forecast = session.create_forecast(ss)
    assert new_forecast == single_forecast


def test_apisession_list_forecasts_empty(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('GET', matcher, content=b'[]')
    fx_list = session.list_forecasts()
    assert fx_list == []


def test_apisession_get_observation_values(requests_mock, observation_values,
                                           observation_values_text):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*/values')
    requests_mock.register_uri('GET', matcher, content=observation_values_text)
    out = session.get_observation_values(
        'obsid', pd.Timestamp('2019-01-01T12:00:00-0600'),
        pd.Timestamp('2019-01-01T12:25:00-0600'))
    pdt.assert_frame_equal(out, observation_values)


@pytest.fixture()
def empty_df():
    return pd.DataFrame([], columns=['value', 'quality_flag'],
                        index=pd.DatetimeIndex([], name='timestamp'))


def test_apisession_get_observation_values_empty(requests_mock, empty_df):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*/values')
    requests_mock.register_uri('GET', matcher, content=b'{"values":[]}')
    out = session.get_observation_values(
        'obsid', pd.Timestamp('2019-01-01T12:00:00-0600'),
        pd.Timestamp('2019-01-01T12:25:00-0600'))
    pdt.assert_frame_equal(out, empty_df)


def test_apisession_get_forecast_values(requests_mock, forecast_values,
                                        forecast_values_text):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*/values')
    requests_mock.register_uri('GET', matcher, content=forecast_values_text)
    out = session.get_forecast_values(
        'fxid', pd.Timestamp('2019-01-01T06:00:00-0600'),
        pd.Timestamp('2019-01-01T11:00:00-0600'))
    pdt.assert_series_equal(out, forecast_values)


def test_apisession_get_forecast_values_empty(requests_mock, empty_df):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*/values')
    requests_mock.register_uri('GET', matcher, content=b'{"values":[]}')
    out = session.get_forecast_values(
        'fxid', pd.Timestamp('2019-01-01T06:00:00-0600'),
        pd.Timestamp('2019-01-01T11:00:00-0600'))
    pdt.assert_series_equal(out, empty_df['value'])


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


@pytest.fixture(scope='session')
def auth_token():
    try:
        token_req = requests.post(
            'https://solarforecastarbiter.auth0.com/oauth/token',
            headers={'content-type': 'application/json'},
            data=('{"grant_type": "password", '
                  '"username": "testing@solarforecastarbiter.org",'
                  '"password": "Thepassword123!", '
                  '"audience": "https://api.solarforecastarbiter.org", '
                  '"client_id": "c16EJo48lbTCQEhqSztGGlmxxxmZ4zX7"}'))
    except Exception:
        return pytest.skip('Cannot retrieve valid Auth0 token')
    else:
        token = token_req.json()['access_token']
        return token


@pytest.fixture(scope='session')
def real_session(auth_token):
    session = api.APISession(
        auth_token, base_url='https://dev-api.solarforecastarbiter.org')
    try:
        session.get('')
    except Exception:
        return pytest.skip('Cannot connect to dev api')
    else:
        return session


def test_real_apisession_get_site(real_session):
    site = real_session.get_site('123e4567-e89b-12d3-a456-426655440001')
    assert isinstance(site, datamodel.Site)


def test_real_apisession_list_sites(real_session):
    sites = real_session.list_sites()
    assert isinstance(sites, list)
    assert isinstance(sites[0], datamodel.Site)


def test_real_apisession_create_site(site_text, real_session):
    sited = json.loads(site_text)
    sited['name'] = f'Test create site {randint(0, 100)}'
    site = datamodel.SolarPowerPlant.from_dict(sited)
    new_site = real_session.create_site(site)
    real_session.delete(f'/sites/{new_site.site_id}')
    assert new_site.name == site.name
    assert new_site.modeling_parameters == site.modeling_parameters


def test_real_apisession_get_observation(real_session):
    obs = real_session.get_observation('123e4567-e89b-12d3-a456-426655440000')
    assert isinstance(obs, datamodel.Observation)


def test_real_apisession_list_observations(real_session):
    obs = real_session.list_observations()
    assert isinstance(obs, list)
    assert isinstance(obs[0], datamodel.Observation)


def test_real_apisession_create_observation(single_observation_text,
                                            single_observation,
                                            real_session):
    observationd = json.loads(single_observation_text)
    observationd['name'] = f'Test create observation {randint(0, 100)}'
    observationd['site'] = single_observation.site
    observation = datamodel.Observation.from_dict(observationd)
    new_observation = real_session.create_observation(observation)
    real_session.delete(f'/observations/{new_observation.observation_id}')
    for attr in ('name', 'site', 'variable', 'interval_label',
                 'interval_length', 'extra_parameters', 'uncertainty'):
        assert getattr(new_observation, attr) == getattr(observation, attr)


def test_real_apisession_get_forecast(real_session):
    fx = real_session.get_forecast('f8dd49fa-23e2-48a0-862b-ba0af6dec276')
    assert isinstance(fx, datamodel.Forecast)


def test_real_apisession_list_forecasts(real_session):
    fxs = real_session.list_forecasts()
    assert isinstance(fxs, list)
    assert isinstance(fxs[0], datamodel.Forecast)


def test_real_apisession_create_forecast(single_forecast_text, single_forecast,
                                         real_session):
    forecastd = json.loads(single_forecast_text)
    forecastd['name'] = f'Test create forecast {randint(0, 100)}'
    forecastd['site'] = single_forecast.site
    forecast = datamodel.Forecast.from_dict(forecastd)
    new_forecast = real_session.create_forecast(forecast)
    real_session.delete(f'/forecasts/single/{new_forecast.forecast_id}')
    for attr in ('name', 'site', 'variable', 'interval_label',
                 'interval_length', 'lead_time_to_start', 'run_length',
                 'extra_parameters'):
        assert getattr(new_forecast, attr) == getattr(forecast, attr)


def test_real_apisession_get_observation_values(real_session):
    obs = real_session.get_observation_values(
        '123e4567-e89b-12d3-a456-426655440000',
        pd.Timestamp('2019-04-15T00:00:00Z'),
        pd.Timestamp('2019-04-15T12:00:00Z'))
    assert isinstance(obs, pd.DataFrame)
    assert set(obs.columns) == set(['value', 'quality_flag'])


def test_real_apisession_get_forecast_values(real_session):
    fx = real_session.get_forecast_values(
        'f8dd49fa-23e2-48a0-862b-ba0af6dec276',
        pd.Timestamp('2019-04-15T00:00:00Z'),
        pd.Timestamp('2019-04-15T12:00:00Z'))
    assert isinstance(fx, pd.Series)


def test_real_apisession_post_observation_values(real_session):
    test_df = pd.DataFrame(
        {'value': [np.random.random()], 'quality_flag': [0]},
        index=pd.DatetimeIndex([pd.Timestamp('2019-04-14T00:00:00Z')],
                               name='timestamp'))
    real_session.post_observation_values(
        '123e4567-e89b-12d3-a456-426655440000', test_df)
    obs = real_session.get_observation_values(
        '123e4567-e89b-12d3-a456-426655440000',
        pd.Timestamp('2019-04-14T00:00:00Z'),
        pd.Timestamp('2019-04-14T00:01:00Z'))
    # quality flag may be altered by validation routine
    pdt.assert_series_equal(obs['value'], test_df['value'])


def test_real_apisession_post_forecast_values(real_session):
    test_ser = pd.Series(
        [np.random.random()], name='value',
        index=pd.DatetimeIndex([pd.Timestamp('2019-04-14T00:00:00Z')],
                               name='timestamp'))
    real_session.post_forecast_values(
        'f8dd49fa-23e2-48a0-862b-ba0af6dec276', test_ser)
    fx = real_session.get_forecast_values(
        'f8dd49fa-23e2-48a0-862b-ba0af6dec276',
        pd.Timestamp('2019-04-14T00:00:00Z'),
        pd.Timestamp('2019-04-14T00:01:00Z'))
    pdt.assert_series_equal(fx, test_ser)
