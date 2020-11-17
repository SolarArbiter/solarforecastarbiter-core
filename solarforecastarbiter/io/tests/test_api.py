import json
from random import randint
import re
import copy
from urllib.parse import parse_qs


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


def test_request_cli_access_token_mocked_kwargs(requests_mock):
    m = requests_mock.register_uri(
        'POST', 'https://solarforecastarbiter.auth0.com/oauth/token',
        content=b'{"access_token": "token"}')
    token = api.request_cli_access_token(
        'test', 'pass', cert="some_certificate.crt")
    assert m.request_history[0].cert == "some_certificate.crt"
    assert token == 'token'


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

    matcher = re.compile('https://api.solarforecastarbiter.org/sites/.*')
    requests_mock.register_uri('GET', matcher, content=get_site_from_text)


@pytest.fixture()
def mock_get_observation(requests_mock, many_observations_text,
                         mock_get_site):
    def get_observation_from_text(request, context):
        obs_id = request.url.split('/')[-2]
        if obs_id == '':  # pragma: no cover
            return many_observations_text
        else:
            obs = json.loads(many_observations_text)
            for ob in obs:
                if ob['observation_id'] == obs_id:
                    return json.dumps(ob).encode('utf-8')

    matcher = re.compile(
        'https://api.solarforecastarbiter.org/observations/.*/metadata')
    requests_mock.register_uri(
        'GET', matcher, content=get_observation_from_text)


@pytest.fixture()
def mock_get_agg(requests_mock, aggregate_text, mock_get_observation):
    matcher = re.compile('https://api.solarforecastarbiter.org/aggregates/.*')
    requests_mock.register_uri('GET', matcher, content=aggregate_text)


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


def test_apisession_list_sites_in_zone(requests_mock, many_sites_text,
                                       many_sites):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri('GET', matcher, content=json.dumps(
        json.loads(many_sites_text)[1:]).encode())
    site_list = session.list_sites_in_zone('Reference Region 5')
    assert site_list == many_sites[1:]


def test_apisession_list_sites_empty(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri('GET', matcher, content=b"[]")
    site_list = session.list_sites()
    assert site_list == []


def test_apisession_list_sites_in_zone_empty(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri('GET', matcher, content=b"[]")
    site_list = session.list_sites_in_zone('bad zone')
    assert site_list == []


def test_apisession_search_climatezones(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/.*')
    requests_mock.register_uri(
        'GET', matcher,
        content=b'[{"name": "Reference Region 3", "created_at": 0}]')
    zone_list = session.search_climatezones(32.1, -110.8)
    assert zone_list == ["Reference Region 3"]


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


def test_apisession_list_observation_single(requests_mock, single_observation,
                                            single_observation_text,
                                            mock_list_sites):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*')
    requests_mock.register_uri('GET', matcher, content=single_observation_text)
    obs_list = session.list_observations()
    assert obs_list == [single_observation]


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
                                   many_forecasts_text, mock_get_site,
                                   mock_get_agg):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('GET', matcher, content=many_forecasts_text)
    fx_list = session.list_forecasts()
    assert fx_list == many_forecasts


def test_apisession_list_forecast_single(requests_mock, single_forecast,
                                         single_forecast_text, mock_get_site,
                                         mock_get_agg):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('GET', matcher, content=single_forecast_text)
    fx_list = session.list_forecasts()
    assert fx_list == [single_forecast]


def test_apisession_create_forecast(requests_mock, single_forecast,
                                    single_forecast_text, mock_get_site,
                                    mock_get_agg):
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


def test_apisession_create_forecast_agg(
        requests_mock, aggregate, aggregateforecast,
        aggregate_forecast_text, mock_get_agg):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*')
    requests_mock.register_uri('POST', matcher,
                               text=aggregateforecast.forecast_id)
    requests_mock.register_uri('GET', matcher, content=aggregate_forecast_text)
    forecast_dict = aggregateforecast.to_dict()
    del forecast_dict['forecast_id']
    del forecast_dict['extra_parameters']
    ss = type(aggregateforecast).from_dict(forecast_dict)
    new_forecast = session.create_forecast(ss)
    assert new_forecast == aggregateforecast


def test_apisession_list_forecasts_empty(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('GET', matcher, content=b'[]')
    fx_list = session.list_forecasts()
    assert fx_list == []


def test_apisession_get_prob_forecast(requests_mock, prob_forecasts,
                                      prob_forecast_text, mock_get_site,
                                      prob_forecast_constant_value_text):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_constant_value_text)
    matcher = re.compile(session.base_url + r'/forecasts/cdf/[\w-]*$')
    requests_mock.register_uri('GET', matcher, content=prob_forecast_text)
    fx = session.get_probabilistic_forecast('')
    assert fx == prob_forecasts


def test_apisession_list_prob_forecasts(requests_mock, many_prob_forecasts,
                                        many_prob_forecasts_text,
                                        mock_list_sites, mock_get_site,
                                        prob_forecast_constant_value_text,
                                        mock_get_agg):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_constant_value_text)
    matcher = re.compile(session.base_url + r'/forecasts/cdf/$')
    requests_mock.register_uri(
        'GET', matcher, content=many_prob_forecasts_text)
    fx_list = session.list_probabilistic_forecasts()
    assert fx_list == many_prob_forecasts


def test_apisession_list_prob_forecast_single(
        requests_mock, prob_forecasts, prob_forecast_text, mock_list_sites,
        mock_get_site, prob_forecast_constant_value_text, mock_get_agg):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_constant_value_text)
    matcher = re.compile(session.base_url + r'/forecasts/cdf/$')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_text)
    fx_list = session.list_probabilistic_forecasts()
    assert fx_list == [prob_forecasts]


def test_apisession_list_prob_forecasts_empty(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/$')
    requests_mock.register_uri('GET', matcher, content=b'[]')
    fx_list = session.list_probabilistic_forecasts()
    assert fx_list == []


def test_apisession_get_prob_forecast_constant_value(
        requests_mock, prob_forecast_constant_value,
        prob_forecast_constant_value_text, mock_get_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_constant_value_text)
    fx = session.get_probabilistic_forecast_constant_value('')
    assert fx == prob_forecast_constant_value


def test_apisession_get_prob_forecast_constant_value_site(
        requests_mock, prob_forecast_constant_value,
        prob_forecast_constant_value_text, single_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_constant_value_text)
    fx = session.get_probabilistic_forecast_constant_value(
        '', site=single_site)
    assert fx == prob_forecast_constant_value


def test_apisession_get_agg_prob_forecast_constant_value(
        requests_mock, agg_prob_forecast_constant_value,
        agg_prob_forecast_constant_value_text, mock_get_agg,
        aggregate):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=agg_prob_forecast_constant_value_text)
    fx = session.get_probabilistic_forecast_constant_value('')
    assert fx == agg_prob_forecast_constant_value


def test_apisession_get_agg_prob_forecast_constant_value_agg(
        requests_mock, agg_prob_forecast_constant_value,
        agg_prob_forecast_constant_value_text, mock_get_agg,
        aggregate):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=agg_prob_forecast_constant_value_text)
    fx = session.get_probabilistic_forecast_constant_value(
        '', aggregate=aggregate)
    assert fx == agg_prob_forecast_constant_value


def test_apisession_get_prob_forecast_constant_value_site_error(
        requests_mock, prob_forecast_constant_value_text, single_site):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_constant_value_text)
    site_dict = single_site.to_dict()
    site_dict['site_id'] = 'nope'
    site = datamodel.Site.from_dict(site_dict)
    with pytest.raises(ValueError):
        session.get_probabilistic_forecast_constant_value('', site=site)


def test_apisession_get_prob_forecast_constant_value_agg_error(
        requests_mock, agg_prob_forecast_constant_value_text, aggregate):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=agg_prob_forecast_constant_value_text)
    agg_dict = aggregate.to_dict()
    agg_dict['aggregate_id'] = 'nope'
    agg = datamodel.Aggregate.from_dict(agg_dict)
    with pytest.raises(ValueError):
        session.get_probabilistic_forecast_constant_value(
            '', aggregate=agg)


def test_apisession_create_prob_forecast(requests_mock, prob_forecasts,
                                         prob_forecast_text, mock_get_site,
                                         prob_forecast_constant_value_text):
    session = api.APISession('')
    matcher = re.compile(session.base_url + r'/forecasts/cdf/$')
    requests_mock.register_uri('POST', matcher,
                               text=prob_forecasts.forecast_id)
    matcher = re.compile(
        f'{session.base_url}/forecasts/cdf/{prob_forecasts.forecast_id}$')
    requests_mock.register_uri('GET', matcher, content=prob_forecast_text)
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_constant_value_text)
    forecast_dict = prob_forecasts.to_dict()
    del forecast_dict['forecast_id']
    del forecast_dict['extra_parameters']
    ss = type(prob_forecasts).from_dict(forecast_dict)
    new_forecast = session.create_probabilistic_forecast(ss)
    assert new_forecast == prob_forecasts


def test_apisession_create_prob_forecast_agg(
        requests_mock, aggregate_prob_forecast,
        aggregate_prob_forecast_text, mock_get_agg,
        agg_prob_forecast_constant_value_text):
    session = api.APISession('')
    matcher = re.compile(session.base_url + r'/forecasts/cdf/$')
    requests_mock.register_uri('POST', matcher,
                               text=aggregate_prob_forecast.forecast_id)
    matcher = re.compile(
        f'{session.base_url}/forecasts/cdf/{aggregate_prob_forecast.forecast_id}$')  # NOQA
    requests_mock.register_uri(
        'GET', matcher, content=aggregate_prob_forecast_text)
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=agg_prob_forecast_constant_value_text)
    forecast_dict = aggregate_prob_forecast.to_dict()
    del forecast_dict['forecast_id']
    del forecast_dict['extra_parameters']
    ss = type(aggregate_prob_forecast).from_dict(forecast_dict)
    new_forecast = session.create_probabilistic_forecast(ss)
    assert new_forecast == aggregate_prob_forecast


@pytest.fixture(params=[0, 1])
def obs_start_end(request):
    if request.param == 0:
        return (pd.Timestamp('2019-01-01T12:00:00-0700'),
                pd.Timestamp('2019-01-01T12:25:00-0700'))
    else:
        return ('2019-01-01T12:00:00-0700',
                '2019-01-01T12:25:00-0700')


@pytest.mark.parametrize('func', ['get_observation_values',
                                  'get_aggregate_values'])
def test_apisession_get_obs_agg_values(
        requests_mock, observation_values, observation_values_text,
        obs_start_end, func):
    session = api.APISession('')
    matcher = re.compile(
        f'{session.base_url}/(observations|aggregates)/.*/values')
    requests_mock.register_uri('GET', matcher, content=observation_values_text)
    out = getattr(session, func)('obsid', *obs_start_end)
    pdt.assert_frame_equal(out, observation_values)


@pytest.mark.parametrize('label,theslice', [
    (None, slice(0, 10)),
    ('beginning', slice(0, -1)),
    ('ending', slice(1, 10))
])
@pytest.mark.parametrize('func', ['get_observation_values',
                                  'get_aggregate_values'])
def test_apisession_get_obs_agg_values_interval_label(
        requests_mock, observation_values, observation_values_text,
        label, theslice, obs_start_end, func):
    session = api.APISession('')
    matcher = re.compile(
        f'{session.base_url}/(observations|aggregates)/.*/values')
    requests_mock.register_uri('GET', matcher, content=observation_values_text)
    out = getattr(session, func)(
        'obsid', obs_start_end[0], obs_start_end[1], label)
    pdt.assert_frame_equal(out, observation_values.iloc[theslice])


@pytest.fixture()
def empty_df():
    return pd.DataFrame(
        [], columns=['value', 'quality_flag'],
        index=pd.DatetimeIndex([], name='timestamp', tz='UTC')
    ).astype({'value': float, 'quality_flag': int})


@pytest.mark.parametrize('func', ['get_observation_values',
                                  'get_aggregate_values'])
def test_apisession_get_obs_agg_values_empty(requests_mock, empty_df,
                                             func):
    session = api.APISession('')
    matcher = re.compile(
        f'{session.base_url}/(observations|aggregates)/.*/values')
    requests_mock.register_uri('GET', matcher, content=b'{"values":[]}')
    out = getattr(session, func)(
        'obsid', pd.Timestamp('2019-01-01T12:00:00-0700'),
        pd.Timestamp('2019-01-01T12:25:00-0700'))
    pdt.assert_frame_equal(out, empty_df)


@pytest.fixture(params=[0, 1])
def fx_start_end(request):
    if request.param == 0:
        return (pd.Timestamp('2019-01-01T06:00:00-0700'),
                pd.Timestamp('2019-01-01T11:00:00-0700'))
    else:
        return ('2019-01-01T06:00:00-0700',
                '2019-01-01T11:00:00-0700')


def test_apisession_get_forecast_values(requests_mock, forecast_values,
                                        forecast_values_text, fx_start_end):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*/values')
    requests_mock.register_uri('GET', matcher, content=forecast_values_text)
    out = session.get_forecast_values(
        'fxid', *fx_start_end)
    pdt.assert_series_equal(out, forecast_values)


def test_apisession_get_prob_forecast_constant_value_values(
        requests_mock, forecast_values, forecast_values_text, fx_start_end):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*/values')
    requests_mock.register_uri('GET', matcher, content=forecast_values_text)
    out = session.get_probabilistic_forecast_constant_value_values(
        'fxid', *fx_start_end)
    pdt.assert_series_equal(out, forecast_values)


@pytest.mark.parametrize('col_values', [
    ['25.0', '50.0', '75.0'], [25., 50., 75.]])
def test_apisession_get_prob_forecast_values(
        requests_mock, mock_get_site, col_values,
        prob_forecast_values, prob_forecast_values_text_list,
        prob_forecast_text, prob_forecast_constant_value_text, fx_start_end):
    session = api.APISession('')

    # modify ProbabilisticForecast json text
    probfx_json = json.loads(prob_forecast_text)
    probfx_json['forecast_id'] = 'FXID'
    probfx_json['axis'] = 'y'
    cvs_json = json.loads(prob_forecast_constant_value_text)
    cvs = []
    for col in col_values:
        new_cv = copy.deepcopy(cvs_json)
        new_cv['constant_value'] = col
        # fixture has keys like CV25
        # we need to convert either float 25. or str '25.0' to int 25
        # str(25.) == '25.0', int('25.0') raises ValueError, so we have
        # to chain it all together
        new_cv['forecast_id'] = 'CV' + str(int(float(col)))
        new_cv['axis'] = 'y'
        cvs.append(new_cv)
    probfx_json['constant_values'] = cvs

    matcher = re.compile(session.base_url + r'/forecasts/cdf/[\w-]*$')
    requests_mock.register_uri(
        'GET', matcher, content=json.dumps(probfx_json).encode('utf-8'))
    for cv in cvs:
        id = cv['forecast_id']
        cv = json.dumps(copy.deepcopy(cv))
        requests_mock.register_uri(
            'GET', f'/forecasts/cdf/single/{id}', content=cv.encode('utf-8'))
    for cvv in prob_forecast_values_text_list:
        id = json.loads(cvv)['forecast_id']
        requests_mock.register_uri(
            'GET', f'/forecasts/cdf/single/{id}/values', content=cvv)
    out = session.get_probabilistic_forecast_values('', *fx_start_end)
    pdt.assert_frame_equal(out, prob_forecast_values)


@pytest.mark.parametrize('label,theslice', [
    (None, slice(0, 10)),
    ('beginning', slice(0, -1)),
    ('ending', slice(1, 10))
])
def test_apisession_get_forecast_values_interval_label(
        requests_mock, forecast_values, forecast_values_text, label, theslice,
        fx_start_end):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*/values')
    requests_mock.register_uri('GET', matcher, content=forecast_values_text)
    out = session.get_forecast_values(
        'fxid', fx_start_end[0], fx_start_end[1], label)
    pdt.assert_series_equal(out, forecast_values.iloc[theslice])


def test_apisession_get_forecast_values_empty(requests_mock, empty_df):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/single/.*/values')
    requests_mock.register_uri('GET', matcher, content=b'{"values":[]}')
    out = session.get_forecast_values(
        'fxid', pd.Timestamp('2019-01-01T06:00:00-0700'),
        pd.Timestamp('2019-01-01T11:00:00-0700'))
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


def test_apisession_post_prob_forecast_constant_value_values(
        requests_mock, forecast_values):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*/values')
    mocked = requests_mock.register_uri('POST', matcher)
    session.post_probabilistic_forecast_constant_value_values(
        'fxid', forecast_values)
    assert mocked.request_history[0].text == '{"values":[{"timestamp":"2019-01-01T13:00:00Z","value":0.0},{"timestamp":"2019-01-01T14:00:00Z","value":1.0},{"timestamp":"2019-01-01T15:00:00Z","value":2.0},{"timestamp":"2019-01-01T16:00:00Z","value":3.0},{"timestamp":"2019-01-01T17:00:00Z","value":4.0},{"timestamp":"2019-01-01T18:00:00Z","value":5.0}]}'  # NOQA


@pytest.mark.parametrize('match, meth', [
    ('observations', 'get_observation_values'),
    ('forecasts/single', 'get_forecast_values'),
    ('forecasts/cdf', 'get_probabilistic_forecast_values'),
    ('forecasts/cdf/single',
     'get_probabilistic_forecast_constant_value_values'),
    ('aggregates', 'get_aggregate_values'),
])
def test_apisession_get_values(
        requests_mock, mocker, single_observation, single_forecast,
        prob_forecasts, prob_forecast_constant_value, aggregate, match, meth,
        fx_start_end):
    objs = {
        'observations': single_observation,
        'forecasts/single': single_forecast,
        'forecasts/cdf': prob_forecasts,
        'forecasts/cdf/single': prob_forecast_constant_value,
        'aggregates': aggregate
    }
    obj = objs[match]
    session = api.APISession('')
    matcher = re.compile(
        f'{session.base_url}/{match}/.*/values')
    requests_mock.register_uri('GET', matcher, content=b'{"values":[]}')
    status = mocker.patch(
        f'solarforecastarbiter.io.api.APISession.{meth}')
    session.get_values(obj, *fx_start_end)
    assert status.called


@pytest.fixture()
def mock_request_fxobs(report_objects, mocker):
    report, obs, fx0, fx1, agg, fxagg = report_objects
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_aggregate',
                 return_value=agg)

    def returnone(fxid):
        if fxid == "da2bc386-8712-11e9-a1c7-0a580a8200ae":
            return fx0
        elif fxid == "49220780-76ae-4b11-bef1-7a75bdc784e3":
            return fxagg
        elif fxid == "refbc386-8712-11e9-a1c7-0a580a8200ae":
            return report.report_parameters.object_pairs[1].reference_forecast
        else:
            return fx1
    mocker.patch('solarforecastarbiter.io.api.APISession.get_forecast',
                 side_effect=returnone)


def test_apisession_get_report(requests_mock, report_text, report_objects,
                               mock_request_fxobs):
    session = api.APISession('')
    requests_mock.register_uri('GET', f'{session.base_url}/reports/',
                               content=report_text)
    out = session.get_report('')
    expected = report_objects[0]
    assert out == expected


@pytest.fixture()
def mock_request_event_fxobs(event_report_objects, mocker):
    _, obs, fx0, fx1 = event_report_objects
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)

    def returnone(fxid):
        if fxid == "da2bc386-8712-11e9-a1c7-0a580a8200ae":
            return fx0
        else:
            return fx1
    mocker.patch('solarforecastarbiter.io.api.APISession.get_forecast',
                 side_effect=returnone)


def test_apisession_get_report_event(requests_mock, event_report_text,
                                     event_report_objects,
                                     mock_request_event_fxobs):
    session = api.APISession('')
    requests_mock.register_uri('GET', f'{session.base_url}/reports/',
                               content=event_report_text)
    out = session.get_report('')
    expected = event_report_objects[0]
    assert out == expected


def test_apisession_get_report_with_raw(
        requests_mock, report_text, report_objects, mock_request_fxobs,
        raw_report, mocker):
    raw = raw_report(False)
    raw_dict = raw.to_dict()
    report = json.loads(report_text)
    report['raw_report'] = raw_dict
    report_text = json.dumps(report).encode()
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_raw_report_processed_data',
        return_value=())
    session = api.APISession('')
    requests_mock.register_uri('GET', f'{session.base_url}/reports/',
                               content=report_text)
    out = session.get_report('')
    expected = report_objects[0].replace(
        raw_report=raw.replace(processed_forecasts_observations=()))
    assert out == expected


def test_apisession_list_reports(requests_mock, report_text, report_objects,
                                 mock_request_fxobs):
    session = api.APISession('')
    requests_mock.register_uri('GET', f'{session.base_url}/reports',
                               content=b'['+report_text+b']')
    out = session.list_reports()
    expected = [report_objects[0]]
    assert out == expected


def test_apisession_list_reports_single(requests_mock, report_text,
                                        report_objects, mock_request_fxobs):
    session = api.APISession('')
    requests_mock.register_uri('GET', f'{session.base_url}/reports',
                               content=report_text)
    out = session.list_reports()
    expected = [report_objects[0]]
    assert out == expected


def test_apisession_list_reports_empty(requests_mock):
    session = api.APISession('')
    requests_mock.register_uri('GET', f'{session.base_url}/reports',
                               content=b'[]')
    out = session.list_reports()
    assert out == []


def test_apisession_create_report(requests_mock, report_objects, mocker):
    session = api.APISession('')
    report = report_objects[0]
    mocked = requests_mock.register_uri('POST', f'{session.base_url}/reports/')
    mocker.patch('solarforecastarbiter.io.api.APISession.get_report',
                 return_value=report)
    expected = {
        "report_parameters": {
            "name": "NREL MIDC OASIS GHI Forecast Analysis",
            "start": "2019-04-01T00:00:00-07:00",
            "end": "2019-04-04T23:59:00-07:00",
            "forecast_fill_method": "forward",
            "filters": [
                {
                    'quality_flags': [
                        "USER FLAGGED",
                        "NIGHTTIME",
                        "LIMITS EXCEEDED",
                        "STALE VALUES",
                        "INTERPOLATED VALUES",
                        "INCONSISTENT IRRADIANCE COMPONENTS",
                    ],
                    'discard_before_resample': True,
                    'resample_threshold_percentage': 10.
                },
                {'time_of_day_range': ['12:00', '14:00']}],
            "metrics": ["mae", "rmse", "mbe", "s", "cost"],
            "categories": ["total", "date", "hour"],
            "object_pairs": [
                {"forecast": "da2bc386-8712-11e9-a1c7-0a580a8200ae",
                 "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9",
                 "cost": "example cost",
                 "uncertainty": "1.0"},
                {"forecast": "68a1c22c-87b5-11e9-bf88-0a580a8200ae",
                 "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9",
                 "normalization": "1000.0",
                 "uncertainty": "15.0",
                 "cost": "example cost",
                 "reference_forecast": "refbc386-8712-11e9-a1c7-0a580a8200ae"},
                {"forecast": "49220780-76ae-4b11-bef1-7a75bdc784e3",
                 "aggregate": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
                 "cost": "example cost",
                 "uncertainty": "5.0"}
            ],
            "costs": [
                {
                    "name": "example cost",
                    "type": "constant",
                    "parameters": {
                        "cost": 1.0,
                        "aggregation": "sum",
                        "net": True
                    }
                }
            ]
        }}
    session.create_report(report)
    posted = mocked.last_request.json()
    assert posted == expected


def test_apisession_create_report_mult_costs(requests_mock, report_objects,
                                             mocker):
    session = api.APISession('')
    report = report_objects[0]
    newobj = list(report.report_parameters.object_pairs)
    ncost = datamodel.Cost(
        name='other cost',
        type='constant',
        parameters=datamodel.ConstantCost(
            cost=2.0,
            aggregation='sum',
            net=True
        )
    )
    newobj[-1] = newobj[-1].replace(cost='other cost')
    report = report.replace(
        report_parameters=report.report_parameters.replace(
            costs=(report.report_parameters.costs[0], ncost),
            object_pairs=tuple(newobj)))
    mocked = requests_mock.register_uri('POST', f'{session.base_url}/reports/')
    mocker.patch('solarforecastarbiter.io.api.APISession.get_report',
                 return_value=report)
    expected = {
        "report_parameters": {
            "name": "NREL MIDC OASIS GHI Forecast Analysis",
            "start": "2019-04-01T00:00:00-07:00",
            "end": "2019-04-04T23:59:00-07:00",
            "forecast_fill_method": "forward",
            "filters": [
                {
                    'quality_flags': [
                        "USER FLAGGED",
                        "NIGHTTIME",
                        "LIMITS EXCEEDED",
                        "STALE VALUES",
                        "INTERPOLATED VALUES",
                        "INCONSISTENT IRRADIANCE COMPONENTS",
                    ],
                    'discard_before_resample': True,
                    'resample_threshold_percentage': 10.
                },
                {'time_of_day_range': ['12:00', '14:00']}],
            "metrics": ["mae", "rmse", "mbe", "s", "cost"],
            "categories": ["total", "date", "hour"],
            "object_pairs": [
                {"forecast": "da2bc386-8712-11e9-a1c7-0a580a8200ae",
                 "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9",
                 "cost": "example cost",
                 "uncertainty": "1.0"},
                {"forecast": "68a1c22c-87b5-11e9-bf88-0a580a8200ae",
                 "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9",
                 "normalization": "1000.0",
                 "uncertainty": "15.0",
                 "cost": "example cost",
                 "reference_forecast": "refbc386-8712-11e9-a1c7-0a580a8200ae"},
                {"forecast": "49220780-76ae-4b11-bef1-7a75bdc784e3",
                 "aggregate": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
                 "cost": "other cost",
                 "uncertainty": "5.0"}
            ],
            "costs": [
                {
                    "name": "example cost",
                    "type": "constant",
                    "parameters": {
                        "cost": 1.0,
                        "aggregation": "sum",
                        "net": True
                    }
                },
                {
                    "name": "other cost",
                    "type": "constant",
                    "parameters": {
                        "cost": 2.0,
                        "aggregation": "sum",
                        "net": True
                    }
                }
            ]
        }}
    session.create_report(report)
    posted = mocked.last_request.json()
    assert posted == expected


def test_apisession_create_report_no_costs(
        requests_mock, report_objects, mocker):
    session = api.APISession('')
    report = report_objects[0]
    report = report.replace(
        report_parameters=report.report_parameters.replace(
            costs=tuple(),
            object_pairs=tuple(
                op.replace(cost=None) for op in
                report.report_parameters.object_pairs)))
    mocked = requests_mock.register_uri('POST', f'{session.base_url}/reports/')
    mocker.patch('solarforecastarbiter.io.api.APISession.get_report',
                 return_value=report)
    expected = {
        "report_parameters": {
            "name": "NREL MIDC OASIS GHI Forecast Analysis",
            "start": "2019-04-01T00:00:00-07:00",
            "end": "2019-04-04T23:59:00-07:00",
            "forecast_fill_method": "forward",
            "filters": [
                {
                    'quality_flags': [
                        "USER FLAGGED",
                        "NIGHTTIME",
                        "LIMITS EXCEEDED",
                        "STALE VALUES",
                        "INTERPOLATED VALUES",
                        "INCONSISTENT IRRADIANCE COMPONENTS",
                    ],
                    'discard_before_resample': True,
                    'resample_threshold_percentage': 10.
                },
                {'time_of_day_range': ['12:00', '14:00']}],
            "metrics": ["mae", "rmse", "mbe", "s", "cost"],
            "categories": ["total", "date", "hour"],
            "object_pairs": [
                {"forecast": "da2bc386-8712-11e9-a1c7-0a580a8200ae",
                 "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9",
                 "uncertainty": "1.0"},
                {"forecast": "68a1c22c-87b5-11e9-bf88-0a580a8200ae",
                 "observation": "9f657636-7e49-11e9-b77f-0a580a8003e9",
                 "normalization": "1000.0",
                 "uncertainty": "15.0",
                 "reference_forecast": "refbc386-8712-11e9-a1c7-0a580a8200ae"},
                {"forecast": "49220780-76ae-4b11-bef1-7a75bdc784e3",
                 "aggregate": "458ffc27-df0b-11e9-b622-62adb5fd6af0",
                 "uncertainty": "5.0"}
            ],
            "costs": []
        }}
    session.create_report(report)
    posted = mocked.last_request.json()
    assert posted == expected


def test_apisession_post_raw_report_processed_data(
        requests_mock, raw_report, report_objects, ref_forecast_id):
    _, obs, fx0, fx1, agg, fxagg = report_objects
    session = api.APISession('')
    ids = [
        fx0.forecast_id, obs.observation_id, fx1.forecast_id,
        obs.observation_id, ref_forecast_id, fxagg.forecast_id,
        agg.aggregate_id,
    ]
    mocked = requests_mock.register_uri(
        'POST', re.compile(f'{session.base_url}/reports/.*/values'),
        [{'text': id_} for id_ in ids])
    inp = raw_report(True)
    out = session.post_raw_report_processed_data('report_id', inp)
    exp = raw_report(False)
    assert out == exp.processed_forecasts_observations
    history = mocked.request_history
    for i, id_ in enumerate(ids):
        assert id_ == history[i].json()['object_id']


def test_apisession_get_raw_report_processed_data(
        requests_mock, raw_report, report_objects, ref_forecast_id):
    _, obs, fx0, fx1, agg, fxagg = report_objects
    session = api.APISession('')
    ser = pd.Series(name='value', index=pd.DatetimeIndex(
        [], tz='UTC', name='timestamp'), dtype=float)
    val = utils.serialize_timeseries(ser)
    requests_mock.register_uri(
        'GET', re.compile(f'{session.base_url}/reports/.*/values'),
        json=[{'id': id_, 'processed_values': val} for id_ in
              (fx0.forecast_id, fx1.forecast_id, obs.observation_id,
               ref_forecast_id, agg.aggregate_id, fxagg.forecast_id)])
    inp = raw_report(False)
    out = session.get_raw_report_processed_data('', inp)
    for fxo in out:
        pdt.assert_series_equal(fxo.forecast_values, ser)
        pdt.assert_series_equal(fxo.observation_values, ser)
        if fxo.reference_forecast_values is not None:
            pdt.assert_series_equal(fxo.reference_forecast_values, ser)


def test_apisession_post_raw_report(requests_mock, raw_report, mocker,
                                    report_objects):
    raw = raw_report(True)
    _, obs, fx0, fx1, agg, fxagg = report_objects
    session = api.APISession('')
    ids = [fx0.forecast_id, obs.observation_id, fx1.forecast_id,
           obs.observation_id, agg.aggregate_id, fxagg.forecast_id]
    requests_mock.register_uri(
        'POST', re.compile(f'{session.base_url}/reports/.*/values'),
        [{'text': id_} for id_ in ids])
    mocked = requests_mock.register_uri(
        'POST', re.compile(f'{session.base_url}/reports/.*/raw'))
    status = mocker.patch(
        'solarforecastarbiter.io.api.APISession.update_report_status')
    session.post_raw_report('', raw)
    assert isinstance(mocked.last_request.json(), dict)
    assert status.called


def test_apisession_update_report_status(requests_mock):
    session = api.APISession('')
    mocked = requests_mock.register_uri(
        'POST', re.compile(f'{session.base_url}/reports/REPORT_ID/status/'))
    session.update_report_status('REPORT_ID', 'complete')
    assert mocked.last_request.url.split('/')[-1] == 'complete'


@pytest.fixture()
def mockobs(many_observations, mocker):
    def _getobs(cls, observation_id):
        for obs in many_observations:
            if observation_id == obs.observation_id:
                return obs

    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation',
        new=_getobs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.list_observations',
        return_value=many_observations)


def test_apisession_get_aggregate(requests_mock, aggregate_text, aggregate,
                                  mockobs):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', re.compile(f'{session.base_url}/aggregates/.*/metadata'),
        content=aggregate_text)
    agg = session.get_aggregate('')
    assert agg == aggregate


def test_apisession_list_aggregates(requests_mock, aggregate_text, aggregate,
                                    mockobs):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', re.compile(f'{session.base_url}/aggregates/'),
        content=b'[' + aggregate_text + b']')
    aggs = session.list_aggregates()
    assert aggs[0] == aggregate


def test_apisession_list_aggregates_single(requests_mock, aggregate_text,
                                           aggregate, mockobs):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', re.compile(f'{session.base_url}/aggregates/'),
        content=aggregate_text)
    aggs = session.list_aggregates()
    assert aggs[0] == aggregate


def test_apisession_list_aggregates_empty(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/aggregates/.*')
    requests_mock.register_uri('GET', matcher, content=b"[]")
    obs_list = session.list_aggregates()
    assert obs_list == []


def test_apisession_create_aggregate(requests_mock, aggregate, aggregate_text,
                                     mockobs):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/aggregates/.*')

    def callback(request, context):
        if request.url.endswith('aggregates/'):
            ad = json.loads(aggregate_text)
            for key in ('provider', 'aggregate_id', 'created_at',
                        'modified_at', 'observations', 'interval_value_type'):
                del ad[key]
            assert ad == request.json()
            return aggregate.aggregate_id
        else:
            rj = request.json()
            assert 'observations' in rj
            assert 'observation_id' in rj['observations'][0]
            assert (
                'effective_from' in rj['observations'][0]
                or 'effective_until' in rj['observations'][0])

    requests_mock.register_uri('POST', matcher, text=callback)
    requests_mock.register_uri('GET', matcher, content=aggregate_text)
    aggregate_dict = aggregate.to_dict()
    del aggregate_dict['aggregate_id']
    del aggregate_dict['interval_value_type']
    ss = datamodel.Aggregate.from_dict(aggregate_dict)
    new_aggregate = session.create_aggregate(ss)
    assert new_aggregate == aggregate


def test_apisession_get_user_info(requests_mock):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/users/current')
    requests_mock.register_uri('GET', matcher, content=b'{"test": "key"}')
    out = session.get_user_info()
    assert out['test'] == 'key'


time_range_params = pytest.mark.parametrize('mint,maxt,expected', [
    ('"2020-03-03T11:34:23+00:00"', '"2020-03-04T13:00:00+00:00"',
     (pd.Timestamp(year=2020, month=3, day=3, hour=11, minute=34, second=23,
                   tz='UTC'),
      pd.Timestamp(year=2020, month=3, day=4, hour=13, tz='UTC'))),
    ('null', 'null', (pd.NaT, pd.NaT)),
    ('"2020-03-03T11:34:23"', '"2020-03-04T13:00:00+00:00"',
     (pd.Timestamp(year=2020, month=3, day=3, hour=11, minute=34, second=23,
                   tz='UTC'),
      pd.Timestamp(year=2020, month=3, day=4, hour=13, tz='UTC'))),
    ('"2020-03-03T11:00:00+00:00"', '"2020-03-11T23:01:00"',
     (pd.Timestamp(year=2020, month=3, day=3, hour=11, tz='UTC'),
      pd.Timestamp(year=2020, month=3, day=11, hour=23, minute=1,
                   tz='UTC')))
])


@time_range_params
def test_get_observation_time_range(requests_mock, mint, maxt, expected):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', f'{session.base_url}/observations/obsid/values/timerange',
        content=(
            '{"observation_id": "obsid", "min_timestamp": '
            f'{mint}, "max_timestamp": {maxt}' + '}').encode())
    out = session.get_observation_time_range('obsid')
    assert out == expected


@time_range_params
def test_get_cdf_forecast_time_range(requests_mock, mint, maxt, expected):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET',
        f'{session.base_url}/forecasts/cdf/single/fxid/values/timerange',
        content=(
            '{"forecast_id": "fxid", "min_timestamp": '
            f'{mint}, "max_timestamp": {maxt}' + '}').encode())
    out = session.get_probabilistic_forecast_constant_value_time_range('fxid')
    assert out == expected


@time_range_params
def test_get_forecast_time_range(requests_mock, mint, maxt, expected):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', f'{session.base_url}/forecasts/single/fxid/values/timerange',
        content=(
            '{"forecast_id": "fxid", "min_timestamp": '
            f'{mint}, "max_timestamp": {maxt}' + '}').encode())
    out = session.get_forecast_time_range('fxid')
    assert out == expected


@pytest.mark.parametrize('inp,exp', [
    ('[]', np.array([], dtype='datetime64[D]')),
    ('["2019-01-01"]',
     np.array(['2019-01-01'], dtype='datetime64[D]')),
    ('["2019-01-03", "2019-04-01"]',
     np.array(['2019-01-03', '2019-04-01'],
              dtype='datetime64[D]')),
])
def test_get_observation_values_not_flagged(requests_mock, inp, exp):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', f'{session.base_url}/observations/obsid/values/unflagged',
        content=(
            '{"observation_id": "obsid", "_links": {}, '
            f'"dates": {inp}' + '}').encode())
    out = session.get_observation_values_not_flagged(
        'obsid', '2019-01-01T00:00Z', '2020-01-01T00:00Z', 16)
    assert all(out == exp)


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


def test_real_apisession_list_sites_in_zone(real_session):
    sites = real_session.list_sites_in_zone('Reference Region 5')
    assert isinstance(sites, list)
    assert isinstance(sites[0], datamodel.Site)


@pytest.mark.parametrize('lat,lon,expected', [
    (32.1, -110.8, {'Reference Region 3'}),
    (45, -70.0, {'Reference Region 7'}),
    (0, 0, set())
])
def test_real_apisession_search_climatezones(real_session, lat, lon, expected):
    zones = real_session.search_climatezones(lat, lon)
    assert set(zones) == expected


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


def test_real_apisession_create_forecast_invalid(
        single_forecast_text, single_forecast, real_session):
    forecastd = json.loads(single_forecast_text)
    forecastd['name'] = f'Test create forecast {randint(0, 100)}'
    forecastd['site'] = single_forecast.site
    forecast = datamodel.Forecast.from_dict(forecastd)
    object.__setattr__(forecast, 'interval_label', 'mean')
    with pytest.raises(requests.exceptions.HTTPError) as e:
        real_session.create_forecast(forecast)
    assert 'Must be one of' in str(e.value)


def test_real_apisession_get_prob_forecast(real_session):
    fx = real_session.get_probabilistic_forecast(
        'ef51e87c-50b9-11e9-8647-d663bd873d93')
    assert isinstance(fx, datamodel.ProbabilisticForecast)


def test_real_apisession_list_prob_forecasts(real_session):
    fxs = real_session.list_probabilistic_forecasts()
    assert isinstance(fxs, list)
    assert isinstance(fxs[0], datamodel.ProbabilisticForecast)


def test_real_apisession_create_prob_forecast(
        prob_forecast_text, prob_forecasts, prob_forecast_constant_value,
        real_session):
    forecastd = json.loads(prob_forecast_text)
    forecastd['name'] = f'Test create forecast {randint(0, 100)}'
    forecastd['site'] = prob_forecasts.site
    fx_prob_cv_d = prob_forecast_constant_value.to_dict()
    fx_prob_cv_d['name'] = forecastd['name']
    del fx_prob_cv_d['forecast_id']
    forecastd['constant_values'] = (
        datamodel.ProbabilisticForecastConstantValue.from_dict(fx_prob_cv_d), )
    forecast = datamodel.ProbabilisticForecast.from_dict(forecastd)
    new_forecast = real_session.create_probabilistic_forecast(forecast)
    real_session.delete(f'/forecasts/cdf/{new_forecast.forecast_id}')
    test_attrs = ('name', 'site', 'variable', 'interval_label',
                  'interval_length', 'lead_time_to_start', 'run_length',
                  'extra_parameters')
    for attr in test_attrs:
        assert getattr(new_forecast, attr) == getattr(forecast, attr)
    for new_cv, cv in zip(new_forecast.constant_values,
                          forecast.constant_values):
        for attr in test_attrs:
            assert getattr(new_cv, attr) == getattr(cv, attr)


def test_real_apisession_get_prob_forecast_constant_value(real_session):
    fx = real_session.get_probabilistic_forecast_constant_value(
        '633f9b2a-50bb-11e9-8647-d663bd873d93')
    assert isinstance(fx, datamodel.ProbabilisticForecastConstantValue)


def test_real_apisession_get_observation_values(real_session):
    start = pd.Timestamp('2019-04-15T00:00:00Z')
    end = pd.Timestamp('2019-04-15T12:00:00Z')
    obs = real_session.get_observation_values(
        '123e4567-e89b-12d3-a456-426655440000',
        start, end)
    assert isinstance(obs, pd.DataFrame)
    assert set(obs.columns) == {'value', 'quality_flag'}
    assert len(obs.index) > 0
    pdt.assert_frame_equal(obs.loc[start:end], obs)


def test_real_apisession_get_observation_values_tz(real_session):
    # use different tzs to confirm that it works
    start = pd.Timestamp('2019-04-14T22:00:00-0200')
    end = pd.Timestamp('2019-04-15T05:00:00-0700')
    obs = real_session.get_observation_values(
        '123e4567-e89b-12d3-a456-426655440000',
        start, end)
    assert isinstance(obs, pd.DataFrame)
    assert set(obs.columns) == {'value', 'quality_flag'}
    assert len(obs.index) > 0
    end = end.tz_convert(start.tzinfo)
    pdt.assert_frame_equal(obs.loc[start:end], obs)


def test_real_apisession_get_forecast_values(real_session):
    start = pd.Timestamp('2019-04-15T00:00:00Z')
    end = pd.Timestamp('2019-04-15T12:00:00Z')
    fx = real_session.get_forecast_values(
        'f8dd49fa-23e2-48a0-862b-ba0af6dec276',
        start, end)
    assert isinstance(fx, pd.Series)
    assert len(fx) > 0
    pdt.assert_series_equal(fx.loc[start:end], fx)


def test_real_apisession_get_forecast_values_tz(real_session):
    # use different tzs to confirm that it works
    start = pd.Timestamp('2019-04-14T20:00:00-0400')
    end = pd.Timestamp('2019-04-15T13:00:00+0100')
    fx = real_session.get_forecast_values(
        'f8dd49fa-23e2-48a0-862b-ba0af6dec276',
        start, end)
    assert isinstance(fx, pd.Series)
    assert len(fx) > 0
    end = end.tz_convert(start.tzinfo)
    pdt.assert_series_equal(fx.loc[start:end], fx)


def test_real_apisession_get_prob_forecast_values_tz(real_session):
    # use different tzs to confirm that it works
    start = pd.Timestamp('2019-04-14T20:00:00-0400')
    end = pd.Timestamp('2019-04-15T13:00:00+0100')
    fx = real_session.get_probabilistic_forecast_constant_value_values(
        '633f9b2a-50bb-11e9-8647-d663bd873d93',
        start, end)
    assert isinstance(fx, pd.Series)
    assert len(fx) > 0
    end = end.tz_convert(start.tzinfo)
    pdt.assert_series_equal(fx.loc[start:end], fx)


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


def test_real_apisession_post_prob_forecast_constant_val_values(real_session):
    test_ser = pd.Series(
        [np.random.random()], name='value',
        index=pd.DatetimeIndex([pd.Timestamp('2019-04-14T00:00:00Z')],
                               name='timestamp'))
    real_session.post_probabilistic_forecast_constant_value_values(
        '633f9b2a-50bb-11e9-8647-d663bd873d93', test_ser)
    fx = real_session.get_probabilistic_forecast_constant_value_values(
        '633f9b2a-50bb-11e9-8647-d663bd873d93',
        pd.Timestamp('2019-04-14T00:00:00Z'),
        pd.Timestamp('2019-04-14T00:01:00Z'))
    pdt.assert_series_equal(fx, test_ser)


def test_real_apisession_get_aggregate(real_session):
    agg = real_session.get_aggregate('458ffc27-df0b-11e9-b622-62adb5fd6af0')
    assert isinstance(agg, datamodel.Aggregate)


def test_real_apisession_list_aggregates(real_session):
    aggs = real_session.list_aggregates()
    assert isinstance(aggs, list)
    assert isinstance(aggs[0], datamodel.Aggregate)
    assert '458ffc27-df0b-11e9-b622-62adb5fd6af0' in {
        a.aggregate_id for a in aggs}


def test_real_apisession_create_aggregate(real_session, aggregate):
    new_agg = real_session.create_aggregate(aggregate)
    real_session.delete(f'/aggregates/{new_agg.aggregate_id}')
    for attr in ('name', 'description', 'variable', 'aggregate_type',
                 'interval_length', 'interval_label', 'timezone'):
        assert getattr(new_agg, attr) == getattr(aggregate, attr)
    for i, obs in enumerate(new_agg.observations):
        assert (
            aggregate.observations[i].observation.observation_id ==
            obs.observation.observation_id)
        for attr in ('effective_from', 'effective_until',
                     'observation_deleted_at'):
            assert getattr(aggregate.observations[i], attr) == (
                getattr(obs, attr))


def test_real_apisession_get_aggregate_values(real_session):
    start = pd.Timestamp('2019-04-15T00:00:00Z')
    end = pd.Timestamp('2019-04-15T12:00:00Z')
    agg = real_session.get_aggregate_values(
        '458ffc27-df0b-11e9-b622-62adb5fd6af0',
        start, end)
    assert isinstance(agg, pd.DataFrame)
    assert set(agg.columns) == {'value', 'quality_flag'}
    assert len(agg.index) > 0
    pdt.assert_frame_equal(agg.loc[start:end], agg)


def test_real_apisession_get_user_info(real_session):
    user_info = real_session.get_user_info()
    assert user_info['organization'] == 'Organization 1'


def test_real_apisession_get_observation_time_range(real_session):
    out = real_session.get_observation_time_range(
        '123e4567-e89b-12d3-a456-426655440000')
    assert out == (
        pd.Timestamp('2019-04-14T00:00:00Z'),
        pd.Timestamp('2019-04-17T06:55:00Z'))


def test_real_apisession_get_forecast_time_range(real_session):
    out = real_session.get_forecast_time_range(
        'f8dd49fa-23e2-48a0-862b-ba0af6dec276')
    assert out == (
        pd.Timestamp('2019-04-14T00:00:00Z'),
        pd.Timestamp('2019-04-17T06:59:00Z'))


def test_real_apisession_get_cdf_forecast_time_range(real_session):
    out = real_session.get_probabilistic_forecast_constant_value_time_range(
        '633f9b2a-50bb-11e9-8647-d663bd873d93')
    assert out == (
        pd.Timestamp('2019-04-14T00:00:00Z'),
        pd.Timestamp('2019-04-17T06:55:00Z'))


def test_real_apisession_get_observation_values_not_flagged(real_session):
    start = pd.Timestamp('2019-04-15T00:00:00Z')
    end = pd.Timestamp('2019-04-15T12:00:00Z')
    out = real_session.get_observation_values_not_flagged(
        '123e4567-e89b-12d3-a456-426655440000',
        start, end, 1)
    assert isinstance(out, np.ndarray)
    assert out.dtype == 'datetime64[D]'
    assert all(out == np.array(['2019-04-15'], dtype='datetime64[D]'))


@pytest.mark.parametrize('ftype,expected_fn', [
    ('forecast', 'get_forecast'),
    ('event_forecast', 'get_forecast'),
    ('probabilistic_forecast', 'get_probabilistic_forecast'),
    ('probabilistic_forecast_constant_value',
     'get_probabilistic_forecast_constant_value'),
])
def test_api_session_forecast_get_by_type(ftype, expected_fn):
    test_session = api.APISession('token')
    get_fn = test_session._forecast_get_by_type(ftype)
    assert get_fn == getattr(test_session, expected_fn)


def test_api_session_forecast_get_by_type_invalid_type():
    test_session = api.APISession('token')
    with pytest.raises(ValueError):
        test_session._forecast_get_by_type('bad_type')


def value_callback(include_qf=False, freq='1H'):
    def fn(request, context):
        query_params = parse_qs(request.query)
        start = pd.Timestamp(query_params['start'][0]).floor(freq)
        end = pd.Timestamp(query_params['end'][0]).floor(freq)
        idx = pd.date_range(start, end, freq=freq)
        data = {'value': 1.0}
        if include_qf:
            data.update({'quality_flag': 0})
        df = pd.DataFrame(index=idx, data=data)
        df['timestamp'] = df.index
        resp_json = df.to_json(orient="records", date_format='iso')
        out = bytes('{"values": ' + resp_json + '}', 'utf8')
        return out
    return fn


def test_apisession_chunk_value_requests_obs_df(requests_mock):
    session = api.APISession('')
    callback = value_callback(True)
    start = pd.Timestamp('2017-01-01T12:00:00-0700')
    end = pd.Timestamp('2020-01-01T12:25:00-0700')
    expected = pd.DataFrame(
        index=pd.date_range(start, end, freq='1H', name='timestamp'),
        data={'value': 1.0, 'quality_flag': 0},
    )
    matcher = re.compile(
        f'{session.base_url}/observations/.*/values')
    requests_mock.register_uri('GET', matcher, content=callback)
    out = session.chunk_value_requests(
        '/observations/obsid/values',
        start,
        end,
        utils.json_payload_to_observation_df,
    )
    pdt.assert_frame_equal(out, expected.tz_convert('UTC'))


def test_apisession_chunk_value_requests_fx_series(requests_mock):
    session = api.APISession('')
    callback = value_callback(True)
    start = pd.Timestamp('2017-01-01T12:00:00-0700')
    end = pd.Timestamp('2020-01-01T12:25:00-0700')
    expected = pd.Series(
        1.,
        index=pd.date_range(start, end, freq='1H', name='timestamp'),
        name='value',
    )
    matcher = re.compile(
        f'{session.base_url}/forecasts/.*/values')
    requests_mock.register_uri('GET', matcher, content=callback)
    out = session.chunk_value_requests(
        '/forecasts/single/fxid/values',
        start,
        end,
        utils.json_payload_to_forecast_series,
    )
    pdt.assert_series_equal(out, expected.tz_convert('UTC'))


@pytest.mark.parametrize('limit', [
    '180D',
    '90D',
])
def test_apisession_chunk_value_requests_recursion(
        mocker, requests_mock, limit):
    session = api.APISession('')
    mocked_get = mocker.patch.object(
        session,
        'chunk_value_requests',
        side_effect=session.chunk_value_requests
    )
    callback = value_callback(True)
    start = pd.Timestamp('2017-01-01T12:00:00-0700')
    end = pd.Timestamp('2020-01-01T12:25:00-0700')
    matcher = re.compile(
        f'{session.base_url}/forecasts/.*/values')
    requests_mock.register_uri('GET', matcher, content=callback)
    session.chunk_value_requests(
        '/forecasts/single/fxid/values',
        start,
        end,
        utils.json_payload_to_forecast_series,
        request_limit=limit,
    )
    # assert number of recursive calls is the period / limit plus the
    # original call
    expected_n_calls = int((end - start) / pd.Timedelta(limit)) + 1
    assert mocked_get.call_count == expected_n_calls


@pytest.mark.parametrize('trange,gaps,exp', [
    # simple case of gap in [start,end] and timerange
    ((pd.Timestamp('2018-01-01T00:00Z'), pd.Timestamp('2021-01-01T00:00Z')),
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T13:00Z'))],
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T13:00Z'))]),
    # simple case of multiple gaps
    ((pd.Timestamp('2018-01-01T00:00Z'), pd.Timestamp('2021-01-01T00:00Z')),
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T13:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T15:00Z'))],
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T13:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T15:00Z'))]),
    # no gaps at all
    ((pd.Timestamp('2018-01-01T00:00Z'), pd.Timestamp('2021-01-01T00:00Z')),
     [], []),
    # gap extends past end
    ((pd.Timestamp('2018-01-01T00:00Z'), pd.Timestamp('2021-01-01T00:00Z')),
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T13:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-15T15:00Z'))],
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T13:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # gap extends past start
    ((pd.Timestamp('2018-01-01T00:00Z'), pd.Timestamp('2021-01-01T00:00Z')),
     [(pd.Timestamp('2020-01-09T12:00Z'), pd.Timestamp('2020-01-10T13:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T15:00Z'))],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-10T13:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T15:00Z'))]),
    # gap extends past start and end
    ((pd.Timestamp('2018-01-01T00:00Z'), pd.Timestamp('2021-01-01T00:00Z')),
     [(pd.Timestamp('2020-01-09T12:00Z'), pd.Timestamp('2020-01-10T13:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-19T15:00Z'))],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-10T13:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # range before start
    ((pd.Timestamp('2018-01-01T00:00Z'), pd.Timestamp('2019-01-01T00:00Z')),
     [],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # range after end
    ((pd.Timestamp('2020-04-01T00:00Z'), pd.Timestamp('2020-05-01T00:00Z')),
     [],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # a nan
    ((pd.Timestamp('2020-04-01T00:00Z'), pd.Timestamp(None)),
     [],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # two nans
    ((pd.Timestamp(None), pd.Timestamp(None)),
     [],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # timerange after start, no gap
    ((pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2021-01-01T00:00Z')),
     [],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-10T12:00Z'))]),
    # timerange after start, gap
    ((pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2021-01-01T00:00Z')),
     [(pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T16:00Z'))],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-10T12:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T16:00Z'))]),
    # timerange before end, no gap
    ((pd.Timestamp('2019-01-10T12:00Z'), pd.Timestamp('2020-01-10T19:00Z')),
     [],
     [(pd.Timestamp('2020-01-10T19:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # timerange before end, gap
    ((pd.Timestamp('2019-01-10T12:00Z'), pd.Timestamp('2020-01-10T19:00Z')),
     [(pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T16:00Z'))],
     [(pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T16:00Z')),
      (pd.Timestamp('2020-01-10T19:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # timerange within, no gap
    ((pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T19:00Z')),
     [],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-10T12:00Z')),
      (pd.Timestamp('2020-01-10T19:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
    # timerange within, gap
    ((pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T19:00Z')),
     [(pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T16:00Z'))],
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-10T12:00Z')),
      (pd.Timestamp('2020-01-10T14:00Z'), pd.Timestamp('2020-01-10T16:00Z')),
      (pd.Timestamp('2020-01-10T19:00Z'), pd.Timestamp('2020-01-11T00:00Z'))]),
])
def test__fixup_gaps(trange, gaps, exp):
    session = api.APISession('')
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    assert list(session._fixup_gaps(trange, gaps, start, end)) == exp


@pytest.mark.parametrize('first,second,exp', [
    ('2020-01-10T12:00:00+00:00', '2020-01-10T14:00:00+00:00',
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T14:00Z'))]),
    ('2020-01-10T12:00:00', '2020-01-10T14:00:00',
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T14:00Z'))])
])
def test__process_gaps(requests_mock, first, second, exp):
    session = api.APISession('')
    url = '/observations/obsid/values/gaps'
    requests_mock.register_uri(
        'GET', session.base_url + url,
        content=(
            '{"observation_id": "obsid", "gaps": [{"timestamp": "' + first +
            '", "next_timestamp": "' + second + '"}]}').encode())
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    out = session._process_gaps(url, start, end)
    assert out == exp


def test__get_obs_gaps_null(requests_mock):
    session = api.APISession('')
    url = '/observations/obsid/values/gaps'
    requests_mock.register_uri(
        'GET', session.base_url + url,
        content=(
            '{"observation_id": "obsid", "gaps": [{"timestamp": null, '
            '"next_timestamp": "2020-01-01T00:00+00:00"}]}').encode())
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    out = session._process_gaps(url, start, end)
    assert out == []


def test__get_obs_gaps_no_gaps(requests_mock):
    session = api.APISession('')
    url = '/observations/obsid/values/gaps'
    requests_mock.register_uri(
        'GET', session.base_url + url,
        content=('{"observation_id": "obsid", "gaps": []}').encode())
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    out = session._process_gaps(url, start, end)
    assert out == []


gaps_parameters = pytest.mark.parametrize('trange,exp,gaps', [
    ((pd.Timestamp('2020-01-01T00:00Z'), pd.Timestamp('2020-02-01T00:00Z')),
     [(pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T14:00Z')),
      (pd.Timestamp('2020-01-10T19:00Z'), pd.Timestamp('2020-01-10T22:00Z'))],
     ('[{"timestamp": "2020-01-10T12:00Z", '
      '"next_timestamp": "2020-01-10T14:00Z"},'
      '{"timestamp": "2020-01-10T19:00Z", '
      '"next_timestamp": "2020-01-10T22:00Z"}]')
     ),
    ((pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-02-01T00:00Z')),
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-10T14:00Z')),
      (pd.Timestamp('2020-01-10T19:00Z'), pd.Timestamp('2020-01-10T22:00Z'))
      ],
     ('[{"timestamp": "2020-01-10T12:00Z", '
      '"next_timestamp": "2020-01-10T14:00Z"},'
      '{"timestamp": "2020-01-10T19:00Z", '
      '"next_timestamp": "2020-01-10T22:00Z"}]')
     ),
    ((pd.Timestamp(None), pd.Timestamp(None)),
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-11T00:00Z'))],
     '[]'
     ),
    ((pd.Timestamp('2020-01-01T00:00Z'), pd.Timestamp('2020-01-10T19:00Z')),
     [(pd.Timestamp('2020-01-10T19:00Z'), pd.Timestamp('2020-01-11T00:00Z'))],
     '[]'
     ),
    ((pd.Timestamp('2020-01-10T19:00Z'), pd.Timestamp('2020-01-20T19:00Z')),
     [(pd.Timestamp('2020-01-10T00:00Z'), pd.Timestamp('2020-01-10T19:00Z'))],
     '[]'
     ),
])


@gaps_parameters
def test_get_observation_value_gaps(requests_mock, mocker, trange, exp, gaps):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', f'{session.base_url}/observations/obsid/values/gaps',
        content=(
            '{"observation_id": "obsid", "gaps": ' + gaps + '}').encode())
    mocker.patch.object(
        session,
        'get_observation_time_range',
        return_value=trange
    )
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    out = list(session.get_observation_value_gaps('obsid', start, end))
    assert out == exp


@gaps_parameters
def test_get_forecast_value_gaps(requests_mock, mocker, trange, exp, gaps):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', f'{session.base_url}/forecasts/single/fxid/values/gaps',
        content=(
            '{"forecast_id": "obsid", "gaps": ' + gaps + '}').encode())
    mocker.patch.object(
        session,
        'get_forecast_time_range',
        return_value=trange
    )
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    assert list(session.get_forecast_value_gaps('fxid', start, end)) == exp


@gaps_parameters
def test_get_probabilistic_forecast_constant_value_value_gaps(
        requests_mock, mocker, trange, exp, gaps):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', f'{session.base_url}/forecasts/cdf/single/fxid/values/gaps',
        content=(
            '{"forecast_id": "obsid", "gaps": ' + gaps + '}').encode())
    mocker.patch.object(
        session,
        'get_probabilistic_forecast_constant_value_time_range',
        return_value=trange
    )
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    assert list(session.get_probabilistic_forecast_constant_value_value_gaps(
        'fxid', start, end)) == exp


@gaps_parameters
def test_get_probabilistic_forecast_value_gaps(
        requests_mock, mocker, trange, exp, gaps, prob_forecasts):
    session = api.APISession('')
    requests_mock.register_uri(
        'GET', f'{session.base_url}/forecasts/cdf/fxid/values/gaps',
        content=(
            '{"forecast_id": "obsid", "gaps": ' + gaps + '}').encode())
    mocker.patch.object(
        session,
        'get_probabilistic_forecast_constant_value_time_range',
        return_value=trange
    )
    mocker.patch.object(
        session,
        'get_probabilistic_forecast',
        return_value=prob_forecasts
    )
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    out = list(session.get_probabilistic_forecast_value_gaps('fxid', start,
                                                             end))
    assert out == exp


@pytest.mark.parametrize('objt,fnc', [
    ('observation', 'get_observation_value_gaps'),
    ('forecast', 'get_forecast_value_gaps'),
    ('prob_forecast', 'get_probabilistic_forecast_value_gaps'),
    ('prob_forecast_cv', 'get_probabilistic_forecast_constant_value_value_gaps'),  # NOQA
    pytest.param('aggregate', 'get_aggregate',
                 marks=pytest.mark.xfail(strict=True, raises=TypeError))
])
def test_get_value_gaps(objt, fnc, prob_forecasts, single_forecast,
                        single_observation, aggregate, mocker):
    start = pd.Timestamp('2020-01-10T00:00Z')
    end = pd.Timestamp('2020-01-11T00:00Z')
    if objt == 'observation':
        obj = single_observation
    elif objt == 'forecast':
        obj = single_forecast
    elif objt == 'aggregate':
        obj = aggregate
    elif objt == 'prob_forecast':
        obj = prob_forecasts
    elif objt == 'prob_forecast_cv':
        obj = prob_forecasts.constant_values[0]

    session = api.APISession('')
    fncpatch = mocker.patch.object(session, fnc)
    session.get_value_gaps(obj, start, end)
    fncpatch.assert_called_once()
