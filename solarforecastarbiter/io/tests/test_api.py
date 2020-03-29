import json
import re
import copy


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
    matcher = re.compile(f'https://api.solarforecastarbiter.org/aggregates/.*')
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


def test_apisession_list_prob_forecast_single(requests_mock, prob_forecasts,
                                              prob_forecast_text,
                                              mock_list_sites, mock_get_site,
                                              prob_forecast_constant_value_text,
                                              mock_get_agg):
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
    return pd.DataFrame([], columns=['value', 'quality_flag'],
                        index=pd.DatetimeIndex([], name='timestamp', tz='UTC'))


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


def test_apisession_get_prob_forecast_values(
        requests_mock, mock_get_site,
        prob_forecast_values, prob_forecast_values_text_list,
        prob_forecast_text, prob_forecast_constant_value_text, fx_start_end):
    session = api.APISession('')

    # modify ProbabilisticForecast json text
    probfx_json = json.loads(prob_forecast_text)
    probfx_json['forecast_id'] = 'FXID'
    probfx_json['axis'] = 'y'
    cvs_json = json.loads(prob_forecast_constant_value_text)
    cvs = []
    for col in prob_forecast_values:
        new_cv = copy.deepcopy(cvs_json)
        new_cv['constant_value'] = col
        new_cv['forecast_id'] = 'CV' + col
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
