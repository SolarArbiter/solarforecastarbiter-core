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
                                        prob_forecast_constant_value_text):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/forecasts/cdf/single/.*')
    requests_mock.register_uri(
        'GET', matcher, content=prob_forecast_constant_value_text)
    matcher = re.compile(session.base_url + r'/forecasts/cdf/$')
    requests_mock.register_uri(
        'GET', matcher, content=many_prob_forecasts_text)
    fx_list = session.list_probabilistic_forecasts()
    assert fx_list == many_prob_forecasts


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


@pytest.fixture(params=[0, 1])
def obs_start_end(request):
    if request.param == 0:
        return (pd.Timestamp('2019-01-01T12:00:00-0700'),
                pd.Timestamp('2019-01-01T12:25:00-0700'))
    else:
        return ('2019-01-01T12:00:00-0700',
                '2019-01-01T12:25:00-0700')


def test_apisession_get_observation_values(
        requests_mock, observation_values, observation_values_text,
        obs_start_end):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*/values')
    requests_mock.register_uri('GET', matcher, content=observation_values_text)
    out = session.get_observation_values(
        'obsid', *obs_start_end)
    pdt.assert_frame_equal(out, observation_values)


@pytest.mark.parametrize('label,theslice', [
    (None, slice(0, 10)),
    ('beginning', slice(0, -1)),
    ('ending', slice(1, 10))
])
def test_apisession_get_observation_values_interval_label(
        requests_mock, observation_values, observation_values_text,
        label, theslice, obs_start_end):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*/values')
    requests_mock.register_uri('GET', matcher, content=observation_values_text)
    out = session.get_observation_values(
        'obsid', obs_start_end[0], obs_start_end[1], label)
    pdt.assert_frame_equal(out, observation_values.iloc[theslice])


@pytest.fixture()
def empty_df():
    return pd.DataFrame([], columns=['value', 'quality_flag'],
                        index=pd.DatetimeIndex([], name='timestamp'))


def test_apisession_get_observation_values_empty(requests_mock, empty_df):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*/values')
    requests_mock.register_uri('GET', matcher, content=b'{"values":[]}')
    out = session.get_observation_values(
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


@pytest.fixture()
def mock_request_fxobs(report_objects, mocker):
    _, obs, fx0, fx1 = report_objects
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)

    def returnone(fxid):
        if fxid == "da2bc386-8712-11e9-a1c7-0a580a8200ae":
            return fx0
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
    # TODO: fix filters
    expected = report_objects[0].replace(filters=())
    assert out == expected


def test_apisession_get_report_with_raw(
        requests_mock, report_text, report_objects, mock_request_fxobs,
        raw_report, mocker):
    raw = raw_report(False)
    raw_txt = utils.serialize_raw_report(raw)
    report = json.loads(report_text)
    report['raw_report'] = raw_txt
    report_text = json.dumps(report).encode()
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_raw_report_processed_data',
        return_value=())
    session = api.APISession('')
    requests_mock.register_uri('GET', f'{session.base_url}/reports/',
                               content=report_text)
    out = session.get_report('')
    # TODO: fix filters
    expected = report_objects[0].replace(
        filters=(),
        raw_report=raw.replace(processed_forecasts_observations=()))
    assert out == expected


def test_apisession_list_reports(requests_mock, report_text, report_objects,
                                 mock_request_fxobs):
    session = api.APISession('')
    requests_mock.register_uri('GET', f'{session.base_url}/reports',
                               content=b'['+report_text+b']')
    out = session.list_reports()
    # TODO: fix filters
    expected = [report_objects[0].replace(filters=())]
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
        "name": "NREL MIDC OASIS GHI Forecast Analysis",
        "report_parameters": {
            "start": "2019-04-01T00:00:00-07:00",
            "end": "2019-04-04T23:59:00-07:00",
            "filters": [],
            "metrics": ["mae", "rmse", "mbe"],
            "categories": ["total", "day", 'hour'],
            "object_pairs": [
                ["da2bc386-8712-11e9-a1c7-0a580a8200ae",
                 "9f657636-7e49-11e9-b77f-0a580a8003e9"],
                ["68a1c22c-87b5-11e9-bf88-0a580a8200ae",
                 "9f657636-7e49-11e9-b77f-0a580a8003e9"]
            ]
        }}
    session.create_report(report)
    posted = mocked.last_request.json()
    assert posted == expected


def test_apisession_post_raw_report_processed_data(
        requests_mock, raw_report, report_objects):
    _, obs, fx0, fx1 = report_objects
    session = api.APISession('')
    ids = [fx0.forecast_id, obs.observation_id, fx1.forecast_id,
           obs.observation_id]
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
        requests_mock, raw_report, report_objects):
    _, obs, fx0, fx1 = report_objects
    session = api.APISession('')
    ser = pd.Series(name='value', index=pd.Index([], name='timestamp'))
    val = utils.serialize_data(ser)
    requests_mock.register_uri(
        'GET', re.compile(f'{session.base_url}/reports/.*/values'),
        json=[{'id': id_, 'processed_values': val} for id_ in
              (fx0.forecast_id, fx1.forecast_id, obs.observation_id)])
    inp = raw_report(False)
    out = session.get_raw_report_processed_data('', inp)
    for fxo in out:
        pdt.assert_series_equal(fxo.forecast_values, ser)
        pdt.assert_series_equal(fxo.observation_values, ser)


def test_apisession_post_raw_report(requests_mock, raw_report, mocker,
                                    report_objects):
    raw = raw_report(True)
    _, obs, fx0, fx1 = report_objects
    session = api.APISession('')
    ids = [fx0.forecast_id, obs.observation_id, fx1.forecast_id,
           obs.observation_id]
    requests_mock.register_uri(
        'POST', re.compile(f'{session.base_url}/reports/.*/values'),
        [{'text': id_} for id_ in ids])
    mocked = requests_mock.register_uri(
        'POST', re.compile(f'{session.base_url}/reports/.*/metrics'))
    status = mocker.patch(
        'solarforecastarbiter.io.api.APISession.update_report_status')
    session.post_raw_report('', raw)
    assert isinstance(mocked.last_request.json()['raw_report'], str)
    assert status.called


def test_apisession_update_report_status(requests_mock):
    session = api.APISession('')
    mocked = requests_mock.register_uri(
        'POST', re.compile(f'{session.base_url}/reports/REPORT_ID/status/'))
    session.update_report_status('REPORT_ID', 'complete')
    assert mocked.last_request.url.split('/')[-1] == 'complete'


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


def test_real_apisession_create_forecast_invalid(
        single_forecast_text, single_forecast, real_session):
    forecastd = json.loads(single_forecast_text)
    forecastd['name'] = f'Test create forecast {randint(0, 100)}'
    forecastd['site'] = single_forecast.site
    forecastd['interval_label'] = 'mean'
    forecast = datamodel.Forecast.from_dict(forecastd)
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
    assert set(obs.columns) == set(['value', 'quality_flag'])
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
    assert set(obs.columns) == set(['value', 'quality_flag'])
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
