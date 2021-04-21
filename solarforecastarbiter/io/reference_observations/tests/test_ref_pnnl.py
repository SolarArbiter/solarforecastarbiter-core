import inspect
import os
from pathlib import Path
import re

import pandas as pd

import pytest

from solarforecastarbiter.datamodel import Site
from solarforecastarbiter.io import api
from solarforecastarbiter.io.reference_observations import pnnl

TEST_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
TEST_DATA_PATH = Path(TEST_DIR) / 'data/pnnl_data'


@pytest.fixture
def session(requests_mock):
    return api.APISession('')


@pytest.fixture
def site():
    return Site(
        name='PNNL',
        latitude=46.341,
        longitude=-119.279,
        elevation=122.61,
        timezone='Etc/GMT+8',
        site_id='',
        provider='',
        extra_parameters='{"network": "PNNL", "network_api_id": "", "network_api_abbreviation": "", "observation_interval_length": 1.0, "attribution": ""}',  # noqa: E501
    )


@pytest.fixture
def site_no_extra(site):
    return site.replace(extra_parameters='')


@pytest.fixture
def log(mocker):
    log = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'pnnl.logger')
    return log


@pytest.fixture()
def mock_list_sites(mocker, many_sites):
    mocker.patch('solarforecastarbiter.io.api.APISession.list_sites',
                 return_value=many_sites)


def test_initialize_site_observations(
        requests_mock, mocker, session, site, single_observation,
        single_observation_text, mock_list_sites):
    matcher = re.compile(f'{session.base_url}/observations/.*')
    requests_mock.register_uri('POST', matcher,
                               text=single_observation.observation_id)
    requests_mock.register_uri('GET', matcher, content=single_observation_text)
    status = mocker.patch(
        'solarforecastarbiter.io.api.APISession.create_observation')
    pnnl.initialize_site_observations(session, site)
    assert status.called


def test_initialize_site_observations_fail(session, site_no_extra, log):
    pnnl.initialize_site_observations(session, site_no_extra)
    assert log.warning.call_count == 1


def test_initialize_site_forecasts(
        requests_mock, mocker, session, site, single_forecast,
        single_forecast_text, mock_list_sites):
    matcher = re.compile(f'{session.base_url}/forecasts/.*')
    requests_mock.register_uri('POST', matcher,
                               text=single_forecast.forecast_id)
    requests_mock.register_uri('GET', matcher, content=single_forecast_text)
    status = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'create_forecasts')
    pnnl.initialize_site_forecasts(session, site)
    assert status.called


def test_fetch(mocker, session, site):
    mocker.patch(
        'solarforecastarbiter.io.reference_observations.pnnl.DATA_PATH',
        TEST_DATA_PATH
    )
    start = pd.Timestamp('2017-01-01T0000Z')
    end = pd.Timestamp('2017-01-01T0200Z')
    out = pnnl.fetch(session, site, start, end)
    assert out.index[0] == pd.Timestamp('20161231T160000', tz='Etc/GMT+8')
    assert out.index[-1] == pd.Timestamp('20161231T175900', tz='Etc/GMT+8')
    assert (out.columns == pnnl.COLUMNS).all()


def test_fetch_over_year(mocker, session, site):
    mocker.patch(
        'solarforecastarbiter.io.reference_observations.pnnl.DATA_PATH',
        TEST_DATA_PATH
    )
    start = pd.Timestamp('2017-12-31T2300Z')
    end = pd.Timestamp('2018-01-01T0100Z')
    out = pnnl.fetch(session, site, start, end)
    assert out.index[0] == pd.Timestamp('20171231T150000', tz='Etc/GMT+8')
    assert out.index[-1] == pd.Timestamp('20171231T165900', tz='Etc/GMT+8')
    assert (out.columns == pnnl.COLUMNS).all()


def test_update_observation_data(mocker, session, site):
    obs_update = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'update_site_observations')
    start = pd.Timestamp('2017-01-01T0000Z')
    end = pd.Timestamp('2017-01-01T0200Z')
    pnnl.update_observation_data(session, [site], [], start, end)
    obs_update.assert_called()
    assert obs_update.call_count == 1
