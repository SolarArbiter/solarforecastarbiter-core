import pytest
import re
import pandas as pd
from requests.exceptions import HTTPError

from solarforecastarbiter.datamodel import Site, Observation
from solarforecastarbiter.io import api
from solarforecastarbiter.io.reference_observations import eia


@pytest.fixture
def session(requests_mock):
    return api.APISession('')


@pytest.fixture
def site():
    return Site(
        name='CAISO',
        latitude=37.0,
        longitude=-120.0,
        elevation=0.0,
        timezone='Etc/GMT+7',
        site_id='',
        provider='',
        extra_parameters='{"network_api_id": "CISO-ALL", "attribution": "https://www.eia.gov/opendata/", "network": "EIA", "network_api_abbreviation": "eia", "observation_interval_length": 1}',  # noqa: E501
    )


@pytest.fixture
def site_no_extra():
    return Site(
        name='CAISO',
        latitude=37.0,
        longitude=-120.0,
        elevation=0.0,
        timezone='Etc/GMT+7',
        site_id='',
        provider='',
        extra_parameters='',
    )


@pytest.fixture
def log(mocker):
    log = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'eia.logger')
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
    eia.initialize_site_observations(session, site)
    assert status.called


@pytest.fixture
def mock_obs_creation(mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'create_observation')
    return mocked


@pytest.fixture
def mock_obs_creation_error(mocker, mock_obs_creation):
    mock_obs_creation.side_effect = HTTPError(
        response=mocker.Mock(response=mocker.Mock(text="eror")))


def test_initialize_site_obs(mock_api, mock_obs_creation, site):
    eia.initialize_site_observations(mock_api, site)
    mock_obs_creation.assert_called()


def test_initialize_site_obs_http_error(
        log, mock_api, mock_obs_creation_error, site):
    eia.initialize_site_observations(mock_api, site)
    assert log.error.call_count == 1
    assert log.debug.call_count == 1


def test_fetch(mocker, session, site):
    status = mocker.patch(
        'solarforecastarbiter.io.fetch.eia.get_eia_data'
    )
    start = pd.Timestamp('2020-01-01T0000Z')
    end = pd.Timestamp('2020-01-02T0000Z')
    api_key = 'bananasmoothie'

    index = pd.date_range(start, end, freq="1h")
    data = {"net_load": range(len(index))}
    df = pd.DataFrame(index=index, data=data)
    status.return_value = df

    out = eia.fetch(session, site, start, end, eia_api_key=api_key)
    assert status.called
    assert not out.empty


def test_fetch_empty(log, mocker, session, site):
    status = mocker.patch(
        'solarforecastarbiter.io.fetch.eia.get_eia_data'
    )
    start = pd.Timestamp('2020-01-01T0000Z')
    end = pd.Timestamp('2020-01-02T0000Z')
    api_key = 'bananasmoothie'
    status.return_value = pd.DataFrame()
    out = eia.fetch(session, site, start, end, eia_api_key=api_key)
    assert status.called
    assert log.warning.call_count == 1
    assert out.empty


def test_fetch_fail(mocker, session, site_no_extra):
    start = pd.Timestamp('2020-01-01T0000Z')
    end = pd.Timestamp('2020-01-02T0000Z')
    api_key = 'bananasmoothie'
    out = eia.fetch(session, site_no_extra, start, end, eia_api_key=api_key)
    assert out.empty


def test_initialize_site_forecasts(mocker, session, site, mock_list_sites):

    obs = Observation(
        name='Net Load',
        variable='net_load',
        site=site,
        interval_label='ending',
        interval_value_type='interval_mean',
        interval_length=pd.Timedelta('1h'),
        uncertainty=0.0,
    )

    mocker.patch('solarforecastarbiter.io.api.APISession.list_observations',
                 return_value=[obs])
    mocker.patch('solarforecastarbiter.io.api.APISession.list_forecasts')
    mocker.patch('solarforecastarbiter.io.api.APISession.'
                 'list_probabilistic_forecasts')
    status = mocker.patch(
        'solarforecastarbiter.io.api.APISession.create_forecast')
    eia.initialize_site_forecasts(session, site)
    assert status.called


@pytest.fixture
def mock_eia_creds(mocker):
    mocker.patch.dict('os.environ', {'EIA_API_KEY': 'fake_key'})


def test_update_observation_data(mocker, session, site, mock_eia_creds):
    obs_update = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'update_site_observations')
    start = pd.Timestamp('20200101T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    eia.update_observation_data(session, [site], [], start, end)
    obs_update.assert_called()
    assert obs_update.call_count == 1


def test_update_observation_data_no_creds(session, site):
    start = pd.Timestamp('20200101T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    with pytest.raises(KeyError) as e:
        eia.update_observation_data(session, [site], [], start, end)
    assert 'environment variable' in str(e.value)
