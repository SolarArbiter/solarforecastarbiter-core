import pytest
import re
import pandas as pd

from solarforecastarbiter.datamodel import Site
from solarforecastarbiter.io import api
from solarforecastarbiter.io.reference_observations import bsrn


@pytest.fixture
def session(requests_mock):
    return api.APISession('')


@pytest.fixture
def site():
    return Site(
        name='WRMC BSRN NASA Langley Research Center',
        latitude=37.1048,
        longitude=-76.3872,
        elevation=3.0,
        timezone='Etc/GMT+5',
        site_id='',
        provider='',
        extra_parameters='{"network_api_id": "LRC", "attribution": "Driemel, A., Augustine, J., Behrens, K., Colle, S., Cox, C., Cuevas-Agull\\u00f3, E., Denn, F. M., Duprat, T., Fukuda, M., Grobe, H., Haeffelin, M., Hodges, G., Hyett, N., Ijima, O., Kallis, A., Knap, W., Kustov, V., Long, C. N., Longenecker, D., Lupi, A., Maturilli, M., Mimouni, M., Ntsangwane, L., Ogihara, H., Olano, X., Olefs, M., Omori, M., Passamani, L., Pereira, E. B., Schmith\\u00fcsen, H., Schumacher, S., Sieger, R., Tamlyn, J., Vogt, R., Vuilleumier, L., Xia, X., Ohmura, A., and K\\u00f6nig-Langlo, G.: Baseline Surface Radiation Network (BSRN): structure and data description (1992\\u20132017), Earth Syst. Sci. Data, 10, 1491-1501, doi:10.5194/essd-10-1491-2018, 2018.", "network": "WRMC BSRN", "network_api_abbreviation": "", "observation_interval_length": 1}',  # noqa: E501
    )


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
    bsrn.initialize_site_observations(session, site)
    assert status.called


def test_initialize_site_obs(mock_api, mocker, site):
    mocked = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'create_observation')

    bsrn.initialize_site_observations(mock_api, site)
    mocked.assert_called()


def test_fetch(mocker, session, site):
    start = pd.Timestamp('2020-01-01T0000Z')
    end = pd.Timestamp('2020-01-02T0000Z')
    with pytest.raises(NotImplementedError):
        bsrn.fetch(session, site, start, end)


def test_initialize_site_forecasts(mocker, session, site):
    mocked = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'create_forecasts')
    bsrn.initialize_site_forecasts(session, site)
    mocked.assert_called()


def test_update_observation_data(mocker, session, site):
    obs_update = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'update_site_observations')
    start = pd.Timestamp('20200101T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    bsrn.update_observation_data(session, [site], [], start, end)
    obs_update.assert_called()
    assert obs_update.call_count == 1
