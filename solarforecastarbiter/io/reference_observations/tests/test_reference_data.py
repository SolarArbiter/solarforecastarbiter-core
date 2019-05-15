import pytest

from solarforecastarbiter.io.reference_observations import reference_data
from solarforecastarbiter.io.reference_observations.tests.conftest import(
    site_dicts,
    site_objects
)


@pytest.fixture
def log(mocker):
    logger = mocker.patch('solarforecastarbiter.io.reference_observations.'
                          'reference_data.logger')
    return logger


site_object_pairs = list(zip(site_dicts, site_objects))[:3]
@pytest.mark.parametrize('site_dict,expected', site_object_pairs)
def test_create_site(mock_api, site_dict, expected, mocker):
    mock_api.create_site.return_value = expected
    reference_data.create_site(mock_api, site_dict)
    mock_api.create_site.assert_called_with(expected)


def test_update_reference_observations(mocker, mock_api, log, start, end, networks):
    api = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'reference_data.get_apisession')
    api.return_value = mock_api
    surfrad = mocker.patch(
        'solarforecastarbiter.io.reference_observations.reference_data.surfrad.'
        'update_observation_data')
    solrad = mocker.patch(
        'solarforecastarbiter.io.reference_observations.solrad.'
        'update_observation_data')
    reference_data.update_reference_observations(start, end, networks)
    mock_api.list_sites.assert_called_once()
    mock_api.list_observations.assert_called_once()
    surfrad.assert_called()
    solrad.assert_called()
    assert log.info.call_count == 1
