import re

import pandas as pd
import pandas.testing as pdt

import pytest

from solarforecastarbiter.datamodel import (
    SolarPowerPlant, FixedTiltModelingParameters)
from solarforecastarbiter.io import api
from solarforecastarbiter.io.reference_observations import pvdaq


@pytest.fixture
def session(requests_mock):
    return api.APISession('')


@pytest.fixture
def site():
    return SolarPowerPlant(
        name='NREL x-Si #1',
        latitude=39.7406,
        longitude=-105.1774,
        elevation=1785.050170898438,
        timezone='Etc/GMT+7',
        site_id='',
        provider='',
        extra_parameters='{"network_api_id": 4, "attribution": "https://developer.nrel.gov/docs/solar/pvdaq-v3/", "network": "NREL PVDAQ", "network_api_abbreviation": "pvdaq", "observation_interval_length": 1, "inverter_mfg": "SMA", "inverter_model": "1800", "module_mfg": "Sanyo", "module_model": "HIP 200-BA3", "module_tech": "1"}',  # noqa: E501
        modeling_parameters=FixedTiltModelingParameters(
            ac_capacity=0.001,
            dc_capacity=0.001,
            temperature_coefficient=0.3,
            dc_loss_factor=0,
            ac_loss_factor=0,
            surface_tilt=40.0,
            surface_azimuth=180.0,
            tracking_type='fixed'))


@pytest.fixture
def site_no_extra():
    return SolarPowerPlant(
        name='NREL x-Si #1',
        latitude=39.7406,
        longitude=-105.1774,
        elevation=1785.050170898438,
        timezone='Etc/GMT+7',
        site_id='',
        provider='',
        extra_parameters='',
        modeling_parameters=FixedTiltModelingParameters(
            ac_capacity=0.001,
            dc_capacity=0.001,
            temperature_coefficient=0.3,
            dc_loss_factor=0,
            ac_loss_factor=0,
            surface_tilt=40.0,
            surface_azimuth=180.0,
            tracking_type='fixed'))


@pytest.fixture
def log(mocker):
    log = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'pvdaq.logger')
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
    pvdaq.initialize_site_observations(session, site)
    assert status.called


def test_initialize_site_observations_fail(session, site_no_extra, log):
    pvdaq.initialize_site_observations(session, site_no_extra)
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
    pvdaq.initialize_site_forecasts(session, site)
    assert status.called


def test_initialize_site_forecasts_fail(session, site_no_extra, log):
    pvdaq.initialize_site_forecasts(session, site_no_extra)
    assert log.warning.call_count == 1


def test_fetch(mocker, session, site):
    status = mocker.patch(
        'solarforecastarbiter.io.fetch.pvdaq.get_pvdaq_data'
    )
    start = pd.Timestamp('2020-01-01T0000Z')
    end = pd.Timestamp('2020-01-02T0000Z')
    api_key = 'nopethepope'
    pvdaq.fetch(session, site, start, end, nrel_pvdaq_api_key=api_key)
    assert status.called


def test_fetch_fail(session, site_no_extra):
    start = pd.Timestamp('2020-01-01T0000Z')
    end = pd.Timestamp('2020-01-02T0000Z')
    api_key = 'nopethepope'
    out = pvdaq.fetch(session, site_no_extra, start, end,
                      nrel_pvdaq_api_key=api_key)
    assert out.empty


def test_fetch_fail_except(session, site, mocker, log):
    start = pd.Timestamp('2020-01-01T0000Z')
    end = pd.Timestamp('2020-01-02T0000Z')
    api_key = 'nopethepope'
    status = mocker.patch(
        'solarforecastarbiter.io.fetch.pvdaq.get_pvdaq_data'
    )
    status.side_effect = Exception
    out = pvdaq.fetch(session, site, start, end,
                      nrel_pvdaq_api_key=api_key)
    assert log.warning.call_count == 1
    assert out.empty


def test_fetch_fail_nonexistenttime(session, site, mocker, log):
    site = site.replace(timezone='America/Denver')
    start = pd.Timestamp('2020-01-01T0000Z')
    end = pd.Timestamp('2020-01-02T0000Z')
    api_key = 'nopethepope'
    index = pd.DatetimeIndex(['2020-03-08 02:00:00'])
    df = pd.DataFrame({'ac_power': 0}, index=index)
    patch = mocker.patch('solarforecastarbiter.io.fetch.pvdaq.get_pvdaq_data')
    patch.return_value = df
    out = pvdaq.fetch(session, site, start, end,
                      nrel_pvdaq_api_key=api_key)
    assert log.warning.call_count == 1
    assert out.empty


@pytest.fixture
def mock_pvdaq_creds(mocker):
    mocker.patch.dict('os.environ', {'NREL_PVDAQ_API_KEY': 'fake_key'})


def test_update_observation_data(mocker, session, site, mock_pvdaq_creds):
    obs_update = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'update_site_observations')
    start = pd.Timestamp('20200101T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    pvdaq.update_observation_data(session, [site], [], start, end)
    obs_update.assert_called()
    assert obs_update.call_count == 1


def test_update_observation_data_no_creds(session, site):
    start = pd.Timestamp('20200101T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    with pytest.raises(KeyError) as e:
        pvdaq.update_observation_data(session, [site], [], start, end)
    assert 'environment variable' in str(e.value)


def test_watts_to_mw():
    columns = [
        'ac_power', 'inv1_ac_power', 'AC_power', 'AC_Power', 'DC_power',
        'wind_speed', 'bla h_power_blob']
    obs_df = pd.DataFrame(
        1.e6, columns=columns, index=[pd.Timestamp('20200514')])
    expected = pd.DataFrame(
        [[1., 1., 1., 1., 1., 1.e6, 1.]], columns=columns,
        index=[pd.Timestamp('20200514')])
    out = pvdaq._watts_to_mw(obs_df)
    pdt.assert_frame_equal(out, expected)
