import datetime as dt
import pandas as pd
import pytest
from urllib import error

from pandas import Timestamp
from pandas.testing import assert_frame_equal

from solarforecastarbiter.io.reference_observations import srml
from solarforecastarbiter.datamodel import Site


test_site_dict = {
    "elevation": 595.0,
    "extra_parameters": "{\"network_api_id\": \"94040.0\", \"attribution\": \"Peterson, J., and Vignola, F., 2017: Structure of a Comprehensive Solar Radiation Dataset. Proceedings of the ASES National Solar Conference 2017. doi: 10.18086/solar.2017.07.02\", \"network\": \"UO SRML\", \"network_api_abbreviation\": \"AS\", \"observation_interval_length\": 1.0}",  # noqa
    "latitude": 42.19,
    "longitude": -122.7,
    "modeling_parameters": {
        "ac_capacity": 0.02,
        "ac_loss_factor": 0.0,
        "dc_capacity": 0.02,
        "dc_loss_factor": 0.0,
        "surface_azimuth": 180.0,
        "surface_tilt": 15.0,
        "temperature_coefficient": 0.3,
        "tracking_type": "fixed"
    },
    "name": "Ashland OR PV",
    "timezone": "Etc/GMT+8",
    "provider": "",
    "site_id": "",
}


test_site_object = Site.from_dict(test_site_dict)


@pytest.fixture()
def test_site():
    return test_site_object


srml_df = pd.DataFrame(
    {'ghi_1': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 0.0},
     'ghi_1_flag': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 11},
     'dni_1': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 0.0},
     'dni_1_flag': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 11},
     'dhi_1': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 0.0},
     'dhi_1_flag': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 11},
     '1161': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 0.0},
     '1161_flag': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 11},
     '5161': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 235.0},
     '5161_flag': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 11},
     '5162': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 250.0},
     '5162_flag': {Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 11},
     'wind_speed_3': {
         Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 0.1},
     'wind_speed_3_flag': {
         Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 11},
     'temp_air_1': {
         Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 2.1},
     'temp_air_1_flag': {
         Timestamp('2020-03-01 00:00:00-0800', tz='Etc/GMT+8'): 11}})


@pytest.fixture()
def test_data():
    return srml_df


@pytest.mark.parametrize('start,end,exp', [
    (dt.datetime(2019, 1, 10, tzinfo=dt.timezone.utc),
     dt.datetime(2019, 3, 11, tzinfo=dt.timezone.utc),
     [(2019, 1), (2019, 2), (2019, 3)]),
    (dt.datetime(2019, 11, 10, tzinfo=dt.timezone.utc),
     dt.datetime(2020, 2, 11, tzinfo=dt.timezone.utc),
     [(2019, 11), (2019, 12), (2020, 1), (2020, 2)]),
    (dt.datetime(2019, 11, 10, tzinfo=dt.timezone.utc),
     dt.datetime(2019, 11, 11, tzinfo=dt.timezone.utc),
     [(2019, 11)]),
    (dt.datetime(2019, 11, 10, tzinfo=dt.timezone.utc),
     dt.datetime(2021, 2, 11, tzinfo=dt.timezone.utc),
     [(2019, 11), (2019, 12), (2020, 1), (2020, 2), (2020, 3),
      (2020, 4), (2020, 5), (2020, 6), (2020, 7), (2020, 8),
      (2020, 9), (2020, 10), (2020, 11), (2020, 12), (2021, 1),
      (2021, 2)]),
    (dt.datetime(2019, 10, 1, tzinfo=dt.timezone.utc),
     dt.datetime(2019, 9, 1, tzinfo=dt.timezone.utc), []),
    (dt.datetime(2019, 10, 1, tzinfo=dt.timezone.utc),
     dt.datetime(2018, 11, 1, tzinfo=dt.timezone.utc), [])
])
def test_fetch(mocker, single_site, start, end, exp):
    rd = mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.request_data',
        return_value=None)
    srml.fetch('', single_site, start, end)
    year_month = [(ca[0][1], ca[0][2]) for ca in rd.call_args_list]
    assert year_month == exp


@pytest.mark.parametrize('start,end', [
    (dt.datetime(2019, 1, 10, tzinfo=dt.timezone.utc),
     dt.datetime(2020, 3, 11, 8, 1, tzinfo=dt.timezone.utc)),
])
def test_fetch_power_conversion(
        mocker, single_site, start, end, test_data):
    mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.request_data',
        return_value=test_data)
    data = srml.fetch('', single_site, start, end)
    assert not [col for col in data.columns if '_flag' in col]
    assert data['5161'].iloc[0] == 0.000235
    assert data['5162'].iloc[0] == 0.000250


def test_fetch_tz(single_site):
    start = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
    with pytest.raises(TypeError):
        srml.fetch('', single_site, start,
                   dt.datetime(2019, 2, 1))


@pytest.mark.parametrize('api_id,has_power', [
    (94040.0, True),
    (94291.0, False),
])
def test_adjust_parameters(api_id, has_power):
    site = srml.adjust_site_parameters({
        'extra_parameters': {
            'network_api_id': str(api_id),
            }
    })
    if has_power:
        modeling_params = site['modeling_parameters']
        assert modeling_params['ac_capacity'] == 0.02
        assert modeling_params['dc_capacity'] == 0.02
        assert modeling_params['ac_loss_factor'] == 0.0
        assert modeling_params['dc_loss_factor'] == 0.0
        assert modeling_params['surface_azimuth'] == 180.0
        assert modeling_params['surface_tilt'] == 15.0
        assert modeling_params['temperature_coefficient'] == 0.3
        assert modeling_params['tracking_type'] == 'fixed'
    else:
        assert 'modeling_parameters' not in site


def test_init_site_observations(
        mocker, test_data, test_site):
    mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.request_data',
        return_value=test_data)
    mock_create_obs = mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.common.'
        'create_observation')
    mock_chk_post = mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.common.'
        'check_and_post_observation')
    mock_api = mocker.MagicMock()
    srml.initialize_site_observations(mock_api, test_site)
    assert mock_create_obs.call_count == 5
    assert mock_chk_post.call_count == 2


def test_request_data(mocker, test_site, test_data):
    mocked_iotools = mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.iotools')
    mocked_iotools.read_srml_month_from_solardat = mocker.MagicMock(
        return_value=test_data)
    data = srml.request_data(test_site, 1, 1)
    assert_frame_equal(data, test_data)


@pytest.mark.parametrize('exception', [
    pd.errors.EmptyDataError,
    error.URLError,
])
def test_request_data_warnings(mocker, exception, test_site):
    mocked_iotools = mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.iotools')
    mocked_iotools.read_srml_month_from_solardat = mocker.MagicMock(
        side_effect=exception('error'))
    logger = mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.logger')
    data = srml.request_data(test_site, 1, 1)
    assert logger.warning.call_count == 3
    assert data is None


def test_initialize_site_forecasts(mocker, test_site):
    mock_create_fx = mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.common.'
        'create_forecasts')
    mock_api = mocker.MagicMock()
    srml.initialize_site_forecasts(mock_api, test_site)
    assert 'ac_power' in mock_create_fx.call_args[0][2]
    assert 'dc_power' in mock_create_fx.call_args[0][2]

    regular_site_dict = test_site_dict.copy()
    regular_site_dict.pop('modeling_parameters')
    reg_site = Site.from_dict(regular_site_dict)
    srml.initialize_site_forecasts(mock_api, reg_site)
    assert 'ac_power' not in mock_create_fx.call_args[0][2]
    assert 'dc_power' not in mock_create_fx.call_args[0][2]
