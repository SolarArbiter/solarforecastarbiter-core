import pandas as pd
import pytest
from requests.exceptions import HTTPError

from solarforecastarbiter.io.reference_observations import arm
from solarforecastarbiter.io.reference_observations.tests.conftest import (
    site_objects,
)


@pytest.fixture
def log(mocker):
    log = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'arm.logger')
    return log


@pytest.mark.parametrize('ds,expected', [
    ('sgpqcradlong1', arm.DOE_ARM_SITE_VARIABLES['qcrad']),
    ('sgpmet', arm.DOE_ARM_SITE_VARIABLES['met']),
    ('badstream', []),
])
def test__detemrine_stream_vars(ds, expected):
    assert arm._determine_stream_vars(ds) == expected


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


def test_initialize_site_obs(mock_api, mock_obs_creation):
    arm.initialize_site_observations(mock_api, site_objects[0])
    mock_obs_creation.assert_called()


def test_initialize_site_obs_http_error(
        log, mock_api, mock_obs_creation_error):
    arm.initialize_site_observations(mock_api, site_objects[0])
    assert log.error.call_count == 6
    assert log.debug.call_count == 6


@pytest.fixture
def mock_arm_creds(mocker):
    mocker.patch.dict('os.environ', {'DOE_ARM_API_KEY': 'fake_key'})
    mocker.patch.dict('os.environ', {'DOE_ARM_USER_ID': 'user'})


def test_update_observation_data(mocker, mock_api, mock_arm_creds):
    obs_update = mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'update_site_observations')
    start = pd.Timestamp('20200101T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    arm.update_observation_data(mock_api, site_objects, [], start, end)
    obs_update.assert_called()
    assert obs_update.call_count == 1


@pytest.mark.parametrize('var,missing', [
    ('DOE_ARM_API_KEY', 'DOE_ARM_USER_ID'),
    ('DOE_ARM_USER_ID', 'DOE_ARM_API_KEY'),
])
def test_update_observation_data_no_creds(mocker, mock_api, var, missing):
    mocker.patch.dict('os.environ', {var: 'tesitng'})
    mocker.patch(
        'solarforecastarbiter.io.reference_observations.common.'
        'update_site_observations')
    start = pd.Timestamp('20200101T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    with pytest.raises(KeyError) as e:
        arm.update_observation_data(mock_api, site_objects, [], start, end)
    assert missing in str(e.value)


def test_fetch(mocker, mock_api):
    start = pd.Timestamp('20200101T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    other_fetch = mocker.patch(
        'solarforecastarbiter.io.fetch.arm.fetch_arm')
    other_fetch.return_value = pd.DataFrame()
    data = arm.fetch(mock_api, site_objects[0], start, end,
                     doe_arm_user_id='id', doe_arm_api_key='key')
    other_fetch.assert_called()
    assert data.empty


@pytest.mark.parametrize('reqstart,reqend,availstart,availend,estart,eend', [
    ('2020-01-01', '2020-02-01',
     '2019-01-01', '2021-02-01',
     '2020-01-01', '2020-02-01'),
    ('2020-01-15', '2020-01-18',
     '2020-01-01', '2020-02-01',
     '2020-01-15', '2020-01-18'),
    ('2020-01-15', '2020-03-18',
     '2020-01-01', '2020-02-01',
     '2020-01-15', '2020-02-01'),
    ('2019-01-15', '2020-03-18',
     '2020-01-01', '2020-02-01',
     '2020-01-01', '2020-02-01'),
    ('2019-01-15', '2020-01-18',
     '2020-01-01', '2020-02-01',
     '2020-01-01', '2020-01-18'),
])
def test_get_period_overlap(
        reqstart, reqend, availstart, availend, estart, eend):
    start, end = arm.get_period_overlap(
        pd.Timestamp(reqstart),
        pd.Timestamp(reqend),
        pd.Timestamp(availstart),
        pd.Timestamp(availend),
    )
    assert start == pd.Timestamp(estart)
    assert end == pd.Timestamp(eend)


@pytest.mark.parametrize('range_string,exstart,exend', [
    ('2019-01-01/2021-01-01',
     pd.Timestamp('2019-01-01', tz='utc'),
     pd.Timestamp('2021-01-01', tz='utc')),
    ('1994-11-13/2020-01-01',
     pd.Timestamp('1994-11-13', tz='utc'),
     pd.Timestamp('2020-01-01', tz='utc')),
    ])
def test_parse_iso_date_range(range_string, exstart, exend):
    start, end = arm.parse_iso_date_range(range_string)
    assert start == exstart
    assert end == exend


def test_parse_iso_date_range_now():
    start, end = arm.parse_iso_date_range('2019-01-01/now')
    assert start == pd.Timestamp('2019-01-01', tz='utc')
    expected_end = pd.Timestamp.utcnow()
    assert end.month == expected_end.month
    assert end.day == expected_end.day
    assert end.year == expected_end.year
    assert end.hour == expected_end.hour


@pytest.mark.parametrize('stream_dict,start,end,expected', [
    ({'stream1': '2020-01-01/2020-02-01',
      'stream2': '2019-01-01/2020-01-01'},
     pd.Timestamp('2019-01-01', tz='utc'),
     pd.Timestamp('2020-02-01', tz='utc'),
     {'stream1': [pd.Timestamp('2020-01-01', tz='utc'),
                  pd.Timestamp('2020-02-01', tz='utc')],
      'stream2': [pd.Timestamp('2019-01-01', tz='utc'),
                  pd.Timestamp('2020-01-01', tz='utc')]}),
])
def test_find_stream_data_availability(stream_dict, start, end, expected):
    available_stream_dict = arm.find_stream_data_availability(
        stream_dict, start, end)
    assert available_stream_dict == expected
