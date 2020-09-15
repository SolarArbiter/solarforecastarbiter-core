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
def test__determine_stream_vars(ds, expected):
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


mock_data = [[{'timestamp': '2020-01-01T0000Z', 'a': 5}],
             [{'timestamp': '2020-01-01T0001Z', 'a': 6}],
             [{'timestamp': '2020-01-01T0000Z', 'b': 5},
              {'timestamp': '2020-01-01T0001Z', 'b': 6}]]


def test_fetch_mocked_data(mocker, mock_api):
    start = pd.Timestamp('20191231T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    return_data = (d for d in mock_data)
    other_fetch = mocker.patch(
        'solarforecastarbiter.io.fetch.arm.fetch_arm')

    def set_return(*args, **kwargs):
        df_data = next(return_data)
        return pd.DataFrame.from_records(df_data, index='timestamp')

    other_fetch.side_effect = set_return

    data = arm.fetch(mock_api, site_objects[0], start, end,
                     doe_arm_user_id='id', doe_arm_api_key='key')
    other_fetch.assert_called()
    expected = pd.DataFrame.from_records(
        [{'timestamp': '2020-01-01T0000Z', 'a': 5, 'b': 5},
         {'timestamp': '2020-01-01T0001Z', 'a': 6, 'b': 6}],
        index='timestamp')
    pd.testing.assert_frame_equal(data, expected)


def test_fetch_param_valueerror(mocker, mock_api):
    decode_ep = mocker.patch(
        'solarforecastarbiter.io.reference_observations.arm.common'
        '.decode_extra_parameters')
    decode_ep.side_effect = ValueError
    start = pd.Timestamp('20191231T0000Z')
    end = pd.Timestamp('20200102T0000Z')
    data = arm.fetch(mock_api, site_objects[0], start, end,
                     doe_arm_user_id='id', doe_arm_api_key='key')
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


def test_get_period_overlap_no_overlap():
    overlap = arm.get_period_overlap(
        pd.Timestamp('2020-01-01', tz='utc'),
        pd.Timestamp('2020-02-01', tz='utc'),
        pd.Timestamp('2020-02-02', tz='utc'),
        pd.Timestamp('2020-03-01', tz='utc'),
    )
    assert overlap is None


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
    ({'stream1': '2019-12-01/2020-02-01',
      'stream2': '2019-01-01/2019-12-01'},
     pd.Timestamp('2019-01-01', tz='utc'),
     pd.Timestamp('2020-02-01', tz='utc'),
     {'stream1': [pd.Timestamp('2019-12-01', tz='utc'),
                  pd.Timestamp('2020-02-01', tz='utc')],
      'stream2': [pd.Timestamp('2019-01-01', tz='utc'),
                  pd.Timestamp('2019-12-01', tz='utc')]}),
    ({'stream2': '2019-01-01/2019-12-01',
      'stream1': '2019-12-01/2020-02-01'},
     pd.Timestamp('2019-12-12', tz='utc'),
     pd.Timestamp('2020-02-01', tz='utc'),
     {'stream1': [pd.Timestamp('2019-12-12', tz='utc'),
                  pd.Timestamp('2020-02-01', tz='utc')]}),
    ({'stream2': '2019-01-01/2019-12-01',
      'stream1': '2019-12-01/2020-02-01'},
     pd.Timestamp('2019-01-01', tz='utc'),
     pd.Timestamp('2019-03-01', tz='utc'),
     {'stream2': [pd.Timestamp('2019-01-01', tz='utc'),
                  pd.Timestamp('2019-03-01', tz='utc')]}),
])
def test_find_stream_data_availability(stream_dict, start, end, expected):
    available_stream_dict = arm.find_stream_data_availability(
        stream_dict, start, end)
    assert available_stream_dict == expected


@pytest.mark.parametrize('stream_dict', [
    ({'stream1': '2019-12-01/2020-02-01',
      'stream2': '2019-01-01/2020-01-01'}),
    ({'stream1': '2019-12-01/2020-02-01',
      'stream2': '2019-01-01/2020-01-01',
      'stream3': '2020-01-20/2020-03-01'}),
])
def test_find_stream_data_availability_overlap(stream_dict):
    with pytest.raises(ValueError):
        arm.find_stream_data_availability(
            stream_dict,
            pd.Timestamp('2019-12-12', tz='utc'),
            pd.Timestamp('2020-02-01', tz='utc'))


@pytest.mark.parametrize('site,expected_params', [
    ({'extra_parameters': {
         'network_api_id': 'E11',
         'network_api_abbreviation': 'sgp'}
      },
     {'extra_parameters': {
         "network_api_abbreviation": "sgp",
         "attribution": "https://www.arm.gov/capabilities/vaps/qcrad",
         "network": "DOE ARM",
         "arm_site_id": "sgp",
         "network_api_id": "E11",
         "observation_interval_length": 1.0,
         "datastreams": {
             "met": "sgpmetE11.b1",
             "qcrad": {
                 "sgpqcrad1longE11.c2": "1995-06-30/2019-08-02",
                 "sgpqcrad1longE11.c1": "2019-08-02/now"
                }
            }
         }
      }), ({
         'extra_parameters': {
            'network_api_id': 'dne',
            'network_api_abbreviation': 'nsa'}
      }, {
        'extra_parameters': {
            'network_api_id': 'dne',
            'network_api_abbreviation': 'nsa'}
      })
])
def test_adjust_site_parameters(site, expected_params):
    adjusted = arm.adjust_site_parameters(site)
    adjusted_params = adjusted['extra_parameters']
    for k, v in adjusted_params.items():
        assert adjusted_params[k] == expected_params['extra_parameters'][k]


@pytest.mark.parametrize('stream_list,expected', [
    ([('stream1', (pd.Timestamp('2019-12-01'), pd.Timestamp('2020-02-01'))),
      ('stream2', (pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-01')))],
     False),
    ([('stream1', (pd.Timestamp('2019-12-01'), pd.Timestamp('2020-02-01'))),
      ('stream2', (pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-15')))],
     True),
    ([('stream1', (pd.Timestamp('2019-12-01'), pd.Timestamp('2020-02-01'))),
      ('stream2', (pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-01'))),
      ('stream3', (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-12-01')))],
     False),
    ([('stream1', (pd.Timestamp('2019-12-01'), pd.Timestamp('2020-02-01'))),
      ('stream2', (pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-15'))),
      ('stream3', (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-12-01')))],
     True),
])
def test_detect_stream_overlap(stream_list, expected):
    assert arm.detect_stream_overlap(stream_list) == expected
