import inspect
import os


from netCDF4 import Dataset
import pandas as pd
import pytest
from requests.exceptions import ChunkedEncodingError


from solarforecastarbiter.io.fetch import arm


TEST_DATA_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
MET_FILE = os.path.join(TEST_DATA_DIR, 'data',
                        'sgpmetE13.b1.20190122.000000.cdf')
IRRAD_FILE = os.path.join(TEST_DATA_DIR, 'data',
                          'sgpqcrad1longC1.c1.20190122.000000.cdf')


test_datastreams = ['ds_1', 'ds_2']
start_date = pd.Timestamp('2019-01-22')
end_date = pd.Timestamp('2019-01-23')


def filenames(*args):
    if args[2] == 'ds_1':
        return ['irrad']
    if args[2] == 'ds_2':
        return ['weather']
    return []


def request_file(*args):
    if args[2] == 'irrad':
        return Dataset(IRRAD_FILE)
    if args[2] == 'weather':
        return Dataset(MET_FILE)


def test_format_date():
    date = pd.Timestamp('2019-01-23T01:01:01Z')
    assert arm.format_date(date) == '2019-01-23'


def mocked_request_get_files(*args, **kwargs):
    class Object:
        pass
    response = Object()
    response.text = '{"files": ["filename1", "filename2"]}'
    return response


@pytest.fixture
def api_key():
    return 'bogus_key'


@pytest.fixture
def user_id():
    return 'user_id'


@pytest.mark.parametrize('stream,variables,start,end', [
    (test_datastreams[0], ['down_short_hemisp',
     'not_real'], start_date, end_date),
    (test_datastreams[1], ['temp_mean'], start_date,
     end_date),
])
def test_fetch_arm(user_id, api_key, stream, variables, start, end, mocker):
    mocker.patch('solarforecastarbiter.io.fetch.arm.list_arm_filenames',
                 side_effect=filenames)
    mocker.patch('solarforecastarbiter.io.fetch.arm.retrieve_arm_dataset',
                 side_effect=request_file)
    data = arm.fetch_arm(user_id, api_key, stream, variables, start, end)
    assert variables[0] in data.columns


@pytest.mark.parametrize('stream,start,end', [
    ('datastream', start_date, end_date)
])
def test_request_file_lists(user_id, api_key, stream, start, end, mocker):
    mocked_get = mocker.patch('solarforecastarbiter.io.fetch.arm.requests.get',
                              side_effect=mocked_request_get_files)
    arm.list_arm_filenames(user_id, api_key, stream, start, end)
    mocked_get.assert_called_with(
        'https://adc.arm.gov/armlive/data/query',
        params={
            'user': f'{user_id}:{api_key}',
            'ds': 'datastream',
            'start': '2019-01-22',
            'end': '2019-01-23',
            'wt': 'json'
        })


def test_request_arm_file(user_id, api_key, mocker):
    mocked_get = mocker.patch('solarforecastarbiter.io.fetch.arm.requests.get')
    arm.request_arm_file(user_id, api_key, 'sgpqcrad1longC1.c1.cdf')
    mocked_get.assert_called_with(
        arm.ARM_FILES_DOWNLOAD_URL,
        params={
            'user': f'{user_id}:{api_key}',
            'file': 'sgpqcrad1longC1.c1.cdf',
        })


def test_extract_arm_variables_exist(mocker):
    nc_file = request_file(None, None, 'irrad')
    extracted = arm.extract_arm_variables(nc_file,
                                          ['down_short_hemisp', 'nonexistent'])
    assert 'down_short_hemisp' in extracted.columns
    assert 'non-existent' not in extracted.columns


def test_extracted_arm_variables_empty(mocker):
    nc_file = request_file(None, None, 'irrad')
    extracted = arm.extract_arm_variables(nc_file,
                                          ['no', 'nein', 'ie', 'non'])
    assert extracted.empty


def test_no_files(user_id, api_key, mocker):
    mocker.patch('solarforecastarbiter.io.fetch.arm.list_arm_filenames',
                 side_effect=filenames)
    mocker.patch('solarforecastarbiter.io.fetch.arm.retrieve_arm_dataset',
                 side_effect=request_file)
    start = end = pd.Timestamp.now()+pd.Timedelta('1 days')
    arm_df = arm.fetch_arm(user_id, api_key, 'ds_no_files',
                           ['down_short_hemisp'], start, end)
    assert arm_df.empty


@pytest.mark.parametrize('num_failures', range(1, 5))
def test_request_arm_file_retries(mocker, num_failures):
    mocked_get = mocker.patch('solarforecastarbiter.io.fetch.arm.requests.get')
    return_values = (d for d in [0, num_failures])

    def get_response(*args, **kwargs):
        call_no = next(return_values)
        if call_no < num_failures:
            raise ChunkedEncodingError
        else:
            response = mocker.MagicMock()
            response.content = 'success'
            return response

    mocked_get.side_effect = get_response
    the_response = arm.request_arm_file('user', 'ley', 'filename')
    assert the_response == 'success'


def test_request_arm_file_failure_after_retries(mocker):
    mocked_get = mocker.patch('solarforecastarbiter.io.fetch.arm.requests.get')
    mocked_get.side_effect = ChunkedEncodingError
    with pytest.raises(ChunkedEncodingError):
        arm.request_arm_file('user', 'ley', 'filename')


def test_fetch_arm_request_file_failure(mocker):
    mocker.patch('solarforecastarbiter.io.fetch.arm.list_arm_filenames',
                 return_value=['afilename'])
    mocker.patch('solarforecastarbiter.io.fetch.arm.retrieve_arm_dataset',
                 side_effect=ChunkedEncodingError)
    mocked_log = mocker.patch('solarforecastarbiter.io.fetch.arm.logger')
    data = arm.fetch_arm('user', 'key', 'stream', ['ghi'], 'start', 'end')
    mocked_log.error.assert_called_with(
            f'Request failed for DOE ARM file afilename')
    assert data.empty
