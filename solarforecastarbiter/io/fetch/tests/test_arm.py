import inspect
import os


from netCDF4 import Dataset
import pandas as pd
import pytest


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


def request_file(*args):
    if args[2] == 'irrad':
        return Dataset(IRRAD_FILE)
    if args[2] == 'weather':
        return Dataset(MET_FILE)


def test_format_date():
    date = pd.Timestamp('2019-01-23T01:01:01Z')
    assert arm.format_date(date) == '2019-01-23'


@pytest.mark.parametrize('user_id,api_key,datastream,variables,start,end', [
    ('user', 'bogus_key', test_datastreams[0], ['down_short_hemisp',
     'not_real'], start_date, end_date),
    ('user', 'bogus_key', test_datastreams[1], ['temp_mean'], start_date,
     end_date),
])
def test_fetch_arm(user_id, api_key, stream, variables, start, end, mocker):
    mocker.patch('solarforecastarbiter.io.fetch.arm.list_arm_filenames',
                 side_effect=filenames)
    mocker.patch('solarforecastarbiter.io.fetch.arm.retrieve_arm_dataset',
                 side_effect=request_file)
    data = arm.fetch_arm(user_id, api_key, stream, variables, start, end)
    assert variables[0] in data.columns
