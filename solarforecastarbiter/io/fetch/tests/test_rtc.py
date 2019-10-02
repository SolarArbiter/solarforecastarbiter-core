import inspect
import os


import pandas as pd
import pytest
import json


from solarforecastarbiter.io.fetch import rtc

TEST_DATA_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
SYSTEM_FILE = os.path.join(TEST_DATA_DIR, 'data', 'doe_rtc_system.json')
WEATHER_FILE = os.path.join(TEST_DATA_DIR, 'data', 'doe_rtc_weather.json')


start_date = pd.Timestamp('2019-01-22')
end_date = pd.Timestamp('2019-01-23')


def mocked_get(url):
    class MockedResponse:
        def __init__(self, json_data):
            self.json_data = json_data

        def json(self):
            return self.json_data

    if 'weather' in url:
        with open(WEATHER_FILE) as f:
            return MockedResponse(json.load(f))
    if 'system' in url:
        with open(SYSTEM_FILE) as f:
            return MockedResponse(json.load(f))


@pytest.mark.parametrize('loc,data_type,start,end,api_key,expected', [
    ('plant_a', 'weather', start_date, end_date, 'bogus_key',
     'https://pv-dashboard.sandia.gov/api/v1.0/location/plant_a/data/weather/start/2019-01-22/end/2019-01-23/key/bogus_key'), # NOQA
])
def test_request_doe_rtc(loc, data_type, start, end, api_key, expected,
                         mocker):
    mock_get = mocker.patch(
        'solarforecastarbiter.io.fetch.rtc.requests.get',
        side_effect=mocked_get)
    rtc.request_doe_rtc_data(loc, data_type, start, end, api_key)
    mock_get.assert_called_with(expected)


@pytest.mark.parametrize('location,api_key,start,end', [
    ('plant_a', 'bogus_key', start_date, end_date),
])
def test_fetch_doe_rtc(location, api_key, start, end, mocker):
    mocker.patch('solarforecastarbiter.io.fetch.rtc.requests.get',
                 side_effect=mocked_get)
    data = rtc.fetch_doe_rtc(location, api_key, start, end)
    assert data.index[0] == pd.Timestamp('2019-01-23T00:01:00', freq='T')
    assert data.index[-1] == pd.Timestamp('2019-01-23T23:59:00', freq='T')
    assert 'AmbientTemp_weather' in data.columns
    assert 'AmbientTemp_system' in data.columns
    assert 'TmStamp' not in data.columns
