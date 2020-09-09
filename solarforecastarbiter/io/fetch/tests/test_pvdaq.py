import inspect
import os


import pandas as pd


from solarforecastarbiter.io.fetch import pvdaq

TEST_DATA_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
SYSTEM_FILE = os.path.join(TEST_DATA_DIR, 'data', 'pvdaq_metadata.json')
DATA_FILE = os.path.join(TEST_DATA_DIR, 'data', 'pvdaq_2020_data.csv')
DATA_FILE_2019 = os.path.join(TEST_DATA_DIR, 'data', 'pvdaq_2019_data.csv')


def test_get_pvdaq_metadata(requests_mock):
    with open(SYSTEM_FILE) as f:
        content = f.read()

    requests_mock.register_uri(
        'GET', 'https://developer.nrel.gov/api/pvdaq/v3/sites.json',
        content=content.encode()
    )

    expected = [{
        'available_years': [2010,
                            2011,
                            2012,
                            2013,
                            2014,
                            2016,
                            2017,
                            2018,
                            2019,
                            2020],
        'comments': None,
        'confidential': False,
        'inverter_mfg': 'Fronius',
        'inverter_model': 'IG 4000',
        'module_mfg': 'Sharp',
        'module_model': 'ND-208U1',
        'module_tech': 2,
        'name_private': 'Nelson Home, Golden, CO, Array 1',
        'name_public': '[2] Residential #1a',
        'site_area': 22.82,
        'site_azimuth': 181.2,
        'site_elevation': 1675.0,
        'site_latitude': 39.7214,
        'site_longitude': 105.0972,
        'site_power': 2912.0,
        'site_tilt': 18.5,
        'system_id': 2}]
    out = pvdaq.get_pvdaq_metadata(2, 'doesntmatter')
    assert out == expected


def test_get_pvdaq_data(requests_mock):
    with open(DATA_FILE) as f:
        content = f.read()

    requests_mock.register_uri(
        'GET', 'https://developer.nrel.gov/api/pvdaq/v3/data_file',
        content=content.encode()
    )

    data = pvdaq.get_pvdaq_data(1276, 2020, 'fakekey')
    assert data.index[0] == pd.Timestamp('2020-01-01 00:00:00')
    assert data.index[-1] == pd.Timestamp('2020-01-02 23:45:00')
    some_cols = ['ac_current', 'ac_power', 'dc_power', 'wind_speed']
    for col in some_cols:
        assert col in data.columns


def test_get_pvdaq_data_2years(requests_mock):

    with open(DATA_FILE) as f:
        content = f.read()

    requests_mock.register_uri(
        'GET',
        'https://developer.nrel.gov/api/pvdaq/v3/data_file?api_key=fakekey&system_id=1276&year=2020',  # noqa: E501
        content=content.encode()
    )

    with open(DATA_FILE_2019) as f:
        content_2019 = f.read()

    requests_mock.register_uri(
        'GET',
        'https://developer.nrel.gov/api/pvdaq/v3/data_file?api_key=fakekey&system_id=1276&year=2019',  # noqa: E501
        content=content_2019.encode()
    )

    data = pvdaq.get_pvdaq_data(1276, [2019, 2020], 'fakekey')
    assert data.index[0] == pd.Timestamp('2019-01-01 00:00:00')
    assert data.index[-1] == pd.Timestamp('2020-01-02 23:45:00')
    some_cols = ['ac_current', 'ac_power', 'dc_power', 'wind_speed']
    for col in some_cols:
        assert col in data.columns
