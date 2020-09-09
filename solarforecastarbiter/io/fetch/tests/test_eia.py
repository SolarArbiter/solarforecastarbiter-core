import os
import inspect
import pandas as pd
from solarforecastarbiter.io.fetch import eia


TEST_DATA_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
GOOD_FILE = os.path.join(TEST_DATA_DIR, 'data', 'eia_caiso.json')
BAD_FILE = os.path.join(TEST_DATA_DIR, 'data', 'eia_caiso_bad.json')


def test_get_eia_data(requests_mock):
    with open(GOOD_FILE) as f:
        content = f.read()

    requests_mock.register_uri(
        "GET",
        "https://api.eia.gov/series/?api_key=fake_key&series_id=EBA.CISO-ALL.D.H&start=20200601T00Z&end=20200602T00Z",   # NOQA
        content=content.encode(),
    )

    series_id = "EBA.CISO-ALL.D.H"
    start = pd.Timestamp("2020-06-01 00:00Z")
    end = pd.Timestamp("2020-06-02 00:00Z")
    api_key = "fake_key"

    df = eia.get_eia_data(series_id, api_key, start, end)
    assert len(df) == 25
    assert df.index[0] == start
    assert df.index[-1] == end


def test_get_eia_data_bad(requests_mock):
    with open(BAD_FILE) as f:
        content = f.read()

    requests_mock.register_uri(
        "GET",
        "https://api.eia.gov/series/?api_key=fake_key&series_id=EBA.CISO-ALL.D.H&start=20200601T00Z&end=20200602T00Z",   # NOQA
        content=content.encode(),
    )

    series_id = "EBA.CISO-ALL.D.H"
    start = pd.Timestamp("2020-06-01 00:00Z")
    end = pd.Timestamp("2020-06-02 00:00Z")
    api_key = "fake_key"

    df = eia.get_eia_data(series_id, api_key, start, end)
    assert len(df) == 25
    assert df.index[0] == start
    assert df.index[-1] == end
    assert len(df.dropna()) == 22
