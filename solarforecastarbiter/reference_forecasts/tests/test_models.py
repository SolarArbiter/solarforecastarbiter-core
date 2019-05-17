from functools import partial
from pathlib import Path

import pandas as pd
# from pandas.util.testing import assert_series_equal

import pytest

from solarforecastarbiter.io import nwp
from solarforecastarbiter.reference_forecasts import models


latitude = 32.2
longitude = -110.9
elevation = 700
init_time = pd.Timestamp('20190515T0000Z')
start = pd.Timestamp('20190515T0000Z')
end = pd.Timestamp('20190515T0100Z')

xfail_g2sub = pytest.mark.xfail(reason='ghi does not exist in g2sub')

BASE_PATH = Path(nwp.__file__).resolve().parents[0] / 'tests/data'
LOAD_FORECAST = partial(nwp.load_forecast, base_path=BASE_PATH)


@pytest.mark.parametrize('model', [
    pytest.param(models.gfs_quarter_deg_3hour_to_hourly_mean,
                 marks=pytest.mark.xfail(reason='gfs_3h not available')),
    pytest.param(models.gfs_quarter_deg_to_hourly_mean,
                 marks=pytest.mark.xfail(raises=NotImplementedError)),
    pytest.param(models.rap_ghi_to_hourly_mean, marks=xfail_g2sub),
    pytest.param(models.rap_ghi_to_instantaneous, marks=xfail_g2sub)
])
def test_default_load_forecast_failures(model):
    model(
        latitude, longitude, elevation, init_time, start, end,
        load_forecast=LOAD_FORECAST)


def test_gfs_quarter_deg_hourly_to_hourly_mean():
    end = pd.Timestamp('20190615T0000Z')
    out = models.gfs_quarter_deg_hourly_to_hourly_mean(
        latitude, longitude, elevation, init_time, start, end,
        load_forecast=LOAD_FORECAST)


def noop():
    models.hrrr_subhourly_to_hourly_mean,
    models.hrrr_subhourly_to_subhourly_instantaneous,
    models.nam_12km_cloud_cover_to_hourly_mean,
    models.nam_12km_hourly_to_hourly_instantaneous,
    models.rap_cloud_cover_to_hourly_mean,
