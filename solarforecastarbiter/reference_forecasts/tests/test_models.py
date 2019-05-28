from functools import partial
from pathlib import Path
import types

import pandas as pd
# from pandas.util.testing import assert_series_equal

import pytest

from solarforecastarbiter.io import nwp
from solarforecastarbiter.reference_forecasts import models


latitude = 32.2
longitude = -110.9
elevation = 700
init_time = pd.Timestamp('20190515T0000Z')
start = pd.Timestamp('20190515T0700Z')
end_short = pd.Timestamp('20190515T1200Z')
# gfs is longer, but mixed intervals fails
end_long = pd.Timestamp('20190520T0000Z')

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
        latitude, longitude, elevation, init_time, start, end_long,
        load_forecast=LOAD_FORECAST)


def check_out(out, start, end, end_strict=True):
    # check times
    for o in out[0:5]:
        assert isinstance(o, pd.Series)
        assert o.index[0] == start
        if end_strict:
            assert o.index[-1] == end
        else:
            assert o.index[-1] <= end
    # check irradiance limits
    for o in out[0:3]:
        assert (o >= 0).all() and (o < 1300).all()
    # check temperature limits
    assert (out[3] > -40).all() and (out[3] < 60).all()
    # check wind speed limits
    assert (out[4] >= 0).all() and (out[4] < 60).all()
    # check resampling function
    assert isinstance(out[5], partial)
    assert isinstance(out[6], (types.FunctionType, partial))


@pytest.mark.parametrize('model', [
    models.gfs_quarter_deg_hourly_to_hourly_mean,
    models.hrrr_subhourly_to_hourly_mean,
    models.hrrr_subhourly_to_subhourly_instantaneous,
    models.nam_12km_cloud_cover_to_hourly_mean,
    models.nam_12km_hourly_to_hourly_instantaneous,
    models.rap_cloud_cover_to_hourly_mean,
])
@pytest.mark.parametrize('end,end_strict', [
    (end_short, True), (end_long, False)
])
def test_gfs_quarter_deg_hourly_to_hourly_mean(model, end, end_strict):
    out = model(
        latitude, longitude, elevation, init_time, start, end,
        load_forecast=LOAD_FORECAST)
    check_out(out, start, end, end_strict=end_strict)


@pytest.mark.parametrize('model', [
    'hrrr_hourly',
    'hrrr_subhourly',
    'gfs_0p25',
    'rap',
    'nam_12km'
])
def test_domain_limits(model):
    # test file has longitudes ranging from -110.50 to -110.35
    # at midlatitudes, a degree of longitude is approximately 85 km
    # a 10 degree difference is then 850 km.
    # nwp.load_forecast calls lat/lon look up with max dist of 500 km
    with pytest.raises(ValueError):
        LOAD_FORECAST(latitude, -120.5, init_time, start, end_short, model)
