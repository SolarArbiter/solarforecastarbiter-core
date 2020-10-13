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
    pytest.param(models.rap_ghi_to_instantaneous, marks=xfail_g2sub)
])
def test_default_load_forecast_failures(model):
    model(
        latitude, longitude, elevation, init_time, start, end_long,
        load_forecast=LOAD_FORECAST)


def check_out(out, start, end, end_strict=True):
    # check times
    for o in out[0:5]:
        assert isinstance(o, (pd.Series, pd.DataFrame))
        assert o.index[0] == start
        if end_strict:
            assert o.index[-1] == end
        else:
            assert o.index[-1] <= end
    # check irradiance limits
    for o in out[0:3]:
        assert (o >= 0).all().all() and (o < 1300).all().all()
    # check temperature limits
    assert (out[3] > -40).all().all() and (out[3] < 60).all().all()
    # check wind speed limits
    assert (out[4] >= 0).all().all() and (out[4] < 60).all().all()
    # check resampling function
    assert isinstance(out[5], (types.FunctionType, partial))
    assert isinstance(out[6], (types.FunctionType, partial))


@pytest.mark.parametrize('interval_label', ['beginning', 'ending'])
@pytest.mark.parametrize('model', [
    models.gfs_quarter_deg_hourly_to_hourly_mean,
    models.gfs_quarter_deg_to_hourly_mean,
    models.hrrr_subhourly_to_hourly_mean,
    models.nam_12km_cloud_cover_to_hourly_mean,
    models.rap_cloud_cover_to_hourly_mean,
    models.gefs_half_deg_to_hourly_mean
])
@pytest.mark.parametrize('end,end_strict', [
    (end_short, True), (end_long, False)
])
def test_mean_models(model, end, end_strict, interval_label):
    out = model(
        latitude, longitude, elevation, init_time, start, end,
        interval_label, load_forecast=LOAD_FORECAST)
    if interval_label == 'beginning':
        start_fx_expected = start
        end_fx_expected = pd.Timestamp(end) - pd.Timedelta('5min')
    elif interval_label == 'ending':
        start_fx_expected = pd.Timestamp(start) + pd.Timedelta('5min')
        end_fx_expected = end
    check_out(out, start_fx_expected, end_fx_expected, end_strict=end_strict)


@pytest.mark.parametrize('model, end_fx_expected', [
    (models.nam_12km_hourly_to_hourly_instantaneous, '20190515T1100Z'),
    (models.hrrr_subhourly_to_subhourly_instantaneous, '20190515T1145Z')
])
def test_instant_models(model, end_fx_expected):
    out = model(
        latitude, longitude, elevation, init_time, start, end_short,
        'instant', load_forecast=LOAD_FORECAST)
    check_out(out, start, pd.Timestamp(end_fx_expected), end_strict=True)


@pytest.mark.parametrize('latitude', [32.0, 32.25, 32.5])
@pytest.mark.parametrize('longitude', [
    -111.0, -110.75, -110.5, -110.25, -110.0])
@pytest.mark.parametrize('start,end,init_time', [
    ('20190515T0100Z', '20190520T0000Z', '20190515T0000Z'),
    ('20190520T0300Z', '20190522T0000Z', '20190515T0000Z'),
    ('20190525T1200Z', '20190531T0000Z', '20190515T0000Z'),
    ('20190715T0100Z', '20190720T0000Z', '20190715T0000Z'),
    ('20190716T0800Z', '20190717T0700Z', '20190715T0000Z'),
])
def test_gfs_quarter_deg_to_hourly_mean(latitude, longitude, start, end,
                                        init_time):
    # Separate test because date ranges are tricky due to GFS output
    # switching from hourly to 3 hourly to 12 hourly.
    # Also ensures that all data is valid at all times (unmixing cloud
    # cover once caused a problem due to rounding beyond our control).
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    init_time = pd.Timestamp(init_time)
    out = models.gfs_quarter_deg_to_hourly_mean(
        latitude, longitude, elevation, init_time, start, end,
        'beginning', load_forecast=LOAD_FORECAST)
    # account for beginning interval label
    end_fx_expected = pd.Timestamp(end) - pd.Timedelta('5min')
    check_out(out, start, end_fx_expected, end_strict=True)


@pytest.mark.parametrize('start,end,init_time', [
    ('20190515T0100Z', '20190520T0000Z', '20190515T0000Z'),
    ('20190520T0300Z', '20190522T0000Z', '20190515T0000Z'),
    ('20190525T1200Z', '20190531T0000Z', '20190515T0000Z'),
    ('20190515T0300Z', '20190531T0000Z', '20190515T0000Z'),
    ('20201003T0100Z', '20201017T0000Z', '20201003T0000Z'),
])
def test_gefs_half_deg_to_hourly_mean(start, end, init_time):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    init_time = pd.Timestamp(init_time)
    out = models.gefs_half_deg_to_hourly_mean(
        latitude, longitude, elevation, init_time, start, end,
        'beginning', load_forecast=LOAD_FORECAST)
    end_fx_expected = pd.Timestamp(end) - pd.Timedelta('5min')
    check_out(out, start, end_fx_expected, end_strict=True)


@pytest.mark.parametrize('model', [
    'hrrr_hourly',
    'hrrr_subhourly',
    'gfs_0p25',
    'rap',
    'nam_12km',
    'gefs_c00',
    'gefs_p01'
])
def test_domain_limits(model):
    # test file has longitudes ranging from -110.50 to -110.35
    # at midlatitudes, a degree of longitude is approximately 85 km
    # a 10 degree difference is then 850 km.
    # nwp.load_forecast calls lat/lon look up with max dist of 500 km
    with pytest.raises(ValueError):
        LOAD_FORECAST(latitude, -120.5, init_time, start, end_short, model)


@pytest.mark.parametrize('model,exp', [
    (models.gfs_quarter_deg_hourly_to_hourly_mean, 'gfs_0p25'),
    (models.gfs_quarter_deg_to_hourly_mean, 'gfs_0p25'),
    (models.hrrr_subhourly_to_hourly_mean, 'hrrr_subhourly'),
    (models.hrrr_subhourly_to_subhourly_instantaneous, 'hrrr_subhourly'),
    (models.nam_12km_cloud_cover_to_hourly_mean, 'nam_12km'),
    (models.nam_12km_hourly_to_hourly_instantaneous, 'nam_12km'),
    (models.rap_cloud_cover_to_hourly_mean, 'rap'),
    (models.rap_ghi_to_instantaneous, 'rap'),
    (models.gefs_half_deg_to_hourly_mean, 'gefs')
])
def test_get_nwp_model(model, exp):
    assert models.get_nwp_model(model) == exp


@pytest.mark.parametrize('end,end_ceil', [
    ('00Z', '00Z'), ('01Z', '06Z')
])
def test_adjust_gfs_start_end_end(end, end_ceil):
    start = pd.Timestamp('20190101 00Z')
    end = pd.Timestamp(f'20190101 {end}')
    end_ceil = pd.Timestamp(f'20190101 {end_ceil}')
    _, end_ceil_out = models._adjust_gfs_start_end(start, end)
    assert end_ceil_out == end_ceil


@pytest.mark.parametrize('start,start_floor', [
    ('01Z', '01Z'), ('06Z', '01Z'), ('07Z', '07Z')
])
def test_adjust_gfs_start_end_start(start, start_floor):
    end = pd.Timestamp('20190102 00Z')
    start = pd.Timestamp(f'20190101 {start}')
    start_floor = pd.Timestamp(f'20190101 {start_floor}')
    start_floor_out, _ = models._adjust_gfs_start_end(start, end)
    assert start_floor_out == start_floor
