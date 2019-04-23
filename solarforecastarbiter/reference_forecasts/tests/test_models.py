from functools import partial
import types

import pandas as pd
# from pandas.util.testing import assert_series_equal

import pytest

from solarforecastarbiter.reference_forecasts import models


# we'll need to do something better once the load_forecast function works
latitude = 32.2
longitude = -110.9
elevation = 700
init_time = pd.Timestamp('20190328T1200Z')
start = pd.Timestamp('20190328T1300Z')
end = pd.Timestamp('20190328T1400Z')

index_exp = pd.DatetimeIndex(start=start, end=end, freq='1h')
ghi_exp = pd.Series([0, 10.], index=index_exp)
dni_exp = pd.Series([0, 15.], index=index_exp)
dhi_exp = pd.Series([0, 9.], index=index_exp)
temp_air_exp = pd.Series([10, 11.], index=index_exp)
wind_speed_exp = pd.Series([0, 1.], index=index_exp)
cloud_cover_exp = pd.Series([100., 0.], index=index_exp)
load_forecast_return_value_3 = (cloud_cover_exp, temp_air_exp, wind_speed_exp)
load_forecast_return_value_5 = (
    ghi_exp, dni_exp, dhi_exp, temp_air_exp, wind_speed_exp)
out_forecast_exp = (ghi_exp, dni_exp, dhi_exp, temp_air_exp, wind_speed_exp)


def check_out(out, expected):
    for o, e in zip(out[0:5], expected):
        assert isinstance(o, pd.Series)
        # assert_series_equal(o, e)
    assert isinstance(out[5], partial)
    assert isinstance(out[6], (types.FunctionType, partial))


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
@pytest.mark.parametrize('model', [
    models.gfs_quarter_deg_3hour_to_hourly_mean,
    models.gfs_quarter_deg_hourly_to_hourly_mean,
    models.gfs_quarter_deg_to_hourly_mean,
    models.hrrr_subhourly_to_hourly_mean,
    models.hrrr_subhourly_to_subhourly_instantaneous,
    models.nam_12km_cloud_cover_to_hourly_mean,
    models.nam_12km_hourly_to_hourly_instantaneous,
    models.rap_cloud_cover_to_hourly_mean,
    models.rap_ghi_to_hourly_mean,
    models.rap_ghi_to_instantaneous,
])
def test_default_load_forecast(model):
    out = models.hrrr_subhourly_to_subhourly_instantaneous(
        latitude, longitude, elevation, init_time, start, end)
    check_out(out, out_forecast_exp)


@pytest.mark.parametrize('model,load_forecast_return_value', [
    pytest.param(
        models.gfs_quarter_deg_3hour_to_hourly_mean,
        load_forecast_return_value_3),
    pytest.param(
        models.gfs_quarter_deg_hourly_to_hourly_mean,
        load_forecast_return_value_3),
    pytest.param(
        models.gfs_quarter_deg_to_hourly_mean,
        load_forecast_return_value_3,
        marks=[pytest.mark.xfail(strict=True, raises=NotImplementedError)]),
    (models.hrrr_subhourly_to_hourly_mean, load_forecast_return_value_5),
    (models.hrrr_subhourly_to_subhourly_instantaneous,
        load_forecast_return_value_5),
    pytest.param(models.nam_12km_cloud_cover_to_hourly_mean,
                 load_forecast_return_value_3),
    (models.nam_12km_hourly_to_hourly_instantaneous,
     load_forecast_return_value_3),
    pytest.param(models.rap_cloud_cover_to_hourly_mean,
                 load_forecast_return_value_3),
    (models.rap_ghi_to_hourly_mean, load_forecast_return_value_3),
    (models.rap_ghi_to_instantaneous, load_forecast_return_value_3),
])
def test_all_models(model, load_forecast_return_value, mocker):
    def load_forecast(*args, **kwargs):
        return load_forecast_return_value
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.forecast.unmix_intervals',
        return_value=cloud_cover_exp)
    out = model(
        latitude, longitude, elevation, init_time, start, end,
        load_forecast=load_forecast)
    check_out(out, out_forecast_exp)
