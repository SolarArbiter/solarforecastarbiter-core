
import pandas as pd
# from pandas.util.testing import assert_series_equal

import pytest

from solarforecastarbiter.reference_forecasts import main, models
from solarforecastarbiter.conftest import requires_tables


# we'll need to do something better once the load_forecast function works
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


def check_out(out, expected, site_type):
    assert len(out) == 6
    for o, e in zip(out[0:5], expected):
        assert isinstance(o, pd.Series)
        # assert_series_equal(o, e)
    if site_type == 'powerplant':
        assert isinstance(out[5], pd.Series)
    elif site_type == 'site':
        assert out[5] is None


@pytest.mark.parametrize('model,load_forecast_return_value', [
    pytest.param(models.gfs_3hour_to_hourly_mean, load_forecast_return_value_3,
                 marks=requires_tables),
    pytest.param(models.gfs_hourly_to_hourly_mean,
                 load_forecast_return_value_3, marks=requires_tables),
    (models.hrrr_subhourly_to_hourly_mean, load_forecast_return_value_5),
    (models.hrrr_subhourly_to_subhourly_instantaneous,
        load_forecast_return_value_5),
    pytest.param(models.nam_cloud_cover_to_hourly_mean,
                 load_forecast_return_value_3, marks=requires_tables),
    (models.nam_to_hourly_instantaneous, load_forecast_return_value_3),
    pytest.param(models.rap_cloud_cover_to_hourly_mean,
                 load_forecast_return_value_3, marks=requires_tables),
    (models.rap_irrad_to_hourly_mean, load_forecast_return_value_5),
    (models.rap_to_instantaneous, load_forecast_return_value_5),
])
def test_run(model, load_forecast_return_value, site_powerplant_site_type,
             mocker):
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.models.load_forecast',
        return_value=load_forecast_return_value)
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.forecast.unmix_intervals',
        return_value=cloud_cover_exp)
    site, site_type = site_powerplant_site_type
    out = main.run(site, model, init_time, start, end)
    check_out(out, out_forecast_exp, site_type)
