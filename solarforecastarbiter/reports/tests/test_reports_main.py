import datetime
from pathlib import Path


import pandas as pd
import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.io import api
from solarforecastarbiter.reports import template, main


tz = 'America/Phoenix'
start = pd.Timestamp('20190401 0000', tz=tz)
end = pd.Timestamp('20190404 2359', tz=tz)


@pytest.fixture(scope='module')
def model_defs():
    site = datamodel.Site(
        name="NREL MIDC University of Arizona OASIS",
        latitude=32.22969,
        longitude=-110.95534,
        elevation=786.0,
        timezone="Etc/GMT+7",
        site_id="9f61b880-7e49-11e9-9624-0a580a8003e9",
        provider="Reference",
        extra_parameters='{"network": "NREL MIDC", "network_api_id": "UAT", "network_api_abbreviation": "UA OASIS", "observation_interval_length": 1}',  # NOQA
    )
    observation = datamodel.Observation(
        name="University of Arizona OASIS ghi",
        variable="ghi",
        interval_value_type="interval_mean",
        interval_length=pd.Timedelta("0 days 00:01:00"),
        interval_label="ending",
        site=site,
        uncertainty=0.0,
        observation_id="9f657636-7e49-11e9-b77f-0a580a8003e9",
        extra_parameters='{"network": "NREL MIDC", "network_api_id": "UAT", "network_api_abbreviation": "UA OASIS", "observation_interval_length": 1, "network_data_label": "Global Horiz (platform) [W/m^2]"}',  # NOQA
    )
    forecast_0 = datamodel.Forecast(
        name="0 Day GFS GHI",
        issue_time_of_day=datetime.time(7, 0),
        lead_time_to_start=pd.Timedelta("0 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi", site=site,
        forecast_id="da2bc386-8712-11e9-a1c7-0a580a8200ae",
        extra_parameters='{"model": "gfs_quarter_deg_to_hourly_mean"}',
    )
    forecast_1 = datamodel.Forecast(
        name="Day Ahead GFS GHI",
        issue_time_of_day=datetime.time(7, 0),
        lead_time_to_start=pd.Timedelta("1 days 00:00:00"),
        interval_length=pd.Timedelta("0 days 01:00:00"),
        run_length=pd.Timedelta("1 days 00:00:00"),
        interval_label="beginning",
        interval_value_type="interval_mean",
        variable="ghi",
        site=site,
        forecast_id="68a1c22c-87b5-11e9-bf88-0a580a8200ae",
        extra_parameters='{"model": "gfs_quarter_deg_to_hourly_mean"}',
    )
    fxobs0 = datamodel.ForecastObservation(forecast_0, observation)
    fxobs1 = datamodel.ForecastObservation(forecast_1, observation)
    quality_flag_filter = datamodel.QualityFlagFilter(
        [
            "USER FLAGGED",
            "NIGHTTIME",
            "LIMITS EXCEEDED",
            "STALE VALUES",
            "INTERPOLATED VALUES",
            "INCONSISTENT IRRADIANCE COMPONENTS",
        ]
    )
    report = datamodel.Report(
        name="NREL MIDC OASIS GHI Forecast Analysis",
        start=start,
        end=end,
        forecast_observations=(fxobs0, fxobs1),
        metrics=("mae", "rmse", "mbe"),
        filters=(quality_flag_filter,),
    )
    return report, observation, forecast_0, forecast_1


@pytest.fixture(scope='module')
def _test_data(model_defs):
    _, observation, forecast_0, forecast_1 = model_defs
    base = Path(__file__).resolve().parent
    data = {
        observation.observation_id: pd.read_csv(
            base / 'observation_values.csv', index_col='timestamp',
            parse_dates=True),
        forecast_0.forecast_id: pd.read_csv(
            base / 'forecast_0_values.csv', header=None, parse_dates=True,
            names=['timestamp', 'value'], index_col='timestamp')['value'],
        forecast_1.forecast_id: pd.read_csv(
            base / 'forecast_1_values.csv', header=None, parse_dates=True,
            names=['timestamp', 'value'], index_col='timestamp')['value'],
    }
    return data


@pytest.fixture()
def mock_data(mocker, _test_data):
    def get_data(id_, start, end):
        return _test_data[id_].loc[start:end]

    mocker.patch('solarforecastarbiter.io.api.APISession.get_forecast_values',
                 side_effect=get_data)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        side_effect=get_data)


def test_get_data(mock_data, model_defs):
    report, observation, forecast_0, forecast_1 = model_defs
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    assert isinstance(data[observation], pd.DataFrame)
    assert isinstance(data[forecast_0], pd.Series)
    assert isinstance(data[forecast_1], pd.Series)


def test_full_render(mock_data, model_defs):
    report, observation, forecast_0, forecast_1 = model_defs
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    raw_report = main.create_raw_report_from_data(report, data)
    report_md = main.render_raw_report(raw_report)
    body = template.prereport_to_html(report_md)
    full_report = template.full_html(body)
    with open('bokeh_report.html', 'w') as f:
        f.write(full_report)
