from pathlib import Path


import pandas as pd
import pytest


from solarforecastarbiter.io import api
from solarforecastarbiter.reports import main


@pytest.fixture()
def _test_data(report_objects):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = \
        report_objects
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
        aggregate.aggregate_id: pd.read_csv(
            base / 'observation_values.csv', index_col='timestamp',
            parse_dates=True),
        forecast_agg.forecast_id: pd.read_csv(
            base / 'forecast_0_values.csv', header=None, parse_dates=True,
            names=['timestamp', 'value'], index_col='timestamp')['value'],
    }
    return data


@pytest.fixture()
def mock_data(mocker, _test_data):
    def get_data(id_, start, end, interval_label=None):
        return _test_data[id_].loc[start:end]

    get_forecast_values = mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_forecast_values',
        side_effect=get_data)
    get_observation_values = mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        side_effect=get_data)
    get_aggregate_values = mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_aggregate_values',
        side_effect=get_data)

    return get_forecast_values, get_observation_values, get_aggregate_values


def test_get_data_for_report(mock_data, report_objects):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = \
        report_objects
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    assert isinstance(data[observation], pd.DataFrame)
    assert isinstance(data[forecast_0], pd.Series)
    assert isinstance(data[forecast_1], pd.Series)
    assert isinstance(data[aggregate], pd.DataFrame)
    assert isinstance(data[forecast_agg], pd.Series)
    get_forecast_values, get_observation_values, get_aggregate_values = \
        mock_data
    assert get_forecast_values.call_count == 3
    assert get_observation_values.call_count == 1
    assert get_aggregate_values.call_count == 1


def test_create_metadata():
    pass


def test_get_version():
    pass


def test_infer_timezone():
    pass


def test_listhandler():
    pass


def test_hijack_loggers():
    pass


def test_create_raw_report_from_data():
    pass


def test_compute_report():
    pass


def test_report_to_html_body():
    pass
