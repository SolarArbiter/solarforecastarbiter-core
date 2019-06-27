from pathlib import Path
import shutil


import pandas as pd
import pytest


from solarforecastarbiter.io import api
from solarforecastarbiter.reports import template, main


@pytest.fixture(scope='module')
def _test_data(report_objects):
    _, observation, forecast_0, forecast_1 = report_objects
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


def test_get_data(mock_data, report_objects):
    report, observation, forecast_0, forecast_1 = report_objects
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    assert isinstance(data[observation], pd.DataFrame)
    assert isinstance(data[forecast_0], pd.Series)
    assert isinstance(data[forecast_1], pd.Series)


@pytest.mark.skipif(shutil.which('pandoc') is None,
                    reason='Pandoc can not be found')
def test_full_render(mock_data, report_objects):
    report, observation, forecast_0, forecast_1 = report_objects
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    raw_report = main.create_raw_report_from_data(report, data)
    report_md = main.render_raw_report(raw_report)
    body = template.report_md_to_html(report_md)
    full_report = template.full_html(body)
    with open('bokeh_report.html', 'w') as f:
        f.write(full_report)
