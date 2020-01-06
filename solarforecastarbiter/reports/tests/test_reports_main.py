from pathlib import Path
import re
import shutil


import pandas as pd
import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.io import api
from solarforecastarbiter.reports import template, main


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


def test_get_data(mock_data, report_objects):
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


@pytest.mark.skipif(shutil.which('pandoc') is None,
                    reason='Pandoc can not be found')
def test_full_render(mock_data, report_objects):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = \
        report_objects
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    raw_report = main.create_raw_report_from_data(report, data)
    report_md = main.render_raw_report(raw_report)
    body = template.report_md_to_html(report_md)
    full_report = template.full_html(body)
    # at least one non whitespace character in body, usually caused
    # by pandoc version error
    assert re.search(
        r'<body>(.*\S.*)</body>', full_report, re.S) is not None
    with open('bokeh_report.html', 'w') as f:
        f.write(full_report)


@pytest.mark.skipif(shutil.which('pandoc') is None,
                    reason='Pandoc can not be found')
def test_all_categories_render(mock_data, report_objects):
    # Create report using template but with all categories
    report, observation, forecast_0, forecast_1, _, _ = report_objects
    all_report = datamodel.Report(
        name=report.name,
        start=report.start,
        end=report.end,
        forecast_observations=report.forecast_observations,
        metrics=("mae", "rmse", "mbe"),
        categories=list(datamodel.ALLOWED_CATEGORIES.keys()),
        report_id=report.report_id,
        filters=report.filters
    )
    session = api.APISession('nope')
    data = main.get_data_for_report(session, all_report)
    raw_report = main.create_raw_report_from_data(all_report, data)
    report_md = main.render_raw_report(raw_report)
    body = template.report_md_to_html(report_md)
    full_report = template.full_html(body)
    with open('bokeh_report.html', 'w') as f:
        f.write(full_report)
