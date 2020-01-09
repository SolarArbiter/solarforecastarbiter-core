import pytest
import jinja2

from solarforecastarbiter import datamodel
from solarforecastarbiter.reports import template


@pytest.fixture
def mocked_timeseries_plots(mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.reports.figures.timeseries_plots')
    mocked.return_value = ('<script></script>', '<div></div>')


@pytest.fixture
def mocked_timeseries_plots_exception(mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.reports.figures.timeseries_plots')
    mocked.side_effect = Exception


@pytest.fixture
def dash_url():
    return 'https://solarforecastarbiter.url'


@pytest.fixture(params=[True, False])
def with_series(request):
    return request.param


@pytest.fixture
def expected_kwargs(report_with_raw, dash_url):
    def fn(with_series):
        kwargs = {}
        kwargs['human_categories'] = datamodel.ALLOWED_CATEGORIES
        kwargs['human_metrics'] = datamodel.ALLOWED_METRICS
        kwargs['report'] = report_with_raw
        kwargs['dash_url'] = dash_url
        version = report_with_raw.raw_report.plots.bokeh_version
        kwargs['bokeh_version'] = version
        if with_series:
            kwargs['timeseries_script'] = '<script></script>'
            kwargs['timeseries_div'] = '<div></div>'
        return kwargs
    return fn


def test__get_render_kwargs_no_series(
        mocked_timeseries_plots, report_with_raw, dash_url, with_series,
        expected_kwargs):
    kwargs = template._get_render_kwargs(
        report_with_raw,
        dash_url,
        with_series
    )
    assert kwargs == expected_kwargs(with_series)


def test__get_render_kwargs_with_series_exception(
        report_with_raw, dash_url, mocked_timeseries_plots_exception):
    kwargs = template._get_render_kwargs(
        report_with_raw,
        dash_url,
        True
    )
    assert kwargs['timeseries_div'] == """<div class="alert alert-warning">
  <strong>Warning</strong> Failed to make timeseries and scatter plots
  from stored data. Try generating report again.
</div>"""
    assert kwargs['timeseries_script'] == ''


def test_get_template_and_kwargs(
        report_with_raw, dash_url, with_series, expected_kwargs,
        mocked_timeseries_plots):
    html_template, kwargs = template.get_template_and_kwargs(
        report_with_raw,
        dash_url,
        with_series,
        True
    )
    assert type(html_template) == jinja2.environment.Template
    assert kwargs == expected_kwargs(with_series)


def test_render_html_body_only(report_with_raw, dash_url, with_series,
                               mocked_timeseries_plots):
    rendered = template.render_html(
        report_with_raw, dash_url, with_series, True)
    assert rendered[1:23] == '<h1 id="report-title">'


def test_render_html_full_html(report_with_raw, dash_url, with_series,
                               mocked_timeseries_plots):
    rendered = template.render_html(
        report_with_raw, dash_url, with_series, False)
    assert rendered[:46] == '<!doctype html>\n<html lang="en" class="h-100">'
