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
    def fn(with_series, with_report=True):
        kwargs = {}
        kwargs['human_categories'] = datamodel.ALLOWED_CATEGORIES
        kwargs['human_metrics'] = datamodel.ALLOWED_METRICS
        kwargs['category_blurbs'] = datamodel.CATEGORY_BLURBS
        if with_report:
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


def test__get_render_kwargs_pending(
        mocked_timeseries_plots, pending_report, dash_url,
        expected_kwargs, mocker):
    mocker.patch('solarforecastarbiter.reports.template.bokeh_version',
                 new='newest')
    kwargs = template._get_render_kwargs(
        pending_report,
        dash_url,
        False
    )
    exp = expected_kwargs(False)
    exp['bokeh_version'] = 'newest'
    exp['report'] = pending_report
    assert kwargs == exp


def test__get_render_kwargs_with_series_exception(
        report_with_raw, dash_url, mocked_timeseries_plots_exception):
    kwargs = template._get_render_kwargs(
        report_with_raw,
        dash_url,
        True
    )
    assert kwargs['timeseries_div'] == """<div class="alert alert-warning">
  <strong>Warning</strong> Failed to make timeseries and scatter plots
  from stored data.
</div>"""
    assert kwargs['timeseries_script'] == ''


@pytest.fixture(params=[0, 1, 2])
def good_or_bad_report(request, report_with_raw, failed_report,
                       pending_report):
    if request.param == 0:
        out = report_with_raw
    elif request.param == 1:
        out = failed_report
    elif request.param == 2:
        out = pending_report
    return out


@pytest.mark.parametrize('with_body', [True, False])
def test_get_template_and_kwargs(
        good_or_bad_report, dash_url, with_series, expected_kwargs,
        mocked_timeseries_plots, with_body):
    html_template, kwargs = template.get_template_and_kwargs(
        good_or_bad_report,
        dash_url,
        with_series,
        with_body
    )
    base = kwargs.pop('base_template')
    kwargs.pop('report')
    assert type(base) == jinja2.environment.Template
    assert type(html_template) == jinja2.environment.Template
    assert kwargs == expected_kwargs(with_series, False)


def test_get_template_and_kwargs_bad_status(
        report_with_raw, dash_url, mocked_timeseries_plots):
    inp = report_with_raw.replace(status='notokay')
    with pytest.raises(ValueError):
        template.get_template_and_kwargs(
            inp, dash_url, False, True)


def test_render_html_body_only(report_with_raw, dash_url, with_series,
                               mocked_timeseries_plots):
    rendered = template.render_html(
        report_with_raw, dash_url, with_series, True)
    assert rendered[:22] == '<h1 id="report-title">'


def test_render_html_full_html(report_with_raw, dash_url, with_series,
                               mocked_timeseries_plots):
    rendered = template.render_html(
        report_with_raw, dash_url, with_series, False)
    assert rendered[:46] == '<!doctype html>\n<html lang="en" class="h-100">'
