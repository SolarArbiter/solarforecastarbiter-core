import shutil
import subprocess

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
import jinja2
from bokeh import __version__ as bokeh_version
from plotly import __version__ as plotly_version

from solarforecastarbiter import datamodel
from solarforecastarbiter.reports import template


expected_metrics_json = """[{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"mae","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"mae","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"mae","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"rmse","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"rmse","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"rmse","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"mbe","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"mbe","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"mbe","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"s","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"s","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"s","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"mae","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"mae","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"mae","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"rmse","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"rmse","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"rmse","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"mbe","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"mbe","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"mbe","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"s","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"s","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"s","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"mae","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"mae","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"mae","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"rmse","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"rmse","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"rmse","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"mbe","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"mbe","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"mbe","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"s","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"s","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"s","value":2,"index":1}]"""  # NOQA


@pytest.fixture
def mocked_timeseries_plots(mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.timeseries_plots')
    mocked.return_value = ('{}', '{}')


@pytest.fixture
def mocked_timeseries_plots_exception(mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.timeseries_plots')
    mocked.side_effect = Exception


@pytest.fixture
def dash_url():
    return 'https://solarforecastarbiter.url'


@pytest.fixture(params=[True, False])
def with_series(request):
    return request.param


@pytest.fixture
def expected_kwargs(dash_url):
    def fn(report, with_series, with_report=True):
        kwargs = {}
        kwargs['human_categories'] = datamodel.ALLOWED_CATEGORIES
        kwargs['human_metrics'] = datamodel.ALLOWED_METRICS
        kwargs['category_blurbs'] = datamodel.CATEGORY_BLURBS
        if with_report:
            kwargs['report'] = report
        if report.status == 'complete':
            kwargs['metrics_json'] = expected_metrics_json
        else:
            kwargs['metrics_json'] = '[]'
        kwargs['dash_url'] = dash_url
        kwargs['bokeh_version'] = bokeh_version
        kwargs['plotly_version'] = plotly_version
        if with_series:
            kwargs['timeseries_spec'] = '{}'
            kwargs['scatter_spec'] = '{}'
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
    exp = expected_kwargs(report_with_raw, with_series)
    assert kwargs == exp


def test__get_render_kwargs_pending(
        mocked_timeseries_plots, pending_report, dash_url,
        expected_kwargs, mocker):
    kwargs = template._get_render_kwargs(
        pending_report,
        dash_url,
        False
    )
    exp = expected_kwargs(pending_report, False)
    assert kwargs == exp


def test__get_render_kwargs_with_series_exception(
        report_with_raw, dash_url, mocked_timeseries_plots_exception):
    kwargs = template._get_render_kwargs(
        report_with_raw,
        dash_url,
        True
    )
    assert kwargs['timeseries_spec'] == '{}'
    assert kwargs['scatter_spec'] == '{}'


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
    assert kwargs == expected_kwargs(good_or_bad_report,
                                     with_series, False)


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


def test_build_metrics_json():
    pass


@pytest.mark.parametrize('val,expected', [
    ('<p> paragraph</p>', ' paragraph\n'),
    ('<em>italic</em>', '\\emph{italic}'),
    ('<code>nan</code>', '\\verb|nan|'),
    ('<b>bold</b>', '\\textbf{bold}'),
    ('<ol>\n<li>item one</li>\n</ol>',
     '\\begin{enumerate}\n\\item item one\n\n\\end{enumerate}'),
    ('<a href="tolink" class="what">stuff</a>', 'stuff'),
    (('<p>paragraph one <em>important</em> code here <code>null</code>'
      ' and more <b>bold</b><em> critical</em> <code>here</code></p>'
      ' <b>masbold</b>'),
     ('paragraph one \\emph{important} code here \\verb|null|'
      ' and more \\textbf{bold}\\emph{ critical} \\verb|here|\n'
     ' \\textbf{masbold}'))
])
def test_html_to_tex(val, expected):
    assert template._html_to_tex(val) == expected


def test_render_pdf(report_with_raw, dash_url):
    if shutil.which('pdflatex') is None:  # pragma: no cover
        pytest.skip('pdflatex must be on PATH to generate PDF reports')
    rendered = template.render_pdf(report_with_raw, dash_url)
    assert rendered.startswith(b'%PDF')


def test_render_pdf_special_chars(
        ac_power_observation_metadata, ac_power_forecast_metadata, dash_url,
        fail_pdf, preprocessing_result_types, report_metrics):
    if shutil.which('pdflatex') is None:  # pragma: no cover
        pytest.skip('pdflatex must be on PATH to generate PDF reports')
    quality_flag_filter = datamodel.QualityFlagFilter(
        (
            "USER FLAGGED",
        )
    )
    forecast = ac_power_forecast_metadata.replace(
        name="ac_power forecast (why,)  ()'-_,")
    observation = ac_power_observation_metadata.replace(
        name="ac_power observations  ()'-_,")
    fxobs = datamodel.ForecastObservation(forecast,
                                          observation)
    tz = 'America/Phoenix'
    start = pd.Timestamp('20190401 0000', tz=tz)
    end = pd.Timestamp('20190404 2359', tz=tz)
    report_params = datamodel.ReportParameters(
        name="NREL MIDC OASIS GHI Forecast Analysis  ()'-_,",
        start=start,
        end=end,
        object_pairs=(fxobs,),
        metrics=("mae", "rmse", "mbe", "s"),
        categories=("total", "date", "hour"),
        filters=(quality_flag_filter,)
    )
    report = datamodel.Report(
        report_id="56c67770-9832-11e9-a535-f4939feddd83",
        report_parameters=report_params
    )
    qflags = list(
        f.quality_flags for f in report.report_parameters.filters if
        isinstance(f, datamodel.QualityFlagFilter)
    )
    qflags = list(qflags[0])
    ser_index = pd.date_range(
        start, end,
        freq=to_offset(forecast.interval_length),
        name='timestamp')
    ser = pd.Series(
        np.repeat(100, len(ser_index)), name='value',
        index=ser_index)
    pfxobs = datamodel.ProcessedForecastObservation(
        forecast.name,
        fxobs,
        forecast.interval_value_type,
        forecast.interval_length,
        forecast.interval_label,
        valid_point_count=len(ser),
        validation_results=tuple(datamodel.ValidationResult(
            flag=f, count=0) for f in qflags),
        preprocessing_results=tuple(datamodel.PreprocessingResult(
            name=t, count=0) for t in preprocessing_result_types),
        forecast_values=ser,
        observation_values=ser
    )

    figs = datamodel.RawReportPlots(
        (
            datamodel.PlotlyReportFigure.from_dict(
                {
                    'name': 'mae tucson ac_power',
                    'spec': '{"data":[{"x":[1],"y":[1],"type":"bar"}]}',
                    'pdf': fail_pdf,
                    'figure_type': 'bar',
                    'category': 'total',
                    'metric': 'mae',
                    'figure_class': 'plotly',
                }
            ),), '4.5.3',
    )
    raw = datamodel.RawReport(
        generated_at=report.report_parameters.end,
        timezone=tz,
        plots=figs,
        metrics=report_metrics(report),
        processed_forecasts_observations=(pfxobs,),
        versions=(('test',  'test_with_underscore?'),),
        messages=(datamodel.ReportMessage(
            message="Failed to make metrics for ac_power forecast ()'-_,",
            step='', level='', function=''),))
    rr = report.replace(raw_report=raw)
    rendered = template.render_pdf(rr, dash_url)
    assert rendered.startswith(b'%PDF')


def test_render_pdf_not_settled(report_with_raw, dash_url):
    if shutil.which('pdflatex') is None:  # pragma: no cover
        pytest.skip('pdflatex must be on PATH to generate PDF reports')
    with pytest.raises(RuntimeError):
        template.render_pdf(report_with_raw, dash_url, 1)


def test_render_pdf_process_error(report_with_raw, dash_url, mocker):
    mocker.patch('solarforecastarbiter.reports.template.subprocess.run',
                 side_effect=subprocess.CalledProcessError(
                     cmd='', returncode=1))
    with pytest.raises(subprocess.CalledProcessError):
        template.render_pdf(report_with_raw, dash_url)
