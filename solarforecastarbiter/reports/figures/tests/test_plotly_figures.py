import shutil


from plotly import graph_objects
import numpy as np
import pandas as pd
import pytest


import solarforecastarbiter.reports.figures.plotly_figures as figures
from solarforecastarbiter import datamodel


@pytest.fixture
def report_with_raw(report_dict, raw_report):
    report_dict['raw_report'] = raw_report(True)
    report = datamodel.Report.from_dict(report_dict)
    return report


@pytest.fixture
def raw_report_pfxobs_values(raw_report):
    # Useful for testing the valid types for a ProcessedForecastObservation's
    # values fields which can be either pd.Series, str or None
    def replace_values(value):
        raw = raw_report(False)
        raw = raw.replace(
            processed_forecasts_observations=tuple(
                pfxobs.replace(
                    forecast_values=value,
                    observation_values=value,
                )
                for pfxobs in raw.processed_forecasts_observations
            )
        )
        return raw
    return replace_values


@pytest.fixture
def set_report_pfxobs_values(report_dict, raw_report_pfxobs_values):
    def set_pfxobs_values(value):
        report_dict['raw_report'] = raw_report_pfxobs_values(value)
        report = datamodel.Report.from_dict(report_dict)
        return report
    return set_pfxobs_values


def test_construct_metrics_dataframe(report_with_raw):
    report = report_with_raw
    metrics = report.raw_report.metrics
    df = figures.construct_metrics_dataframe(metrics)
    names = df['name']
    abbrev = df['abbrev']
    categories = df['category']
    metrics = df['metric']
    values = df['value']
    report_params = report.report_parameters

    expected_length = (len(report_params.metrics) *
                       len(report_params.categories) *
                       len(report_params.object_pairs))
    assert all([len(v) == expected_length for k, v in df.items()])

    original_names = [fxobs.forecast.name
                      for fxobs in report_params.object_pairs]
    assert np.all(
        names == np.repeat(
            np.array(original_names),
            len(report_params.metrics) * len(report_params.categories))
    )
    assert np.all(names == abbrev)

    assert np.all(
        metrics == np.tile(
            np.repeat(np.array(report_params.metrics, dtype=object),
                      len(report_params.categories)),
            len(report_params.object_pairs))
    )

    assert np.all(
        categories == np.tile(
            np.array(report_params.categories),
            len(report_params.metrics) * len(report_params.object_pairs))
    )

    # this could maybe use value variance, but asserting the dataframe process
    # did not mangle values for now
    assert (values == 2).all()


def test_construct_metrics_dataframe_with_rename(report_with_raw):
    metrics = report_with_raw.raw_report.metrics
    df = figures.construct_metrics_dataframe(metrics,
                                             rename=figures.abbreviate)
    report_params = report_with_raw.report_parameters
    original_names = [fxobs.forecast.name
                      for fxobs in report_params.object_pairs]
    abbreviated = list(map(figures.abbreviate, original_names))
    assert np.all(
        df['abbrev'] == np.repeat(
            np.array(abbreviated, dtype=object),
            len(report_params.metrics) * len(report_params.categories))
    )


def test_construct_metric_dataframe_no_values():
    # Iterative metrics datafame creation just builds an empty dataframe
    # with correct columns if no MetricResults are found in the metrics tuple
    df = figures.construct_metrics_dataframe(())
    assert df['index'].size == 0
    assert 'abbrev' in df


@pytest.fixture
def fxobs_name_mock(mocker):
    def fn(obs_name, fx_name, agg=False):
        if agg:
            obspec = datamodel.Aggregate
        else:
            obspec = datamodel.Observation
        fxobs = mocker.Mock()
        obs = mocker.Mock(spec=obspec)
        obs.name = obs_name
        fx = mocker.Mock()
        fx.name = fx_name
        fxobs.forecast = fx
        fxobs.data_object = obs
        return fxobs
    return fn


@pytest.mark.parametrize('y_min,y_max,pad,expected', [
    (1.0, 10.0, 0.5, (0.0, 5.0)),
    (-5.0, 10.0, 1.5, (-7.5, 15.0)),
    (-5.0, -1.0, 1.0, (-5.0, 0.0)),
    (-100.0, 100.0, 1.03, (-103.0, 103.0)),
])
def test_start_end(y_min, y_max, pad, expected):
    result = figures.calc_y_start_end(y_min, y_max, pad)
    assert result == expected


@pytest.mark.parametrize('arg,expected', [
    ('a', 'a'),
    ('abcd', 'abc.'),
    ('ABCDEFGHIJKLMNOP', 'ABCDEFGHIJKLMNOP'),
    ('University of Arizona OASIS Day Ahead GFS ghi',
     'Uni. of Ari. OASIS Day Ahe. GFS ghi')
])
def test_abbreviate(arg, expected):
    out = figures.abbreviate(arg)
    assert out == expected


def test_raw_report_plots(report_with_raw):
    metrics = report_with_raw.raw_report.metrics
    plots = figures.raw_report_plots(report_with_raw, metrics)
    assert plots is not None


def test_output_svg_with_plotly_figure(mocker):
    logger = mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.logger')
    if shutil.which('orca') is None:  # pragma: no cover
        pytest.skip('orca must be on PATH to make SVGs')
    values = list(range(5))
    fig = graph_objects.Figure(data=graph_objects.Scatter(x=values, y=values))
    svg = figures.output_svg(fig)
    assert svg.startswith('<svg')
    assert svg.endswith('</svg>')
    assert not logger.error.called


@pytest.fixture(scope='function')
def remove_orca():
    import plotly.io as pio
    pio.orca.config.executable = '/dev/null'


def test_output_svg_with_plotly_figure_no_orca(mocker, remove_orca):
    logger = mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.logger')
    values = list(range(5))
    fig = graph_objects.Figure(data=graph_objects.Scatter(x=values, y=values))
    svg = figures.output_svg(fig)
    assert svg.startswith('<svg')
    assert 'Unable' in svg
    assert svg.endswith('</svg>')
    assert logger.error.called


@pytest.fixture()
def metric_dataframe():
    return pd.DataFrame({
        'name': ['First', 'Next'],
        'abbrev': ['1st', 'N'],
        'category': ['hour', 'total'],
        'metric': ['mae', 'mae'],
        'value': [0.1, 10.3],
        'index': [14, 0]
    })


def test_bar(metric_dataframe):
    out = figures.bar(metric_dataframe, 'mae')
    assert isinstance(out, graph_objects.Figure)


def test_bar_no_metric(metric_dataframe):
    out = figures.bar(metric_dataframe, 'rmse')
    assert isinstance(out, graph_objects.Figure)


def test_bar_empty_df(metric_dataframe):
    df = pd.DataFrame({k: [] for k in metric_dataframe.columns})
    out = figures.bar(df, 's')
    assert isinstance(out, graph_objects.Figure)


def test_bar_subdivisions(metric_dataframe):
    out = figures.bar_subdivisions(metric_dataframe, 'hour', 'mae')
    assert isinstance(out, dict)
    assert len(out) == 1
    assert all([isinstance(v, graph_objects.Figure) for v in out.values()])


def test_bar_subdivisions_no_cat(metric_dataframe):
    out = figures.bar_subdivisions(metric_dataframe, 'date', 'mae')
    assert isinstance(out, dict)
    assert len(out) == 0


def test_bar_subdivisions_no_metric(metric_dataframe):
    out = figures.bar_subdivisions(metric_dataframe, 'hour', 'rmse')
    assert isinstance(out, dict)
    assert len(out) == 0


def test_bar_subdivisions_empty_df(metric_dataframe):
    df = pd.DataFrame({k: [] for k in metric_dataframe.columns})
    out = figures.bar_subdivisions(df, 'hour', 's')
    assert isinstance(out, dict)
    assert len(out) == 0
