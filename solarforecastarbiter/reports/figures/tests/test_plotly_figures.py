import base64
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
def report_with_raw_xy(report_dict, raw_report_xy):
    report_dict['raw_report'] = raw_report_xy(True)
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
    def fn(obs_name, fx_name, agg=False, fxspec=None):
        if agg:
            obspec = datamodel.Aggregate
        else:
            obspec = datamodel.Observation
        fxobs = mocker.Mock()
        obs = mocker.Mock(spec=obspec)
        obs.name = obs_name
        fx = mocker.Mock(spec=fxspec)
        fx.name = fx_name
        fxobs.forecast = fx
        fxobs.data_object = obs
        return fxobs
    return fn


@pytest.mark.parametrize('obs_name,fx_name,fx_name_expected', [
    ('obs', 'fx', 'fx'),
    ('name', 'name', 'name Forecast')
])
def test__fx_name(fxobs_name_mock, obs_name, fx_name, fx_name_expected):
    fxobs = fxobs_name_mock(obs_name, fx_name)
    fx_name = figures._fx_name(fxobs.forecast, fxobs.data_object)
    assert fx_name == fx_name_expected


@pytest.mark.parametrize('axis,fx_name_expected', [
    ('x', 'fx name Prob(x <= 10.0 MW)'),
    ('y', 'fx name Prob(f <= x) = 10.0%')
])
def test__fx_name_prob(axis, fx_name_expected, fxobs_name_mock):
    fxobs = fxobs_name_mock(
        'obs name', 'fx name',
        fxspec=datamodel.ProbabilisticForecastConstantValue)
    fxobs.forecast.constant_value = 10.
    fxobs.forecast.axis = axis
    fxobs.forecast.units = 'MW'
    fx_name = figures._fx_name(fxobs.forecast, fxobs.data_object)
    assert fx_name == fx_name_expected


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
    if shutil.which('xvfb-run') is None:  # pragma: no cover
        pytest.skip('xvfb-run must be on PATH to make SVGs')
    values = list(range(5))
    fig = graph_objects.Figure(data=graph_objects.Scatter(x=values, y=values))
    svg = figures.output_svg(fig)
    assert svg.startswith('<svg')
    assert svg.endswith('</svg>')
    assert not logger.error.called


def test_output_pdf_with_plotly_figure(mocker):
    logger = mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.logger')
    if shutil.which('orca') is None:  # pragma: no cover
        pytest.skip('orca must be on PATH to make PDFs')
    if shutil.which('xvfb-run') is None:  # pragma: no cover
        pytest.skip('xvfb-run must be on PATH to make PDFs')
    values = list(range(5))
    fig = graph_objects.Figure(data=graph_objects.Scatter(x=values, y=values))
    pdf = figures.output_pdf(fig)
    pdf_bytes = base64.a85decode(pdf)
    assert pdf_bytes.startswith(b'%PDF-')
    assert pdf_bytes.rstrip(b'\n').endswith(b'%%EOF')
    assert not logger.error.called


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


def test_output_pdf_with_plotly_figure_no_orca(mocker, remove_orca):
    logger = mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.logger')
    values = list(range(5))
    fig = graph_objects.Figure(data=graph_objects.Scatter(x=values, y=values))
    pdf = figures.output_pdf(fig)
    pdf_bytes = base64.a85decode(pdf)
    assert pdf_bytes.startswith(b'%PDF-')
    assert pdf_bytes.rstrip(b'\n').endswith(b'%%EOF')
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
    plot_spec = out.to_dict()
    layout = plot_spec['layout']
    assert 'automargin' not in layout['xaxis']
    assert 'tickangle' not in layout['xaxis']
    assert layout['height'] == figures.PLOT_LAYOUT_DEFAULTS['height']


def test_bar_no_metric(metric_dataframe):
    out = figures.bar(metric_dataframe, 'rmse')
    assert isinstance(out, graph_objects.Figure)


def test_bar_empty_df(metric_dataframe):
    df = pd.DataFrame(columns=metric_dataframe.columns)
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
    df = pd.DataFrame(columns=metric_dataframe.columns)
    out = figures.bar_subdivisions(df, 'hour', 's')
    assert isinstance(out, dict)
    assert len(out) == 0


@pytest.mark.parametrize('input_str,expected', [
    ('not long', 'not long'),
    ('pretty long name and stuff', 'pretty long name and<br>stuff'),
    ('long word in middlemiddlemiddlemiddle of it',
     'long word in<br>middlemiddlemiddlemiddle<br>of it'),
])
def test_legend_text(input_str, expected):
    assert figures._legend_text(input_str) == expected


ts_df_data = [
    {'timestamp': '2020-01-01T00:00Z',
     'value': 5},
    {'timestamp': '2020-01-01T00:05Z',
     'value': 5},
    {'timestamp': '2020-01-01T00:10Z',
     'value': 5},
    {'timestamp': '2020-01-01T00:25Z',
     'value': 5},
    {'timestamp': '2020-01-01T00:30Z',
     'value': 5},
]


def test_fill_timeseries():
    data = pd.DataFrame(ts_df_data)
    data = data.set_index('timestamp')
    filled = figures._fill_timeseries(data, np.timedelta64(5, 'm'))
    assert filled.index.size == 7
    assert pd.isnull(filled.iloc[-4].value)
    assert pd.isnull(filled.iloc[-3].value)


def test_fill_timeseries_empty():
    data = pd.DataFrame()
    filled = figures._fill_timeseries(data, np.timedelta64(5, 'm'))
    assert filled.empty


meta_df_data = [
    {
        'pair_index': 0,
        'observation_name': 'obs one',
        'forecast_name': 'fx one',
        'observation_hash': str(hash('obs one')),
        'forecast_hash': str(hash('fx one')),
        'interval_label': 'beginning',
        'interval_length': np.timedelta64(1, 'm'),
        'observation_color': '#abc',
        'forecast_type': None,
        'axis': None,
        'constant_value': None
    }, {
        'pair_index': 1,
        'observation_name': 'obs two',
        'forecast_name': 'fx two',
        'observation_hash': str(hash('obs two')),
        'forecast_hash': str(hash('fx two')),
        'interval_label': 'beginning',
        'interval_length': np.timedelta64(5, 'm'),
        'observation_color': '#ccc',
        'forecast_type': None,
        'axis': None,
        'constant_value': None
    },
]


@pytest.fixture(params=[0, 1])
def meta_entries(request):
    return meta_df_data[request.param]


@pytest.mark.parametrize('hash_key', [
    'observation_hash', 'forecast_hash'
])
def test_extract_metadata(hash_key, meta_entries):
    meta_df = pd.DataFrame(meta_df_data)
    extracted = figures._extract_metadata_from_df(
        meta_df, meta_entries[hash_key], hash_key)
    for k, v in extracted.items():
        assert meta_entries[k] == v


def test_construct_timeseries_dataframe_timeseries(report_with_raw):
    ts_df, _ = figures.construct_timeseries_dataframe(report_with_raw)
    assert str(ts_df.index.tz) == report_with_raw.raw_report.timezone
    assert (ts_df.count() == 288).all()
    assert list(np.unique(ts_df['pair_index'])) == [0, 1, 2]
    assert (ts_df['observation_values'] == 100).all()
    assert (ts_df['forecast_values'] == 100).all()


def test_timeseries_plots(report_with_raw):
    ts_spec, scatter_spec, ts_prob_spec, inc_dist = figures.timeseries_plots(
        report_with_raw)
    assert isinstance(ts_spec, str)
    assert isinstance(scatter_spec, str)
    assert ts_prob_spec is None
    assert not inc_dist


def test_timeseries_plots_xy(report_with_raw_xy):
    ts_spec, scatter_spec, ts_prob_spec, inc_dist = figures.timeseries_plots(
        report_with_raw_xy)
    assert isinstance(ts_spec, str)
    assert isinstance(scatter_spec, str)
    assert isinstance(ts_prob_spec, str)
    assert inc_dist


@pytest.fixture
def report_with_raw_xy_asymmetric_cv(report_dict, raw_report_xy):
    raw = raw_report_xy(True)
    pfxobs = raw.processed_forecasts_observations
    fx_cvs = pfxobs[0].original.forecast.constant_values
    cv_1 = pfxobs[0].replace(
        original=pfxobs[0].original.replace(
            forecast=pfxobs[0].original.forecast.replace(
                constant_values=(
                    fx_cvs[0],
                    fx_cvs[1],
                    fx_cvs[2].replace(constant_value=85.0))
            ),
        ),
        forecast_values=pfxobs[0].forecast_values.rename(
            columns={'75.0': '85.0'})
    )
    raw = raw.replace(
        processed_forecasts_observations=(cv_1, pfxobs[1], pfxobs[2])
    )
    report_dict['raw_report'] = raw
    return datamodel.Report.from_dict(report_dict)


def test_probabilistic_plotting_asymmetric_cv(
        report_with_raw_xy_asymmetric_cv):
    ts_spec, scatter_spec, ts_prob_spec, inc_dist = figures.timeseries_plots(
        report_with_raw_xy_asymmetric_cv)
    assert isinstance(ts_spec, str)
    assert isinstance(scatter_spec, str)
    assert isinstance(ts_prob_spec, str)
    assert inc_dist


@pytest.mark.parametrize('new_name,tickangle,height', [
    ('some what long name used when test',
     45, 250 + 34 * figures.X_LABEL_HEIGHT_FACTOR),
    ('very long name used when test very long names with plot tick angle',
     90, 250 + 66 * figures.X_LABEL_HEIGHT_FACTOR),
])
def test_bar_height_tick_adjustment(
        metric_dataframe, new_name, tickangle, height):
    metric_dataframe['abbrev'] = new_name
    out = figures.bar(metric_dataframe, 'mae')
    assert isinstance(out, graph_objects.Figure)
    assert out.layout.height == height
    assert out.layout.xaxis.tickangle == tickangle
    assert out.layout.xaxis.automargin
