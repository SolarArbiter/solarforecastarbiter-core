import datetime as dt


import bokeh
import pandas as pd
import pytest


from solarforecastarbiter.plotting import timeseries


@pytest.mark.parametrize('times,exp', [
    ((pd.Timestamp('20190501T0000Z'), pd.Timestamp('20190501T1200Z')),
     'OBJECT 2019-05-01 00:00 to 2019-05-01 12:00 UTC'),
    ((dt.datetime(2019, 5, 1, 0, 0, tzinfo=dt.timezone.utc),
      dt.datetime(2019, 5, 1, 12, 0, tzinfo=dt.timezone.utc)),
     'OBJECT 2019-05-01 00:00 to 2019-05-01 12:00 UTC'),
    pytest.param((pd.Timestamp('20190501T0000'),
                  pd.Timestamp('20190501T1200Z')),
                 '', marks=pytest.mark.xfail(raises=ValueError)),
    pytest.param((pd.Timestamp('20190501T0000-0700'),
                  pd.Timestamp('20190501T1200Z')),
                 '', marks=pytest.mark.xfail(raises=ValueError)),
    pytest.param((dt.datetime.now(),
                  pd.Timestamp('20190501T1200Z')),
                 '', marks=pytest.mark.xfail(raises=ValueError))
])
def test_build_figure_title(times, exp):
    out = timeseries.build_figure_title(
        'OBJECT', *times)
    assert out == exp


TEST_FLAGS = pd.DataFrame({'USER FLAGGED': [False, False, True],
                           'NIGHTTIME': [True, False, False],
                           'CLEARSKY EXCEEDED': [None, None, None]},
                          index=pd.date_range(start='20190501T0000Z',
                                              freq='1h', periods=3))


def test_make_quality_bars():
    out = timeseries.make_quality_bars(
        TEST_FLAGS, 800,
        (TEST_FLAGS.index[0], TEST_FLAGS.index[-1]))
    assert isinstance(out, list)
    assert out[0].title.text == 'Quality Flags'
    assert len(out) == 2
    assert isinstance(out[0], bokeh.models.Plot)
    assert isinstance(out[1], bokeh.models.Plot)


@pytest.mark.parametrize('active', [True, False])
@pytest.mark.parametrize('addline', [True, False])
def test_add_hover_tool(addline, active):
    fig = bokeh.plotting.figure()
    if active:
        source = bokeh.models.ColumnDataSource({'active_flags': [[0]]})
    else:
        source = bokeh.models.ColumnDataSource()
    timeseries.add_hover_tool(fig, source, add_line=addline)
    assert len(fig.renderers) == int(addline)
    assert isinstance(fig.tools[-1], bokeh.models.HoverTool)
    if active:
        assert fig.tools[-1].tooltips[-1][0] == 'quality flags'
    else:
        assert len(fig.tools[-1].tooltips) == 2


@pytest.mark.parametrize('label', ['instant', 'beginning', 'ending'])
@pytest.mark.parametrize('source', [True, False])
def test_make_basic_timeseries(source, label):
    if source:
        source = bokeh.models.ColumnDataSource()
    else:
        source = None
    vals = pd.DataFrame(index=pd.date_range(start='now', freq='1min',
                                            periods=2, tz='UTC'))
    fig = timeseries.make_basic_timeseries(vals, 'OBJECT', 'ghi',
                                           label, 800, source)

    if label == 'instant':
        assert len(fig.renderers) == 1
    else:
        assert len(fig.renderers) == 2  # has hover line

    assert 'Time' in fig.xaxis[0].axis_label
    assert 'GHI' in fig.yaxis[0].axis_label
    assert 'OBJECT'in fig.title.text


def test_generate_forecast_figure(ac_power_forecast_metadata, forecast_values):
    # not great, just testing that script and div are returned
    out = timeseries.generate_forecast_figure(ac_power_forecast_metadata,
                                              forecast_values)
    assert '<script' in out[0]
    assert '<div' in out[1]


def test_generate_forecast_figure_empty(ac_power_forecast_metadata):
    assert timeseries.generate_forecast_figure(ac_power_forecast_metadata,
                                               pd.Series()) is None


def test_generate_observation_figure(validated_observation_values,
                                     ac_power_observation_metadata_label):
    out = timeseries.generate_observation_figure(
        ac_power_observation_metadata_label, validated_observation_values)
    assert '<script' in out[0]
    assert '<div' in out[1]


def test_generate_observation_figure_missing_data(
        validated_observation_values, ac_power_observation_metadata):
    validated_observation_values.iloc[1] = [None, None]
    out = timeseries.generate_observation_figure(
        ac_power_observation_metadata, validated_observation_values)
    assert '<script' in out[0]
    assert 'MISSING' in out[0]
    assert '<div' in out[1]


def test_generate_observation_figure_empty(ghi_observation_metadata):
    assert timeseries.generate_observation_figure(ghi_observation_metadata,
                                                  pd.DataFrame()) is None
