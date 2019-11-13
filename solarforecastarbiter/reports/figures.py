"""
Functions to make all of the figures for Solar Forecast Arbiter reports.
"""
from itertools import cycle
import textwrap

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.ranges import Range1d
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh import palettes

import pandas as pd
import numpy as np

from solarforecastarbiter.plotting.utils import (line_or_step,
                                                 format_variable_name)


PALETTE = (
    palettes.d3['Category20'][20][::2] + palettes.d3['Category20'][20][1::2])
_num_obs_colors = 3
OBS_PALETTE = palettes.grey(_num_obs_colors+1)[0:_num_obs_colors]  # drop white
OBS_PALETTE.reverse()
OBS_PALETTE_TD_RANGE = pd.timedelta_range(
    freq='10min', end='60min', periods=_num_obs_colors)

START_AT_ZER0 = ['mae', 'rmse']
START_OR_END_AT_ZER0 = ['mbe']


def construct_fx_obs_cds(fx_values, obs_values):
    """
    Construct a standardized Bokeh CDS for the plot functions.

    Parameters
    ----------
    fx_values : pandas.Series
    obs_values : pandas.Series

    Returns
    -------
    cds : bokeh.models.ColumnDataSource
        Keys are 'observation', 'forecast', and 'timestamp'.
        tz-aware input times are converted to tz-naive times in the
        input time zone.
    """
    data = pd.DataFrame({'observation': obs_values, 'forecast': fx_values})
    # drop tz info from localized times. GH164
    data = data.tz_localize(None)
    data = data.rename_axis('timestamp')
    cds = ColumnDataSource(data)
    return cds


def _obs_name(fx_obs):
    # TODO: add code to ensure obs names are unique
    name = fx_obs.observation.name
    if fx_obs.forecast.name == fx_obs.observation.name:
        name += ' Observation'
    return name


def _fx_name(fx_obs):
    # TODO: add code to ensure fx names are unique
    name = fx_obs.forecast.name
    if fx_obs.forecast.name == fx_obs.observation.name:
        name += ' Forecast'
    return name


def _obs_color(interval_length):
    idx = np.searchsorted(OBS_PALETTE_TD_RANGE, interval_length)
    obs_color = OBS_PALETTE[idx]
    return obs_color


def timeseries(fx_obs_cds, start, end, timezone='UTC'):
    """
    Timeseries plot of one or more forecasts and observations.

    Parameters
    ----------
    obs_fx_cds : list
        List of (ProcessedForecastObservation, ColumnDataSource) tuples.
        ColumnDataSource must have columns timestamp, observation,
        forecast.
    start : pandas.Timestamp
        Report start time
    end : pandas.Timestamp
        Report end time
    timezone : str
        Timezone consistent with the data in the obs_fx_cds.

    Returns
    -------
    fig : bokeh.plotting.figure
    """

    palette = cycle(PALETTE)

    fig = figure(
        sizing_mode='scale_width', plot_width=900, plot_height=300,
        x_range=(start, end), x_axis_type='datetime',
        tools='pan,xwheel_zoom,box_zoom,box_select,lasso_select,reset,save',
        name='timeseries')

    plotted_objects = []
    for proc_fx_obs, cds in fx_obs_cds:
        unique_obs = (
            proc_fx_obs.original.observation, proc_fx_obs.interval_value_type,
            proc_fx_obs.interval_length, proc_fx_obs. interval_label
        )
        unique_fx = (
            proc_fx_obs.original.forecast, proc_fx_obs.interval_value_type,
            proc_fx_obs.interval_length, proc_fx_obs. interval_label
        )
        if unique_obs in plotted_objects:
            pass
        else:
            plotted_objects.append(unique_obs)
            plot_method, plot_kwargs, hover_kwargs = line_or_step(
                proc_fx_obs.interval_label)
            name = _obs_name(proc_fx_obs.original)
            obs_color = _obs_color(proc_fx_obs.interval_length)
            getattr(fig, plot_method)(
                x='timestamp', y='observation', source=cds,
                color=obs_color, legend=name, **plot_kwargs)
        if unique_fx in plotted_objects:
            pass
        else:
            plotted_objects.append(unique_fx)
            plot_method, plot_kwargs, hover_kwargs = line_or_step(
                proc_fx_obs.interval_label)
            name = _fx_name(proc_fx_obs.original)
            getattr(fig, plot_method)(
                x='timestamp', y='forecast', source=cds,
                color=next(palette), legend=name, **plot_kwargs)

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    fig.xaxis.axis_label = f'Time ({timezone})'
    fig.yaxis.axis_label = format_variable_name(
        proc_fx_obs.original.forecast.variable)
    return fig


def _get_scatter_limits(fx_obs_cds):
    extremes = []
    for _, cds in fx_obs_cds:
        for kind in ('forecast', 'observation'):
            extremes.append(np.nanmin(cds.data[kind]))
            extremes.append(np.nanmax(cds.data[kind]))
    min_ = min(extremes)
    if min_ == np.nan:
        min_ = -999
    max_ = max(extremes)
    if max_ == np.nan:
        max_ = 999
    return min_, max_


def scatter(fx_obs_cds):
    """
    Scatter plot of one or more forecasts and observations.

    Parameters
    ----------
    obs_fx_cds : list
        List of (ProcessedForecastObservation, ColumnDataSource) tuples.
        ColumnDataSource must have columns timestamp, observation,
        forecast.

    Returns
    -------
    fig : bokeh.plotting.figure
    """
    xy_min, xy_max = _get_scatter_limits(fx_obs_cds)

    fig = figure(
        plot_width=450, plot_height=400, match_aspect=True,  # does not work?
        x_range=Range1d(xy_min, xy_max), y_range=Range1d(xy_min, xy_max),
        tools='pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save',
        name='scatter')

    kwargs = dict(size=6, line_color=None)

    palette = cycle(PALETTE)

    for proc_fx_obs, cds in fx_obs_cds:
        fig.scatter(
            x='observation', y='forecast', source=cds,
            fill_color=next(palette),
            legend=proc_fx_obs.original.forecast.name, **kwargs)

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    label = format_variable_name(proc_fx_obs.original.forecast.variable)
    fig.xaxis.axis_label = 'Observed ' + label
    fig.yaxis.axis_label = 'Forecast ' + label
    return fig


def construct_metrics_cds(metrics, kind, index='forecast', rename=None):
    """
    Possibly bad assumptions:
    * metrics contains keys: name, total, month, day, hour

    Parameters
    ----------
    metrics : list of metrics dicts
        Each metric dict is for a different forecast. Forecast name is
        specified by the name key.
    kind : str
        One of total, month, day, hour
    index : str
        Determines if the index is the array of metrics ('metric') or
        forecast ('forecast') names
    rename : function or None
        Function of one argument that is applied to each forecast name.

    Returns
    -------
    cds : bokeh.models.ColumnDataSource
    """
    if kind == 'total':
        if rename:
            f = rename
        else:
            def f(x): return x
        d = {f(m['name']): m[kind] for m in metrics}
        df = pd.DataFrame(d)
    df = df.rename_axis(index='metric', columns='forecast')
    if index == 'metric':
        pass
    elif index == 'forecast':
        df = df.T
    else:
        raise ValueError('index must be metric or forecast')
    cds = ColumnDataSource(df)
    return cds


def abbreviate(x, limit=3):
    # might need to add logic to ensure uniqueness
    # and/or enforce max length using textwrap.shorten
    components = x.split(' ')
    out_components = []
    for c in components:
        if len(c) <= limit:
            out = c
        elif c.upper() == c:
            # probably an acronym
            out = c
        else:
            out = f'{c[0:limit]}.'
        out_components.append(out)
    return ' '.join(out_components)


def construct_metrics_series(metrics, kind):
    """
    Contructs a series of metrics values with a MultiIndex.
    MultiIndex names are metric, forecast, kind.

    Parameters
    ----------
    metrics : list of metrics dicts
        Each metric dict is for a different forecast. Forecast name is
        specified by the name key.
    kind : str
        One of total, month, day, hour

    Returns
    -------
    pandas.Series
    """
    forecasts = []
    m_types = []
    m_indexes = []
    m_values = []
    # There is probably a more clever way to do this but
    # this seems the most straightforward to me
    for m in metrics:
        for col in m[kind]:
            for i, v in m[kind][col].items():
                forecasts.append(m['name'])
                m_types.append(col)
                m_indexes.append(i)
                m_values.append(v)
    index = pd.MultiIndex.from_arrays([forecasts, m_types, m_indexes],
                                      names=['forecast', 'metric', kind])
    metrics_series = pd.Series(m_values, index=index)
    metrics_series = metrics_series.reorder_levels(
        ('metric', 'forecast', kind))
    return metrics_series


def construct_metrics_cds2(metrics_series, metric):
    """
    Create a ColumnDataSource for a single metric (MAE, MBE, etc.) for
    many forecasts.

    Parameters
    ----------
    metrics_series : pd.Series
        Has a MultiIndex with levels 'forecast', 'metric', group
    metric : str
        e.g. MAE

    Returns
    -------
    cds : bokeh.models.ColumnDataSource
    """
    df = metrics_series.xs(metric, level='metric').unstack().T
    # unstack sorts alphabetically, so manually reorder the columns
    # to put them in the expected order. this ensures that plots are
    # created in the same order and thus with the same colors for each
    # forecast. GH issue 204, pull 244
    idx = pd.unique(metrics_series.index.get_level_values('forecast'))
    cds = ColumnDataSource(df[idx])
    return cds


def bar(cds, metric):
    """
    Create a bar graph comparing a single metric across forecasts.

    Parameters
    ----------
    cds : bokeh.models.ColumnDataSource
        Fields must be 'forecast' and the names of the metrics.

    Returns
    -------
    data_table : bokeh.widgets.DataTable
    """
    x_range = cds.data['forecast']
    palette = cycle(PALETTE)
    palette = [next(palette) for _ in x_range]
    # TODO: add units to title
    fig = figure(x_range=x_range, width=800, height=200, title=metric.upper())
    fig.vbar(x='forecast', top=metric, width=0.8, source=cds,
             line_color='white',
             fill_color=factor_cmap('forecast', palette, factors=x_range))
    fig.xgrid.grid_line_color = None
    if metric in START_AT_ZER0:
        fig.y_range.start = 0
    else:
        # TODO: add heavy 0 line
        pass
    tooltips = [
        ('forecast', '@forecast'),
        (metric.upper(), f'@{metric}'),
    ]
    hover = HoverTool(tooltips=tooltips, mode='vline')
    # more accurate would be if any single name is longer than each
    # name's allotted space. For example, never need to rotate labels
    # if forecasts are named A, B, C, D... but quickly need to rotate
    # if they have long names.
    if len(x_range) > 6:
        # pi/4 looks a lot better, but first tick label flows off chart
        # and I can't figure out how to add padding in bokeh
        fig.xaxis.major_label_orientation = np.pi/2
        fig.width = 800
        # add more height to figure so that the names can go somewhere.
        fig.height = 400
    fig.add_tools(hover)
    return fig


def bar_subdivisions(cds, kind, metric):
    """
    Create bar graphs comparing a single metric across subdivisions of
    time for multiple forecasts. e.g.

    Fx 1 MAE |
             |_________________
    Fx 2 MAE |
             |_________________
               year, month, day, or hour

    Parameters
    ----------
    cds : bokeh.models.ColumnDataSource
        Fields must be kind and the names of the forecasts
    kind : str
        One of year, month, day, hour

    Returns
    -------
    figs : tuple of figures
    """
    palette = cycle(PALETTE)
    tools = 'pan,xwheel_zoom,box_zoom,box_select,reset,save'
    fig_kwargs = dict(tools=tools)
    figs = []
    if kind == 'day':
        fig_kwargs['x_axis_type'] = 'datetime'
        width = 0.8 * pd.Timedelta('1day')
    else:
        width = 0.8

    y_min = min(d.min() for k, d in cds.data.items() if k != kind)
    y_max = max(d.max() for k, d in cds.data.items() if k != kind)
    pad_factor = 1.03
    y_max, y_min = pad_factor * y_max, pad_factor * y_min

    for num, field in enumerate(filter(lambda x: x != kind, cds.data)):
        title = field + ' ' + metric.upper()
        fig = figure(width=800, height=200, title=title, **fig_kwargs)
        fig.vbar(x=kind, top=field, width=width, source=cds,
                 line_color='white', fill_color=next(palette))
        fig.xgrid.grid_line_color = None
        fig.y_range.start = y_min
        fig.y_range.end = y_max
        if metric in START_AT_ZER0:
            fig.y_range.start = 0
        elif metric in START_OR_END_AT_ZER0:
            if y_max < 0:
                fig.y_range.start = y_min
                fig.y_range.end = 0
            if y_min > 0:
                fig.y_range.start = 0
                fig.y_range.end = y_max
        if num == 0:
            # add x_range to plots to link panning
            fig_kwargs['x_range'] = fig.x_range
        if kind == 'day':
            tooltips = [
                (kind, f'@{kind}{{%F}}'),
                (f'{field} {metric.upper()}', f'@{{{field}}}'),
            ]
            formatters = {kind: 'datetime'}
            hover_kwargs = dict(tooltips=tooltips, formatters=formatters)
        else:
            tooltips = [
                (kind, f'@{kind}'),
                (f'{field} {metric.upper()}', f'@{{{field}}}'),
            ]
            hover_kwargs = dict(tooltips=tooltips)
        hover = HoverTool(mode='vline', **hover_kwargs)
        fig.add_tools(hover)
        figs.append(fig)
    return figs


def nested_bar():
    raise NotImplementedError


def _table_title(name):
    # bokeh doesn't care :(
    title = '\n'.join(textwrap.wrap(name, width=15))
    return title


def metrics_table(cds):
    """
    Create an ugly, poorly formatted Bokeh table of metrics

    Parameters
    ----------
    cds : bokeh.models.ColumnDataSource
        Fields must be 'forecast' and the names of the metrics.

    Returns
    -------
    data_table : bokeh.widgets.DataTable
    """
    formatter = NumberFormatter(format="0.000")
    # construct list of columns. make sure that forecast name is first
    name_width = 300
    metric_width = 60
    columns = [TableColumn(field='forecast', title='Forecast',
                           width=name_width)]
    for field in filter(lambda x: x != 'forecast', cds.data):
        title = _table_title(field)
        col = TableColumn(field=field, title=title.upper(),
                          formatter=formatter, width=metric_width)
        columns.append(col)
    width = name_width + metric_width * (len(field) - 1)
    height = 25 * (1 + len(cds.data['forecast']))
    data_table = DataTable(source=cds, columns=columns, width=width,
                           height=height, index_position=None,
                           fit_columns=False)
    return data_table


def joint_distribution():
    raise NotImplementedError


def marginal_distribution():
    raise NotImplementedError


def taylor_diagram():
    raise NotImplementedError


def probabilistic_timeseries():
    raise NotImplementedError


def reliability_diagram():
    raise NotImplementedError


def rank_histogram():
    raise NotImplementedError
