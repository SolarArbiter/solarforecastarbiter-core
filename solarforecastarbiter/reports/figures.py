"""
Functions to make all of the figures for Solar Forecast Arbiter reports.
"""
from itertools import cycle
import textwrap
import calendar

from bokeh.models import (ColumnDataSource, HoverTool, Legend,
                          DatetimeTickFormatter, CategoricalTickFormatter)
from bokeh.models.ranges import Range1d
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh import palettes

import pandas as pd
import numpy as np

from solarforecastarbiter import datamodel
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
    name = fx_obs.data_object.name
    if fx_obs.forecast.name == fx_obs.data_object.name:
        if isinstance(fx_obs, datamodel.Observation):
            name += ' Observation'
        else:
            name += ' Aggregate'
    return name


def _fx_name(fx_obs):
    # TODO: add code to ensure fx names are unique
    name = fx_obs.forecast.name
    if fx_obs.forecast.name == fx_obs.data_object.name:
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
            proc_fx_obs.original.data_object, proc_fx_obs.interval_value_type,
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
                color=next(palette), legend_label=name, **plot_kwargs)

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    if len(plotted_objects) > 10:
        fig.legend.label_height = 10
        fig.legend.label_text_font_size = '8px'
        fig.legend.glyph_height = 10
        fig.legend.spacing = 1
        fig.legend.margin = 0
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

    # match_aspect=True does not work well, so these need to be close
    plot_height = 400
    # width will be updated later based on label length
    plot_width = plot_height + 50
    fig = figure(
        plot_width=plot_width, plot_height=plot_height, match_aspect=True,
        x_range=Range1d(xy_min, xy_max), y_range=Range1d(xy_min, xy_max),
        tools='pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save',
        name='scatter')

    kwargs = dict(size=6, line_color=None)

    palette = cycle(PALETTE)

    # accumulate labels and plot objects for manual legend
    scatters_labels = []
    for proc_fx_obs, cds in fx_obs_cds:
        label = proc_fx_obs.original.forecast.name
        r = fig.scatter(
            x='observation', y='forecast', source=cds,
            fill_color=next(palette), **kwargs)
        scatters_labels.append((label, [r]))

    # manual legend so it can be placed outside the plot area
    legend = Legend(items=scatters_labels, location='top_center',
                    click_policy='hide')
    fig.add_layout(legend, 'right')

    # compute new plot width accounting for legend label text width.
    # also considered using second figure for legend so it doesn't
    # distort the first when text length/size changes. unfortunately,
    # that doesn't work due to bokeh's inability to communicate legend
    # information across figures.
    # widest part of the legend
    max_legend_length = max((len(label) for label, _ in scatters_labels))
    px_per_length = 7.75  # found through trial and error
    fig.plot_width = int(fig.plot_width + max_legend_length * px_per_length)

    label = format_variable_name(proc_fx_obs.original.forecast.variable)
    fig.xaxis.axis_label = 'Observed ' + label
    fig.yaxis.axis_label = 'Forecast ' + label
    return fig


def construct_metrics_cds(metrics, kind, index='forecast', rename=None):
    """
    Possibly bad assumptions:
    * metrics contains keys: name, Total, etc.

    Parameters
    ----------
    metrics : list of metrics dicts
        Each metric dict is for a different forecast. Forecast name is
        specified by the name key.
    kind : str
        One of the available metrics grouping categories (e.g., Total)
    index : str
        Determines if the index is the array of metrics ('metric') or
        forecast ('forecast') names
    rename : function or None
        Function of one argument that is applied to each forecast name.

    Returns
    -------
    cds : bokeh.models.ColumnDataSource
    """

    if kind == 'Total':
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
        One of the available metrics grouping categories (e.g., Total)

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
            if kind == 'Total':
                forecasts.append(m['name'])
                m_types.append(col)
                m_indexes.append(0)
                m_values.append(m[kind][col])
            else:
                for i, v in m[kind][col].items():
                    forecasts.append(m['name'])
                    m_types.append(i)
                    m_indexes.append(col)
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
               Year, Month of the year, etc.

    Parameters
    ----------
    cds : bokeh.models.ColumnDataSource
        Fields must be kind and the names of the forecasts
    kind : str
        One of the available metrics grouping categories (e.g., Total)

    Returns
    -------
    figs : list of figures
    """
    palette = cycle(PALETTE)
    tools = 'pan,xwheel_zoom,box_zoom,box_select,reset,save'
    fig_kwargs = dict(tools=tools)
    figs = []

    width = 0.8

    fig_kwargs['x_axis_label'] = kind

    # Special handling for x-axis with dates
    if kind == 'Date':
        fig_kwargs['x_axis_type'] = 'datetime'
        width = width * pd.Timedelta(days=1)
    elif kind == 'Month of the year':
        fig_kwargs['x_range'] = calendar.month_abbr[1:]
    elif kind == 'Day of the week':
        fig_kwargs['x_range'] = calendar.day_abbr[0:]

    # vertical axis limits
    y_min = min(d.min() for k, d in cds.data.items() if k != kind)
    y_max = max(d.max() for k, d in cds.data.items() if k != kind)
    pad_factor = 1.03
    y_max, y_min = pad_factor * y_max, pad_factor * y_min

    for num, field in enumerate(filter(lambda x: x != kind, cds.data)):

        # Create figure
        title = field + ' ' + metric.upper()
        fig = figure(width=800, height=200, title=title,
                     **fig_kwargs)

        fig.vbar(x=kind, top=field, width=width, source=cds,
                 line_color='white', fill_color=next(palette))

        # axes parameters
        fig.xgrid.grid_line_color = None
        fig.xaxis.minor_tick_line_color = None
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
            fig_kwargs['y_range'] = fig.y_range

        # Hover tool and format specific changes
        if kind == 'Date':
            # Datetime x-axis
            formatter = DatetimeTickFormatter(days='%Y-%m-%d')
            fig.xaxis.formatter = formatter
            tooltips = [
                (kind, f'@{kind}{{%F}}'),
                (f'{field} {metric.upper()}', f'@{{{field}}}'),
            ]
            hover_kwargs = dict(tooltips=tooltips,
                                formatters={kind: 'datetime'})
        elif kind == 'Month of the year' or kind == 'Day of the week':
            # Categorical x-axis
            formatter = CategoricalTickFormatter()
            fig.xaxis.formatter = formatter
            tooltips = [
                (kind, f'@{{{kind}}}'),
                (f'{metric.upper()}', f'@{{{field}}}'),
            ]
            hover_kwargs = dict(tooltips=tooltips)
        else:
            # Numerical x-axis
            fig.xaxis.ticker = cds.data[kind]
            tooltips = [
                (kind, f'@{{{kind}}}'),
                (f'{metric.upper()}', f'@{{{field}}}'),
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
