"""
Functions to make all of the metrics figures for Solar Forecast Arbiter reports
using Plotly.
"""
import calendar
import datetime as dt
from itertools import cycle
import logging


from bokeh import palettes
import pandas as pd
from plotly import __version__ as plotly_version
import numpy as np


from solarforecastarbiter import datamodel

import plotly.graph_objects as go


logger = logging.getLogger(__name__)
PALETTE = (
    palettes.d3['Category20'][20][::2] + palettes.d3['Category20'][20][1::2])


PLOT_BGCOLOR = '#FFF'
PLOT_MARGINS = {'l': 50, 'r': 50, 'b': 50, 't': 50, 'pad': 4}
PLOT_LAYOUT_DEFAULTS = {
    'autosize': True,
    'height': 250,
    'margin': PLOT_MARGINS,
    'plot_bgcolor': PLOT_BGCOLOR,
    'font': {'size': 14}
}


def configure_axes(fig, x_ticks, y_range):
    """Applies plotly axes configuration to display zero line and grid.
    Parameters
    ----------
    fig: plotly.graph_objects.Figure

    x_range: list
        List of values for the x axis, None will allow cause the plot to use
        x values as labels and restrict to available data.
    y_range: Tuple
        Tuple of (min, max) to set the initial y range of the plot.
    """
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    if x_ticks is not None:
        fig.update_xaxes(ticks='outside', tickvals=x_ticks,
                         range=(-.5, len(x_ticks)))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#CCC')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(ticks='outside', range=y_range)


def construct_metrics_dataframe(metrics, rename=None):
    """
    Possibly bad assumptions:
    * metrics contains keys: name, Total, etc.

    Parameters
    ----------
    metrics : list of datamodel.MetricResults
        Each metric dict is for a different forecast. Forecast name is
        specified by the name key.
    rename : function or None
        Function of one argument that is applied to each forecast name.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe of computed metrics for the report.
    """

    if rename:
        f = rename
    else:
        def f(x): return x  # NOQA

    data = []
    for metric_result in metrics:
        for mvalue in metric_result.values:
            new = {
                'name': metric_result.name,
                'abbrev': f(metric_result.name),
                'category': mvalue.category,
                'metric': mvalue.metric,
                'value': mvalue.value
            }
            if new['category'] == 'date':
                new['index'] = dt.datetime.strptime(
                    mvalue.index, '%Y-%m-%d')
            else:
                new['index'] = mvalue.index
            data.append(new)
    df = pd.DataFrame(data, columns=[
        'name', 'abbrev', 'category', 'metric', 'value', 'index'
    ])
    return df


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


def bar(df, metric):
    """
    Create a bar graph comparing a single metric across forecasts.

    Parameters
    ----------
    df: pandas Dataframe
        Metric dataframe by :py:func:`solarforecastarbiter.reports.figures.construct_metrics_dataframe`
    metric: str
        The metric to plot. This value should be found in df['metric'].

    Returns
    -------
    data_table : bokeh.widgets.DataTable
    """  # NOQA
    data = df[(df['category'] == 'total') & (df['metric'] == metric)]
    y_range = None
    x_values = data['abbrev']
    palette = cycle(PALETTE)
    palette = [next(palette) for _ in x_values]
    metric_name = datamodel.ALLOWED_METRICS[metric]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_values, y=data['value'],
                         marker=go.bar.Marker(color=palette)))
    fig.update_layout(
        title=f'<b>{metric_name}</b>',
        xaxis_title=metric_name,
        **PLOT_LAYOUT_DEFAULTS)
    configure_axes(fig, None, y_range)
    return fig


def calc_y_start_end(y_min, y_max, pad_factor=1.03):
    """
    Determine y axis start, end.

    Parameters
    ----------
    y_min : float
    y_max : float
    pad_factor : float
        Number by which to multiply the start, end.

    Returns
    -------
    start, end : float, float
    """
    # bokeh does not play well with nans
    y_min = np.nan_to_num(y_min)
    y_max = np.nan_to_num(y_max)

    if y_max < 0:
        # all negative, so set range from y_min to 0
        start = y_min
        end = 0
    elif y_min > 0:
        # all positive, so set range from 0 to y_max
        start = 0
        end = y_max
    else:
        start = y_min
        end = y_max

    start, end = pad_factor * start, pad_factor * end
    return start, end


def bar_subdivisions(df, category, metric):
    """
    Create bar graphs comparing a single metric across subdivisions of
    time for multiple forecasts. e.g.::

        Fx 1 MAE |
                 |_________________
        Fx 2 MAE |
                 |_________________
                   Year, Month of the year, etc.

    Parameters
    ----------
    df: pandas.DataFrame
        Fields must be kind and the names of the forecasts
    category : str
        One of the available metrics grouping categories (e.g., total)

    Returns
    -------
    figs : dict of figures
    """
    palette = cycle(PALETTE)

    figs = {}

    human_category = datamodel.ALLOWED_CATEGORIES[category]
    metric_name = datamodel.ALLOWED_DETERMINISTIC_METRICS[metric]

    x_axis_label = human_category
    y_axis_label = metric_name

    data = df[(df['category'] == category) & (df['metric'] == metric)]

    x_offset = None

    # Special handling for x-axis with dates
    if category == 'weekday':
        x_ticks = calendar.day_abbr[0:]

    elif category == 'hour':
        x_ticks = list(range(25))
        # plotly's offset of 0, makes the bars left justified at the tick
        x_offset = 0
    else:
        x_ticks = None

    y_data = np.asarray(data['value'])
    if len(y_data) == 0:
        y_range = (None, None)
    else:
        y_min = np.nanmin(y_data)
        y_max = np.nanmax(y_data)
        y_range = calc_y_start_end(y_min, y_max)

    unique_names = np.unique(np.asarray(data['name']))
    palette = [next(palette) for _ in unique_names]
    for i, name in enumerate(unique_names):
        plot_data = data[data['name'] == name]
        if len(plot_data['index']):
            x_values = plot_data['index']
        else:
            x_values = []
        if category == 'weekday':
            # Fill with mon-fri values and pass to enforce displaying the full
            # week of data.
            y_values = [plot_data[plot_data['index'] == day]['value'].iloc[0]
                        if not plot_data[plot_data['index'] == day].empty
                        else np.nan for day in x_ticks]
            x_values = x_ticks
        else:
            y_values = plot_data['value']
        # Create figure
        title = name + ' ' + metric_name
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_values, y=y_values, offset=x_offset,
                             marker=go.bar.Marker(color=palette[i])))

        fig.update_layout(
            title=f'<b>{title}</b>',
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            **PLOT_LAYOUT_DEFAULTS)
        configure_axes(fig, x_ticks, y_range)
        figs[name] = fig
    return figs


def nested_bar():
    raise NotImplementedError


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


def output_svg(fig):
    """
    Generates an SVG from the Plotly figure. Errors in the process are logged
    and an SVG with error text is returned.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure

    Returns
    -------
    svg : str

    Notes
    -----
    Requires `Orca <https://plot.ly/python/orca-management/>`_ for generating
    svgs. If orca is not installed, an svg with an error message will be
    returned.
    """
    try:
        svg = fig.to_image(format='svg').decode('utf-8')
    except Exception:
        try:
            name = fig.layout.title['text'][3:-4]
        except Exception:
            name = 'unnamed'
        logger.error('Could not generate SVG for figure %s', name)
        svg = (
            '<svg width="100%" height="100%">'
            '<text x="50" y="50" class="alert alert-error">'
            'Unable to generate SVG plot.'
            '</text>'
            '</svg>')
    return svg


def raw_report_plots(report, metrics):
    """Create a RawReportPlots object from the metrics of a report.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`
    metrics: tuple of :py:class:`solarforecastarbiter.datamodel.MetricResult`

    Returns
    -------
    :py:class:`solarforecastarbiter.datamodel.RawReportPlots`
    """
    metrics_df = construct_metrics_dataframe(metrics, rename=abbreviate)
    # Create initial bar figures
    figure_dict = {}
    # Components for other metrics
    for category in report.report_parameters.categories:
        for metric in report.report_parameters.metrics:
            if category == 'total':
                fig = bar(metrics_df, metric)
                figure_dict[f'total::{metric}::all'] = fig
            else:
                figs = bar_subdivisions(metrics_df, category, metric)
                for name, fig in figs.items():
                    figure_dict[f'{category}::{metric}::{name}'] = fig
    mplots = []

    for k, v in figure_dict.items():
        cat, met, name = k.split('::', 2)
        figure_spec = v.to_json()
        svg = output_svg(v)
        mplots.append(datamodel.PlotlyReportFigure(
            name=name, category=cat, metric=met, spec=figure_spec,
            svg=svg, figure_type='bar'))

    out = datamodel.RawReportPlots(tuple(mplots), plotly_version)
    return out
