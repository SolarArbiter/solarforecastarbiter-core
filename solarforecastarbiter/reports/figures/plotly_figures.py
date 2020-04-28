"""
Functions to make all of the metrics figures for Solar Forecast Arbiter reports
using Plotly.
"""
import calendar
import datetime as dt
from itertools import cycle
import logging


import pandas as pd
from plotly import __version__ as plotly_version
import plotly.graph_objects as go
import numpy as np


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics.event import _event2count

from solarforecastarbiter.plotting.utils import line_or_step_plotly


logger = logging.getLogger(__name__)
D3_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
              '#dbdb8d', '#9edae5']
PALETTE = (D3_PALETTE[::2] + D3_PALETTE[1::2])


def gen_grays(num_colors):
    """Generate a grayscale color list of length num_colors.
    """
    rgb_delta = int(255/num_colors + 1)
    color_list = ["#{h}{h}{h}".format(h=hex(i*rgb_delta)[2:])
                  for i in range(num_colors)]
    return color_list


_num_obs_colors = 3
# drop white
OBS_PALETTE = gen_grays(_num_obs_colors)
OBS_PALETTE.reverse()
OBS_PALETTE_TD_RANGE = pd.timedelta_range(
            freq='10min', end='60min', periods=_num_obs_colors)

PLOT_BGCOLOR = '#FFF'
PLOT_MARGINS = {'l': 50, 'r': 50, 'b': 50, 't': 50, 'pad': 4}
PLOT_LAYOUT_DEFAULTS = {
    'autosize': True,
    'height': 250,
    'margin': PLOT_MARGINS,
    'plot_bgcolor': PLOT_BGCOLOR,
    'font': {'size': 14}
}


def construct_timeseries_dataframe(report):
    """Construct two standardized Dataframes for the timeseries and scatter
    plot functions. One with timeseries data for all observations,
    aggregates, and forecasts in the report, and the other with
    associated metadata sharing a common `pair_index` key.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    data : pandas.DataFrame
        Keys are an integer `pair_index` for pairing values with the metadata
        in the metadata_cds, and two pandas.Series, `observation_values` and
        `forecast_values`.

    metadata : pandas.DataFrame
        This dataframe has the following columns:

        - `pair_index`: Integer for pairing metadata with the values in the data dataframe.
        - `observation_name`: Observation name.
        - `forecast_name`: Forecast name.
        - `interval_label`: Interval label of the processed forecast and observation data.
        - `observation_hash`: Hash of the original observation object and the `datamodel.ProcessedForecastObservations` metadata.
        - `forecast_hash`: Hash of the original forecast object and the `datamodel.ProcessedForecastObservations` metadata.

    """  # NOQA
    value_frames = []
    meta_rows = []
    for idx, pfxobs in enumerate(
            report.raw_report.processed_forecasts_observations):
        value_frame_dict = {
            'pair_index': idx,
            'observation_values': pfxobs.observation_values,
            'forecast_values': pfxobs.forecast_values,
        }
        meta_row_dict = {
            'pair_index': idx,
            'observation_name': _obs_name(pfxobs.original),
            'forecast_name': _fx_name(pfxobs.original),
            'interval_label': pfxobs.interval_label,
            'interval_length': pfxobs.interval_length,
            'observation_hash': str(hash(
                (pfxobs.original.data_object,
                 pfxobs.interval_length,
                 pfxobs.interval_value_type,
                 pfxobs.interval_label))),
            'forecast_hash': str(hash(
                (pfxobs.original.forecast,
                 pfxobs.interval_length,
                 pfxobs.interval_value_type,
                 pfxobs.interval_label))),
            'observation_color': _obs_color(
                pfxobs.interval_length)
        }
        value_frames.append(pd.DataFrame(value_frame_dict))
        meta_rows.append(meta_row_dict)
    data = pd.concat(value_frames)
    metadata = pd.DataFrame(meta_rows)
    # convert data to report timezone
    data = data.tz_convert(report.raw_report.timezone)
    data = data.rename_axis('timestamp')
    return data, metadata


def _fill_timeseries(df, interval_length):
    """Returns a dataframe with a datetimeindex with regular frequency of
    interval_length minutes. Previously missing values will be filled with
    nans. Useful for creating gaps in plotted timeseries data.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with timeseries data.
    interval_length: numpy.timedelta64
        Interval length of the processed forecast observation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with filled datetime index data.
    """
    if not df.index.empty:
        start = df.index[0]
        end = df.index[-1]
        freq_mins = int(interval_length / np.timedelta64(1, 'm'))
        filled_idx = pd.date_range(start, end, freq=f'{freq_mins}min')
        return df.reindex(filled_idx)
    else:
        return df


def _obs_name(fx_obs):
    # TODO: add code to ensure obs names are unique
    name = fx_obs.data_object.name
    if fx_obs.forecast.name == fx_obs.data_object.name:
        if isinstance(fx_obs.data_object, datamodel.Observation):
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


def _boolean_filter_indices_by_pair(value_cds, pair_index):
    return value_cds.data['pair_index'] == pair_index


def _extract_metadata_from_df(metadata_df, hash_, hash_key):
    metadata = metadata_df[metadata_df[hash_key] == hash_]
    return {
        'pair_index': metadata['pair_index'].values[0],
        'observation_name': metadata['observation_name'].values[0],
        'forecast_name': metadata['forecast_name'].values[0],
        'interval_label': metadata['interval_label'].values[0],
        'interval_length': metadata['interval_length'].values[0],
        'observation_color': metadata['observation_color'].values[0],
    }


def _legend_text(name, max_length=20):
    """Inserts <br> tags in a name to mimic word-wrap behavior for long names
    in the legend of timeseries plots.

    Parameters
    ----------
    name: str
        The name/string to apply word-wrap effect to.
    max_length: int
        The maximum length of any line of text. Note that this will not break
        words across lines, but on the closest following space.

    Returns
    -------
    str
        The name after it is split appropriately.
    """
    if len(name) > max_length:
        temp = []
        new = []
        for part in name.split(' '):
            if len(' '.join(temp + [part])) > max_length:
                new.append(' '.join(temp))
                temp = [part]
            else:
                temp.append(part)
        if temp:
            new.append(' '.join(temp))
        return '<br>'.join(new)
    else:
        return name


def timeseries(timeseries_value_df, timeseries_meta_df,
               start, end, units, timezone='UTC'):
    """
    Timeseries plot of one or more forecasts and observations.

    Parameters
    ----------
    timeseries_value_df: pandas.DataFrame
        DataFrame of timeseries data. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.
    timeseries_meta_df: pandas.DataFrame
        DataFrame of metadata for each Observation Forecast pair. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.
    start : pandas.Timestamp
        Report start time
    end : pandas.Timestamp
        Report end time
    timezone : str
        Timezone consistent with the data in the timeseries_metadata_df.

    Returns
    -------
    plotly.Figure
    """  # NOQA
    fig = go.Figure()
    plotted_objects = 0
    for obs_hash in np.unique(timeseries_meta_df['observation_hash']):
        metadata = _extract_metadata_from_df(
            timeseries_meta_df, obs_hash, 'observation_hash')
        pair_idcs = timeseries_value_df['pair_index'] == metadata['pair_index']
        plot_kwargs = line_or_step_plotly(metadata['interval_label'])
        data = _fill_timeseries(
            timeseries_value_df[pair_idcs],
            metadata['interval_length'],
        )
        fig.add_trace(go.Scattergl(
            y=data['observation_values'],
            x=data.index,
            name=_legend_text(metadata['observation_name']),
            legendgroup=metadata['observation_name'],
            marker=dict(color=metadata['observation_color']),
            connectgaps=False,
            **plot_kwargs),
        )
        plotted_objects += 1

    palette = cycle(PALETTE)
    for fx_hash in np.unique(timeseries_meta_df['forecast_hash']):
        metadata = _extract_metadata_from_df(
            timeseries_meta_df, fx_hash, 'forecast_hash')
        pair_idcs = timeseries_value_df['pair_index'] == metadata['pair_index']
        plot_kwargs = line_or_step_plotly(metadata['interval_label'])
        data = _fill_timeseries(
            timeseries_value_df[pair_idcs],
            metadata['interval_length'],
        )
        fig.add_trace(go.Scattergl(
            y=data['forecast_values'],
            x=data.index,
            name=_legend_text(metadata['forecast_name']),
            legendgroup=metadata['forecast_name'],
            marker=dict(color=next(palette)),
            connectgaps=False,
            **plot_kwargs),
        )
        plotted_objects += 1
    fig.update_xaxes(title_text=f'Time ({timezone})', showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside')
    fig.update_yaxes(title_text=f'Data ({units})', showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside',
                     fixedrange=True)
    return fig


def _get_scatter_limits(df):
    extremes = [np.nan]
    for kind in ('forecast_values', 'observation_values'):
        arr = np.asarray(df[kind]).astype(float)
        if len(arr) != 0:
            extremes.append(np.nanmin(arr))
            extremes.append(np.nanmax(arr))
    min_ = np.nanmin(extremes)
    if np.isnan(min_):
        min_ = -999
    max_ = np.nanmax(extremes)
    if np.isnan(max_):
        max_ = 999
    return min_, max_


def scatter(timeseries_value_df, timeseries_meta_df, units):
    """
    Adds Scatter plot traces of one or more forecasts and observations to
    the figure.

    Parameters
    ----------
    timeseries_value_df: pandas.DataFrame
        DataFrame of timeseries data. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.
    timeseries_meta_df: pandas.DataFrame
        DataFrame of metadata for each Observation Forecast pair. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.

    Returns
    -------
    plotly.Figure
    """  # NOQA
    scatter_range = _get_scatter_limits(timeseries_value_df)

    palette = cycle(PALETTE)
    fig = go.Figure()
    # accumulate labels and plot objects for manual legend
    for fxhash in np.unique(timeseries_meta_df['forecast_hash']):
        metadata = _extract_metadata_from_df(
            timeseries_meta_df, fxhash, 'forecast_hash')
        pair_idcs = timeseries_value_df['pair_index'] == metadata['pair_index']
        data = timeseries_value_df[pair_idcs]

        fig.add_trace(go.Scattergl(
            x=data['observation_values'],
            y=data['forecast_values'],
            name=_legend_text(metadata['forecast_name']),
            showlegend=True,
            legendgroup=metadata['forecast_name'],
            marker=dict(color=next(palette)),
            mode='markers'),
        )
    # compute new plot width accounting for legend label text width.
    # also considered using second figure for legend so it doesn't
    # distort the first when text length/size changes. unfortunately,
    # that doesn't work due to bokeh's inability to communicate legend
    # information across figures.
    # widest part of the legend
    label = f'({units})'
    x_label = 'Observed ' + label
    y_label = 'Forecast ' + label
    fig.update_xaxes(title_text=x_label, showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside',
                     range=scatter_range)
    fig.update_yaxes(title_text=y_label, showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside',
                     range=scatter_range)
    return fig


def event_histogram(timeseries_value_df, timeseries_meta_df):
    """
    Adds histogram plot traces of the event outcomes of one or more event
    forecasts and observations to the figure.

    Parameters
    ----------
    timeseries_value_df: pandas.DataFrame
        DataFrame of timeseries data. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.
    timeseries_meta_df: pandas.DataFrame
        DataFrame of metadata for each Observation Forecast pair. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.

    Returns
    -------
    plotly.Figure
    """  # NOQA

    fig = go.Figure()
    palette = cycle(PALETTE)
    # accumulate labels and plot objects for manual legend
    for fxhash in np.unique(timeseries_meta_df['forecast_hash']):
        metadata = _extract_metadata_from_df(
            timeseries_meta_df, fxhash, 'forecast_hash')
        pair_idcs = timeseries_value_df['pair_index'] == metadata['pair_index']
        data = timeseries_value_df[pair_idcs]

        tp, fp, tn, fn = _event2count(data["observation_values"],
                                      data["forecast_values"])
        x = ["True Pos.", "False Pos.", "True Neg.", "False Neg."]
        y = [tp, fp, tn, fn]

        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name=_legend_text(metadata['forecast_name']),
            showlegend=True,
            legendgroup=metadata['forecast_name'],
            marker_color=next(palette),
        ))

    # update axes
    x_label = "Outcome"
    y_label = "Count"
    fig.update_xaxes(title_text=x_label, showgrid=True,
                     gridwidth=0, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside')
    fig.update_yaxes(title_text=y_label, showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside')

    return fig


def configure_axes(fig, x_axis_kwargs, y_axis_kwargs):
    """Applies plotly axes configuration to display zero line and grid, and the
    configuration passed in x_axis_kwargs and y_axis kwargs. Currently
    configured to supply base layout for metric plots.

    Parameters
    ----------
    fig: plotly.graph_objects.Figure

    x_axis_kwargs: dict
        Dictionary to expand as arguments to fig.update_xaxes.
    y_axis_kwargs: dict
        Dictionary to expand as arguments to fig.update_x_axes.
    """
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black',
                     ticks='outside')
    if x_axis_kwargs:
        fig.update_xaxes(**x_axis_kwargs)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#CCC')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black',
                     ticks='outside')
    if y_axis_kwargs:
        fig.update_yaxes(**y_axis_kwargs)


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
        # error here for ProbabilisticForecasts.
        # metrics is a 2 element tuple:
        # element 0: list of MetricsResults for each ProbFxConstVal in the ProbFx
        # element 1: MetricsResult for distribution metrics (CRPS)
        # so metric_result.values fails on the list object
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
    plotly.Figure
        A bar chart representing the total category of the metric for each
        forecast.
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
    # remove height limit when long abbreviations are used or there are more
    # than 5 pairs to problems with labels being cutt off.
    plot_layout_args = PLOT_LAYOUT_DEFAULTS.copy()
    if x_values.map(len).max() > 15 or x_values.size > 6:
        plot_layout_args.pop('height')
    fig.update_layout(
        title=f'<b>{metric_name}</b>',
        xaxis_title=metric_name,
        **plot_layout_args)
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
    # limits cannot be nans or infs
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

    # if y_max or min was +/- inf then padding will result in overflow
    # that can be ignored
    with np.errstate(over='ignore'):
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
    metric : str
        One of the available metrics (e.g. mae)

    Returns
    -------
    figs : dict of figures
    """
    palette = cycle(PALETTE)

    figs = {}

    human_category = datamodel.ALLOWED_CATEGORIES[category]
    metric_name = datamodel.ALLOWED_METRICS[metric]

    x_axis_label = human_category
    y_axis_label = metric_name

    data = df[(df['category'] == category) & (df['metric'] == metric)]

    x_offset = None

    # Special handling for x-axis with dates
    if category == 'weekday':
        x_ticks = calendar.day_abbr[0:]
        x_axis_kwargs = {'tickvals': x_ticks,
                         'range': (-.5, len(x_ticks))}
    elif category == 'hour':
        x_ticks = list(range(25))
        x_axis_kwargs = {'tickvals': x_ticks,
                         'range': (-.5, len(x_ticks))}
        # plotly's offset of 0, makes the bars left justified at the tick
        x_offset = 0
    elif category == 'year':
        x_axis_kwargs = {'dtick': 1}
    elif category == 'date':
        x_axis_kwargs = {'dtick': '1d'}
    else:
        x_axis_kwargs = {}

    y_data = np.asarray(data['value'])
    if len(y_data) == 0 or np.isnan(y_data).all():
        y_range = (None, None)
    else:
        y_min = np.nanmin(y_data)
        y_max = np.nanmax(y_data)
        y_range = calc_y_start_end(y_min, y_max)
    y_axis_kwargs = {'range': y_range}
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
        configure_axes(fig, x_axis_kwargs, y_axis_kwargs)
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


def timeseries_plots(report):
    """Return the bokeh components (script and div element) for timeseries
    and scatter plots of the processed forecasts and observations.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    tuple of str
        Tuple of string json specifications of plotly figures. The first member
        of the tuple is the specification for the timeseries plot, and the
        second is the specification of the scatter plot.
    """
    value_df, meta_df = construct_timeseries_dataframe(report)
    pfxobs = report.raw_report.processed_forecasts_observations
    units = pfxobs[0].original.forecast.units
    ts_fig = timeseries(
        value_df, meta_df, report.report_parameters.start,
        report.report_parameters.end, units,
        report.raw_report.timezone)
    ts_fig.update_layout(
        plot_bgcolor=PLOT_BGCOLOR,
        font=dict(size=14),
        margin=PLOT_MARGINS,

    )

    # switch secondary plot based on forecast type
    pfxobs = report.raw_report.processed_forecasts_observations
    fx = pfxobs[0].original.forecast
    if isinstance(fx, datamodel.EventForecast):
        scat_fig = event_histogram(value_df, meta_df)
        scat_fig.update_layout(
            plot_bgcolor=PLOT_BGCOLOR,
            font=dict(size=14),
            margin=PLOT_MARGINS,
        )
    else:
        scat_fig = scatter(value_df, meta_df, units)
        scat_fig.update_layout(
            plot_bgcolor=PLOT_BGCOLOR,
            font=dict(size=14),
            width=600,
            height=500,
            autosize=False,
        )
    return ts_fig.to_json(), scat_fig.to_json()
