"""
Functions to make all of the figures for Solar Forecast Arbiter reports.
"""
import calendar
from contextlib import contextmanager
import datetime as dt
from itertools import cycle
import logging
import warnings


from bokeh.embed import components
from bokeh.io.export import get_svgs
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Legend, CDSView, BooleanFilter
from bokeh.models.ranges import Range1d
from bokeh.plotting import figure, Figure
from bokeh import palettes
from bokeh import __version__ as bokeh_version
import pandas as pd
from plotly import __version__ as plotly_version
import numpy as np


from solarforecastarbiter import datamodel
from solarforecastarbiter.plotting.utils import line_or_step

import plotly.graph_objects as go


logger = logging.getLogger(__name__)
PALETTE = (
    palettes.d3['Category20'][20][::2] + palettes.d3['Category20'][20][1::2])
_num_obs_colors = 3
# drop white
OBS_PALETTE = palettes.grey(_num_obs_colors + 1)[0:_num_obs_colors]
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


def configure_axes(fig, x_range, y_range):
    """Applies plotly axes configuration to display zero line and grid.
    Parameters
    ----------
    fig: plotly.graph_objects.Figure
    """
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_xaxes(ticks='outside', range=x_range)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#CCC')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(ticks='outside', range=y_range)


def construct_timeseries_cds(report):
    """Construct two standardized Bokeh CDS for the timeseries and scatter
    plot functions. One with timeseries data for all observations,
    aggregates, and forecasts in the report, and the other with
    associated metadata sharing a common `pair_index` key.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    value_cds : bokeh.models.ColumnDataSource
        Keys are an integer `pair_index` for pairing values with the metadata
        in the metadata_cds, and two pandas.Series, `observation_values` and
        `forecast_values`.

    metadata_cds : bokeh.models.ColumnDataSource
        This cds has the following keys:

        - `pair_index`: Integer for pairing metadata with the values in the value_cds.
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
    # drop tz info from localized times. GH164
    data = data.tz_localize(None)
    data = data.rename_axis('timestamp')
    value_cds = ColumnDataSource(data)
    metadata_cds = ColumnDataSource(metadata)
    return value_cds, metadata_cds


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


def _extract_metadata_from_cds(metadata_cds, hash_, hash_key):
    first_row = np.argwhere(metadata_cds.data[hash_key] == hash_)[0][0]
    return {
        'pair_index': metadata_cds.data['pair_index'][first_row],
        'observation_name': metadata_cds.data['observation_name'][first_row],
        'forecast_name': metadata_cds.data['forecast_name'][first_row],
        'interval_label': metadata_cds.data['interval_label'][first_row],
        'observation_color': metadata_cds.data['observation_color'][first_row],
    }


def timeseries(timeseries_value_cds, timeseries_meta_cds,
               start, end, units, timezone='UTC'):
    """
    Timeseries plot of one or more forecasts and observations.

    Parameters
    ----------
    timeseries_value_cds: bokeh.models.ColumnDataSource
        ColumnDataSource of timeseries data. See :py:func:`solarforecastarbiter.reports.reoports.figures.construct_timeseries_cds` for format.
    timeseries_meta_cds: bokeh.models.ColumnDataSource
        ColumnDataSource of metadata for each Observation Forecast pair. See :py:func:`solarforecastarbiter.reports.reoports.figures.construct_timeseries_cds` for format.
    start : pandas.Timestamp
        Report start time
    end : pandas.Timestamp
        Report end time
    timezone : str
        Timezone consistent with the data in the obs_fx_cds.

    Returns
    -------
    fig : bokeh.plotting.figure
    """  # NOQA
    palette = cycle(PALETTE)

    fig = figure(
        sizing_mode='scale_width', plot_width=900, plot_height=300,
        x_range=(start, end), x_axis_type='datetime',
        tools='pan,xwheel_zoom,box_zoom,box_select,lasso_select,reset,save',
        name='timeseries')

    plotted_objects = 0
    for obs_hash in np.unique(timeseries_meta_cds.data['observation_hash']):
        metadata = _extract_metadata_from_cds(
            timeseries_meta_cds, obs_hash, 'observation_hash')
        pair_indices = _boolean_filter_indices_by_pair(
            timeseries_value_cds, metadata['pair_index'])
        view = CDSView(source=timeseries_value_cds, filters=[
            BooleanFilter(pair_indices)
        ])
        plot_method, plot_kwargs, hover_kwargs = line_or_step(
            metadata['interval_label'])
        legend_label = metadata['observation_name']
        color = metadata['observation_color']
        getattr(fig, plot_method)(
            x='timestamp', y='observation_values', source=timeseries_value_cds,
            view=view, color=color, legend_label=legend_label,
            **plot_kwargs)
        plotted_objects += 1

    for fx_hash in np.unique(timeseries_meta_cds.data['forecast_hash']):
        metadata = _extract_metadata_from_cds(
            timeseries_meta_cds, fx_hash, 'forecast_hash')
        pair_indices = _boolean_filter_indices_by_pair(
            timeseries_value_cds, metadata['pair_index'])
        view = CDSView(source=timeseries_value_cds,
                       filters=[BooleanFilter(pair_indices)])
        plot_method, plot_kwargs, hover_kwargs = line_or_step(
            metadata['interval_label'])
        legend_label = metadata['forecast_name']
        color = next(palette)
        getattr(fig, plot_method)(
            x='timestamp', y='forecast_values', source=timeseries_value_cds,
            view=view, color=color, legend_label=legend_label,
            **plot_kwargs)
        plotted_objects += 1

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    if plotted_objects > 10:
        fig.legend.label_height = 10
        fig.legend.label_text_font_size = '8px'
        fig.legend.glyph_height = 10
        fig.legend.spacing = 1
        fig.legend.margin = 0
    fig.xaxis.axis_label = f'Time ({timezone})'
    fig.yaxis.axis_label = f'Data ({units})'
    return fig


def _get_scatter_limits(cds):
    extremes = [np.nan]
    for kind in ('forecast_values', 'observation_values'):
        arr = np.asarray(cds.data[kind]).astype(float)
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


def scatter(timeseries_value_cds, timeseries_meta_cds, units):
    """
    Scatter plot of one or more forecasts and observations.

    Parameters
    ----------
    timeseries_value_cds: bokeh.models.ColumnDataSource
        ColumnDataSource of timeseries data. See
        :py:func:`solarforecastarbiter.reports.reoports.figures.construct_timeseries_cds`
        for format.
    timeseries_meta_cds: bokeh.models.ColumnDataSource
        ColumnDataSource of metadata for each Observation Forecast pair. See
        :py:func:`solarforecastarbiter.reports.reoports.figures.construct_timeseries_cds`
        for format.

    Returns
    -------
    fig : bokeh.plotting.figure
    """  # NOQA
    xy_min, xy_max = _get_scatter_limits(timeseries_value_cds)

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
    for fxhash in np.unique(timeseries_meta_cds.data['forecast_hash']):
        metadata = _extract_metadata_from_cds(
            timeseries_meta_cds, fxhash, 'forecast_hash')
        pair_indices = _boolean_filter_indices_by_pair(
            timeseries_value_cds, metadata['pair_index'])
        view = CDSView(source=timeseries_value_cds,
                       filters=[BooleanFilter(pair_indices)])
        label = metadata['forecast_name']
        r = fig.scatter(
            x='observation_values', y='forecast_values',
            source=timeseries_value_cds, view=view,
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

    label = f'({units})'
    fig.xaxis.axis_label = 'Observed ' + label
    fig.yaxis.axis_label = 'Forecast ' + label
    return fig


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
        The metric to plot. This value should be found in cds['metric'].

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
    cds : bokeh.models.ColumnDataSource
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

    # fallback to plotly defaults for x axis range and offset when not needed
    x_range = None
    x_offset = None

    # Special handling for x-axis with dates
    if category == 'date':
        if len(data['index']):
            x_values = np.unique(data['index'].dt.strftime('%Y-%m-%d'))
        else:
            x_values = []
    elif category == 'month':
        x_values = calendar.month_abbr[1:]
    elif category == 'weekday':
        x_values = calendar.day_abbr[0:]
    elif category == 'hour':
        x_values = [str(i) for i in range(25)]
        x_range = (0, 24)
        x_offset = 1
    else:
        x_values = np.unique(data['index'])

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
        # Create figure
        title = name + ' ' + metric_name
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_values, y=data['value'], offset=x_offset,
                             marker=go.bar.Marker(color=palette[i])))

        fig.update_layout(
            title=f'<b>{title}</b>',
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            **PLOT_LAYOUT_DEFAULTS)
        configure_axes(fig, x_range, y_range)
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


@contextmanager
def _make_webdriver():
    """Necessary until Bokeh 2.0 when using chrome/firefox drivers will be
    preferred and to avoid zombie phantomjs processes for now"""
    from selenium import webdriver
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            driver = webdriver.PhantomJS()
        except Exception:
            yield None
        else:
            yield driver
            driver.quit()


def output_svg(fig, driver=None):
    """
    Generates an SVG from the Bokeh or Plotly figure. Errors in the
    process are logged and an SVG with error text is returned.

    Parameters
    ----------
    fig : bokeh.plotting.Figure or plotly.graph_objects.Figure
    driver : selenium.webdriver.remote.webdriver.WebDriver, default None
        Web driver to use to render SVG figures. With bokeh<2.0 this
        defaults to trying to use phantomjs.

    Returns
    -------
    svg : str
    """
    try:
        if isinstance(fig, Figure):
            fig.output_backend = 'svg'
            svg = get_svgs(fig, driver=driver)[0]
        else:
            svg = fig.to_image(format='svg').decode('utf-8')
    except Exception as e:
        logger.error('Could not generate SVG for figure %s',
                     getattr(fig, 'name', 'unnamed'))
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
        figure_spec = figure_dict[k].to_json()
        svg = output_svg(fig)
        mplots.append(datamodel.ReportFigure(
            name=name, category=cat, metric=met, spec=figure_spec,
            svg=svg, figure_type='bar'))

    out = datamodel.RawReportPlots(
        bokeh_version, tuple(mplots), plotly_version)
    return out


def timeseries_plots(report):
    """Return the bokeh components (script and div element) for timeseries
    and scatter plots of the processed forecasts and observations.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    script: str
        A script element to insert into an html template
    div: str
        A div element to insert into an html template.
    """
    value_cds, meta_cds = construct_timeseries_cds(report)
    pfxobs = report.raw_report.processed_forecasts_observations
    units = pfxobs[0].original.forecast.units
    tfig = timeseries(value_cds, meta_cds, report.report_parameters.start,
                      report.report_parameters.end, units,
                      report.raw_report.timezone)
    sfig = scatter(value_cds, meta_cds, units)
    layout = gridplot((tfig, sfig), ncols=1)
    script, div = components(layout)
    return script, div
