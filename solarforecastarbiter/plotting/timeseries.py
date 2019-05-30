import datetime as dt
import logging


from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Label, HoverTool
from bokeh.plotting import figure
from bokeh import palettes
import pandas as pd
import pytz


from solarforecastarbiter.plotting import utils as plot_utils
from solarforecastarbiter.validation import quality_mapping


UTCS = (dt.timezone.utc, pytz.UTC)
logger = logging.getLogger('sfa.plotting.timeseries')
PLOT_WIDTH = 900
PALETTE = palettes.all_palettes['Category20b'][20][::4]
FLAG_COLORS = {
    'MISSING': '#e6550d',
    'NOT VALIDATED': '#ff7f0e',
    'USER FLAGGED': '#d62728',
    'NIGHTTIME': None,
    'CLOUDY': None,
    'SHADED': None,
    'UNEVEN FREQUENCY': None,
    'LIMITS EXCEEDED': None,
    'CLEARSKY EXCEEDED': None,
    'STALE VALUES': None,
    'INTERPOLATED VALUES': None,
    'CLIPPED VALUES': None,
    'INCONSISTENT IRRADIANCE COMPONENTS': None
}


def build_figure_title(object_name, start, end):
    """Builds a title for the plot

    Parameters
    ----------
    object_name : str
        Name of the object being plotted

    start: datetime-like
        The start of the interval being plot.

    end: datetime-like
        The end of the interval being plot.

    Returns
    -------
    string
        The appropriate figure title.

    Raises
    ------
    ValueError
        If start or end is not localized to UTC
    """
    if start.tzinfo not in UTCS or end.tzinfo not in UTCS:
        raise ValueError('start and end must be localized to UTC')

    start_string = start.strftime('%Y-%m-%d %H:%M')
    end_string = end.strftime('%Y-%m-%d %H:%M')
    figure_title = (f'{object_name} {start_string} to {end_string} UTC')
    return figure_title


def _single_quality_bar(flag_name, plot_width, x_range, color, source):
    qfig = figure(sizing_mode='stretch_width',
                  plot_height=30,
                  plot_width=plot_width,
                  x_range=x_range,
                  toolbar_location=None,
                  min_border_bottom=0,
                  min_border_top=0,
                  tools='xpan',
                  x_axis_location=None,
                  y_axis_location=None)
    qfig.ygrid.grid_line_color = None
    qfig.line(x='timestamp', y=flag_name,
              line_width=qfig.plot_height,
              source=source, alpha=0.6,
              line_color=color)
    flag_label = Label(x=5, y=0,
                       x_units='screen', y_units='screen',
                       text=flag_name, render_mode='css',
                       border_line_color=None,
                       background_fill_alpha=0,
                       text_font_size='1em',
                       text_font_style='bold')
    qfig.add_layout(flag_label)
    return qfig


def make_quality_bars(flags, plot_width, x_range, source=None):
    """Make the quality bar figures for observation validation"""
    if source is None:
        source = ColumnDataSource(flags)
    palette = iter(PALETTE * 3)
    out = []
    for flag, color in FLAG_COLORS.items():
        if flag not in flags or flags[flag].dropna().empty:
            continue
        if color is None:
            color = next(palette)
        nextfig = _single_quality_bar(flag, plot_width, x_range,
                                      color, source)
        out.append(nextfig)

    # add the title to the top bar if there are any bars
    if out:
        out[0].plot_height = 60
        out[0].title.text = 'Quality Flags'
        out[0].title.text_font_size = '1em'

    return out


def add_hover_tool(fig, source, **hover_kwargs):
    """Add a hover tool to fig. If `add_line=True` in `hover_kwargs`
    an invisible line is added to enable hover values step plots"""
    if hover_kwargs.pop('add_line', False):
        fig.line(x='timestamp', y='value', source=source, line_alpha=0)

    tooltips = [
        ('timestamp', '@timestamp{%FT%H:%M:%S%z}'),
        ('value', '@value{%0.2f}')
    ]
    if 'active_flags' in source.data.keys():
        tooltips.append(('quality flags', '@active_flags'))
    hover = HoverTool(tooltips=tooltips, formatters={
        'timestamp': 'datetime', 'value': 'printf'}, mode='vline',
                      **hover_kwargs)
    fig.add_tools(hover)


def make_basic_timeseries(values, object_name, variable, interval_label,
                          plot_width, source=None):
    """Make a basic timeseries plot (with either a step or line)
    and add a hover tool"""
    if source is None:
        source = ColumnDataSource(values)
    plot_method, plot_kwargs, hover_kwargs = plot_utils.line_or_step(
        interval_label)
    figure_title = build_figure_title(object_name, values.index[0],
                                      values.index[-1])
    fig = figure(title=figure_title, sizing_mode='scale_width',
                 plot_width=plot_width,
                 plot_height=300, x_range=(values.index[0], values.index[-1]),
                 x_axis_type='datetime',
                 tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,reset,save',
                 toolbar_location='above',
                 min_border_bottom=50)
    getattr(fig, plot_method)(x='timestamp', y='value', source=source,
                              **plot_kwargs)
    fig.yaxis.axis_label = plot_utils.format_variable_name(variable)
    fig.xaxis.axis_label = 'Time (UTC)'
    add_hover_tool(fig, source, **hover_kwargs)
    return fig


def _make_layout(figs):
    layout = gridplot(figs,
                      ncols=1,
                      merge_tools=True,
                      toolbar_location='above',
                      sizing_mode='scale_width')
    return components(layout)


def generate_forecast_figure(forecast, data):
    """
    Creates a bokeh timeseries figure for forcast data

    Parameters
    ----------
    forecast : datamodel.Forecast
        The Forecast that is being plotted

    data : pandas.Series
        The forecast data with a datetime index to be plotted

    Returns
    -------
    script, div or None: str
        The <script> and <div> components for the Bokeh plot.
        Returns None when data is empty
    """
    logger.info('Starting forecast figure generation...')
    if len(data.index) == 0:
        return None
    data = plot_utils.align_index(data, forecast.interval_length)
    cds = ColumnDataSource(data.reset_index())
    fig = make_basic_timeseries(
        data, forecast.name, forecast.variable,
        forecast.interval_label, PLOT_WIDTH, source=cds)
    layout = _make_layout([fig])
    logger.info('Figure generated succesfully')
    return layout


def generate_observation_figure(observation, data):
    """
    Creates a bokeh figure from API responses for an observation

    Parameters
    ----------
    observation : datamodel.Observation
        The Observation that is being plotted

    data : pandas.DataFrame
        The observation data to be plotted with datetime index
        and ('value', 'quality_flag') columns

    Returns
    -------
    script, div or None: str
        The <script> and <div> components for the Bokeh plot.
        Returns None when data is empty
    """
    logger.info('Starting observation forecast generation...')
    if len(data.index) == 0:
        return None
    data = plot_utils.align_index(data, observation.interval_length,
                                  pd.Timedelta('3d'))
    quality_flag = data.pop('quality_flag').dropna().astype(int)
    bool_flags = quality_mapping.convert_mask_into_dataframe(quality_flag)
    active_flags = quality_mapping.convert_flag_frame_to_strings(bool_flags)
    active_flags.name = 'active_flags'
    flags = bool_flags.mask(~bool_flags).reindex(data.index)  # add missing
    flags['MISSING'] = pd.Series(1.0, index=data.index)[pd.isna(data['value'])]
    # need to fill as line needs more than a single point to show up
    if observation.interval_label == ' ending':
        flags.bfill(axis=0, limit=1, inplace=True)
    else:
        # for interval beginning and instantaneous
        flags.ffill(axis=0, limit=1, inplace=True)

    cds = ColumnDataSource(pd.concat([data, flags, active_flags], axis=1))
    figs = [make_basic_timeseries(
        data['value'], observation.name, observation.variable,
        observation.interval_label, PLOT_WIDTH, source=cds)]

    figs.extend(make_quality_bars(flags, PLOT_WIDTH, figs[0].x_range,
                                  cds))
    layout = _make_layout(figs)
    logger.info('Figure generated succesfully')
    return layout
