import datetime as dt
from functools import wraps
import logging


from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Label, HoverTool
from bokeh.plotting import figure
from bokeh import palettes
from matplotlib import cm
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import pandas as pd
import pytz


from solarforecastarbiter.plotting import utils as plot_utils
from solarforecastarbiter.validation import quality_mapping


UTCS = (dt.timezone.utc, pytz.UTC)
logger = logging.getLogger('sfa.plotting.timeseries')
PLOT_WIDTH = 900
PALETTE = palettes.all_palettes['Category20b'][20][::4]
# flags with color None will be assigned a color from PALETTE
FLAG_COLORS = {
    'MISSING': '#e6550d',
    'NOT VALIDATED': '#ff7f0e',
    'USER FLAGGED': '#d62728',
    'NIGHTTIME': None,
    'DAYTIME': None,
    'CLEARSKY': None,
    'SHADED': None,
    'UNEVEN FREQUENCY': None,
    'LIMITS EXCEEDED': None,
    'CLEARSKY EXCEEDED': None,
    'STALE VALUES': None,
    'DAYTIME STALE VALUES': None,
    'INTERPOLATED VALUES': None,
    'DAYTIME INTERPOLATED VALUES': None,
    'CLIPPED VALUES': None,
    'INCONSISTENT IRRADIANCE COMPONENTS': None,
}


def build_figure_title(object_name, start, end):
    """Builds a title for the plot

    Parameters
    ----------
    object_name : str
        Name of the object being plotted

    start: datetime-like
        The start of the interval being plotted.

    end: datetime-like
        The end of the interval being plotted.

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


def make_quality_bars(source, plot_width, x_range):
    """
    Make figures to display the whether a time is flagged for any
    of the columns in source.

    Parameters
    ----------
    source : bokeh.models.ColumnDataSource
        The predefined data source with flags loaded. Only columns
        in FLAG_COLORS will be made into bars. If data for a flag
        is empty, a bar will not be generated for that flag.
    plot_width : int
        The width of the figures
    x_range : bokeh.Range or tuple
        If x_range is a bokeh Range from another plot, the plots will
        be linked on panning/zooming/etc.

    Returns
    -------
    list
       Of bar figures. The top figure will have an appropriate title.
    """
    palette = iter(PALETTE * 3)
    out = []
    for flag, color in FLAG_COLORS.items():
        # only display bars for flags that have at least on occurence
        if flag not in source.data or pd.isna(source.data[flag]).all():
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


def make_basic_timeseries(source, object_name, variable, interval_label,
                          plot_width):
    """
    Make a basic timeseries plot (with either a step or line)
    and add a hover tool.

    Parameters
    ----------
    source : bokeh.models.ColumnDataSource
        The datasource with 'timestamp' and 'value' columns that will be
        plotted.
    object_name : str
        Name of the object to be plotted so an appropriate title can be made.
    variable : str
        Variable of the plotted object
    interval_label : str
        Interval label of the object to determine whether a line or step
        is most appropriate.
    plot_width : int
        The width of the output figure

    Returns
    -------
    bokeh.models.Figure
        The figure with the timeseries

    Raises
    ------
    KeyError
       If timestamp is not a column in source
    IndexError
       If the timestamp column is empty
    """
    timestamps = source.data['timestamp']
    first = pd.Timestamp(timestamps[0], tz='UTC')
    last = pd.Timestamp(timestamps[-1], tz='UTC')
    plot_method, plot_kwargs, hover_kwargs = plot_utils.line_or_step(
        interval_label)
    figure_title = build_figure_title(object_name, first, last)
    fig = figure(title=figure_title, sizing_mode='scale_width',
                 plot_width=plot_width,
                 plot_height=300, x_range=(first, last),
                 x_axis_type='datetime',
                 tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,reset,save',
                 toolbar_location='above',
                 min_border_bottom=50)
    getattr(fig, plot_method)(x='timestamp', y='value', source=source,
                              **plot_kwargs)
    fig.yaxis.axis_label = plot_utils.format_variable_name(variable)
    fig.xaxis.axis_label = 'Time (UTC)'
    if variable == 'event':
        fig.yaxis.ticker = [0, 1]
        fig.yaxis.major_label_overrides = {1: 'True', 0: 'False'}
    add_hover_tool(fig, source, **hover_kwargs)
    return fig


def _make_layout(figs):
    layout = gridplot(figs,
                      ncols=1,
                      merge_tools=True,
                      toolbar_location='above',
                      sizing_mode='scale_width')
    return layout


def to_components(f):
    """Return script and div of a bokeh object if the return_components
    kwarg is True"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if kwargs.pop('return_components', False):
            out = f(*args, **kwargs)
            if out is not None:
                return components(out)
            else:
                return out
        else:
            return f(*args, **kwargs)
    return wrapper


@to_components
def generate_forecast_figure(forecast, data, limit=None):
    """
    Creates a bokeh timeseries figure for forcast data

    Parameters
    ----------
    forecast : datamodel.Forecast
        The Forecast that is being plotted

    data : pandas.Series
        The forecast data with a datetime index to be plotted

    limit : pandas.Timedelta or None
        The time limit from the last datapoint to plot. If None, all
        data is plotted.


    Returns
    -------
    None
        When the data is empty
    script, div : str
        When return_components = True, return the <script> and <div>
        components for the Bokeh plot.
    bokeh components from gridplot
        When return_components = False
    """
    logger.info('Starting forecast figure generation...')
    if len(data.index) == 0:
        return None
    data = plot_utils.align_index(data, forecast.interval_length, limit)
    cds = ColumnDataSource(data.reset_index())
    fig = make_basic_timeseries(cds, forecast.name, forecast.variable,
                                forecast.interval_label, PLOT_WIDTH)
    layout = _make_layout([fig])
    logger.info('Figure generated succesfully')
    return layout


@to_components
def generate_observation_figure(observation, data, limit=pd.Timedelta('3d')):
    """
    Creates a bokeh figure from API responses for an observation

    Parameters
    ----------
    observation : datamodel.Observation
        The Observation that is being plotted

    data : pandas.DataFrame
        The observation data to be plotted with datetime index
        and ('value', 'quality_flag') columns

    limit : pandas.Timedelta or None
        The time limit from the last datapoint to plot. If None, all
        data is plotted.

    Returns
    -------
    None
        When the data is empty
    script, div : str
        When return_components = True, return the <script> and <div>
        components for the Bokeh plot.
    bokeh components from gridplot
        When return_components = False
    """
    logger.info('Starting observation forecast generation...')
    if len(data.index) == 0:
        return None
    data = plot_utils.align_index(data, observation.interval_length, limit)
    quality_flag = data.pop('quality_flag').dropna().astype(int)
    bool_flags = quality_mapping.convert_mask_into_dataframe(quality_flag)
    active_flags = quality_mapping.convert_flag_frame_to_strings(bool_flags)
    active_flags.name = 'active_flags'
    flags = bool_flags.mask(~bool_flags).reindex(data.index)  # add missing
    flags['MISSING'] = pd.Series(1.0, index=data.index)[pd.isna(data['value'])]
    # need to fill as line needs more than a single point to show up
    if observation.interval_label == 'ending':
        flags.bfill(axis=0, limit=1, inplace=True)
    else:
        # for interval beginning and instantaneous
        flags.ffill(axis=0, limit=1, inplace=True)
    cds = ColumnDataSource(pd.concat([data, flags, active_flags], axis=1))
    figs = [make_basic_timeseries(cds, observation.name, observation.variable,
                                  observation.interval_label, PLOT_WIDTH)]

    figs.extend(make_quality_bars(cds, PLOT_WIDTH, figs[0].x_range))
    layout = _make_layout(figs)
    logger.info('Figure generated succesfully')
    return layout


PLOTLY_MARGINS = {'l': 50, 'r': 50, 'b': 50, 't': 100, 'pad': 4}
PLOTLY_LAYOUT_DEFAULTS = {
    'autosize': True,
    'height': 300,
    'margin': PLOTLY_MARGINS,
    'plot_bgcolor': '#FFF',
    'title_font_size': 16,
    'font': {'size': 14}
}


def _plot_probabilsitic_distribution_axis_y(fig, forecast, data):
    """
    Plot all probabilistic forecast values for axis='y' by adding traces to
    fig.

    Parameters
    ----------
    fig: plotly.graph_objects.Figure
    forecast: :py:class`solarforecastarbiter.datamodel.ProbabilisticForecast`
    data: pd.DataFrame
    """
    color_map = cm.get_cmap('viridis')
    color_scaler = cm.ScalarMappable(
        Normalize(vmin=0, vmax=1),
        color_map,
    )

    units = forecast.units

    percentiles_are_symmetric = plot_utils.percentiles_are_symmetric(
        data.columns.values.astype('float'))

    # may not work for constant values that don't convert nicely from str/float
    constant_values = data.columns.astype('float').sort_values()
    for i, constant_value in enumerate(constant_values):
        if i == 0:
            fill = None
        else:
            fill = 'tonexty'

        if percentiles_are_symmetric:
            if constant_value <= 50 and i != 0:
                fill_value = constant_values[i - 1]
            else:
                fill_value = constant_value
            fill_value = 2 * abs(fill_value - 50)
        else:
            fill_value = 100 - constant_value

        fill_color = plot_utils.distribution_fill_color(
            color_scaler, fill_value)

        plot_kwargs = plot_utils.line_or_step_plotly(forecast.interval_label)

        forecast_name = f'Prob(f <= x) = {str(constant_value)}%'

        go_ = go.Scatter(
            x=data.index,
            y=data[str(constant_value)],
            name=f'{str(constant_value)} %',
            hovertemplate=(
                f'<b>{forecast_name}</b><br>'
                '<b>Value</b>: %{y} '+f'{units}<br>'
                '<b>Time</b>: %{x}<br>'),
            connectgaps=False,
            showlegend=False,
            mode='lines',
            fill=fill,
            fillcolor=fill_color,
            line=dict(
                color=fill_color,
            ),
            **plot_kwargs,
        )
        fig.add_trace(go_)


def _plot_probabilsitic_distribution_axis_x(fig, forecast, data):
    """
    Plot all probabilistic forecast values for axis='x' by adding traces to
    fig.

    Parameters
    ----------
    fig: plotly.graph_objects.Figure
    forecast: :py:class`solarforecastarbiter.datamodel.ProbabilisticForecast`
    data: pd.DataFrame
    """
    palette = iter(PALETTE * 3)

    units = forecast.units

    for constant_value in data.columns:
        line_color = next(palette)

        plot_kwargs = plot_utils.line_or_step_plotly(forecast.interval_label)

        forecast_name = f'Prob(x <= {str(constant_value)} {units})'
        go_ = go.Scatter(
            x=data.index,
            y=data[str(constant_value)],
            name=forecast_name,
            hovertemplate=(
                f'<b>{forecast_name}</b><br>'
                '<b>Value</b>: %{y} %<br>'
                '<b>Time</b>: %{x}<br>'),
            connectgaps=False,
            showlegend=True,
            mode='lines',
            line=dict(
                color=line_color,
            ),
            **plot_kwargs,
        )
        fig.add_trace(go_)


def generate_probabilistic_forecast_figure(
        forecast, data, limit=pd.Timedelta('3d')):
    """
    Creates a plotly figure spec from api response for a probabilistic forecast
    group.

    Parameters
    ----------
    forecast : datamodel.ProbabilisticForecast
    data : pandas.DataFrame
        DataFrame with forecast values in each column, column names as the
        constant values and a datetime index.
    limit : pandas.Timedelta or None

    Returns
    -------
    None
        When the data is empty.
    figure: Plotly.graph_objects.Figure
        Plotly json specification for the plot.
    """
    logger.info('Starting probabilistic forecast figure generation...')
    if len(data.index) == 0:
        return None

    fig = go.Figure()
    if 'x' in forecast.axis:
        ylabel = 'Probability (%)'
        _plot_probabilsitic_distribution_axis_x(fig, forecast, data)
    else:
        ylabel = plot_utils.format_variable_name(forecast.variable)
        _plot_probabilsitic_distribution_axis_y(fig, forecast, data)
    fig.update_xaxes(title_text=f'Time (UTC)', showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside')
    fig.update_yaxes(title_text=ylabel, showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside',
                     fixedrange=True)
    first = data.index[0]
    last = data.index[-1]
    fig.update_layout(
        title=build_figure_title(forecast.name, first, last),
        legend=dict(font=dict(size=10)),
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig
