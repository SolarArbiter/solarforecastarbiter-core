from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Label
from bokeh.plotting import figure
from bokeh import palettes
import pandas as pd


from solarforecastarbiter.io.utils import (json_payload_to_observation_df,
                                           json_payload_to_forecast_series)
from solarforecastarbiter.plotting.utils import format_variable_name
from solarforecastarbiter.validation import quality_mapping


def build_figure_title(object_name, start, end):
    """Builds a title for the plot from a metadata object.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary used to label the plot. Must include a 'site'
        key containing a nested site object as well as the 'variable' key.

    start: datetime-like
        The start of the interval being plot.

    end: datetime-like
        The end of the interval being plot.

    Returns
    -------
    string
        The appropriate figure title.
    """
    start_string = start.strftime('%Y-%m-%d %H:%M')
    end_string = end.strftime('%Y-%m-%d %H:%M')
    figure_title = (f'{object_name} {start_string} to {end_string} UTC')
    return figure_title


def _build_quality_bar(flag_name, plot_width, x_range, color, source):
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
              source=source,
              line_color=color)
    flag_label = Label(x=5, y=0,
                       x_units='screen', y_units='screen',
                       text=flag_name, render_mode='css',
                       border_line_color=None,
                       background_fill_alpha=0,
                       text_font_size='1em')
    qfig.add_layout(flag_label)
    return qfig


def generate_quality_bars(flags, plot_width, x_range, source=None):
    if source is None:
        source = ColumnDataSource(flags)
    colors = iter(palettes.all_palettes['Category10'][8][::-1])
    out = []
    for flag in flags.columns:
        if flags[flag].dropna().empty:
            continue
        nextfig = _build_quality_bar(flag, plot_width, x_range,
                                     next(colors), source)
        out.append(nextfig)

    if out:
        out[0].plot_height = 60
        out[0].title.text = 'Quality Flags'
        out[0].title.text_font_size = '1em'

    return out


def generate_basic_timeseries(values, object_name, variable, plot_width,
                              source=None):
    if source is None:
        source = ColumnDataSource(values)
    figure_title = build_figure_title(object_name, values.index[0],
                                      values.index[-1])
    fig = figure(title=figure_title, sizing_mode='scale_width',
                 plot_width=plot_width,
                 plot_height=300, x_range=(values.index[0], values.index[-1]),
                 x_axis_type='datetime',
                 tools='pan,wheel_zoom,box_zoom,reset,save',
                 toolbar_location='above',
                 min_border_bottom=50)
    fig.line(x='timestamp', y='value', source=source)
    fig.yaxis.axis_label = format_variable_name(variable)
    fig.xaxis.axis_label = 'Time (UTC)'
    return fig


def align_index(df, interval_length):
    # If there is more than 3 days of data, limit the default x_range
    # to display only the most recent 3 day. Enable scrolling back in future
    # release.
    period_end = df.index[-1]
    x_range_start = df.index[df.index.get_loc(
        period_end - pd.Timedelta('3d'), method='bfill')]
    # align the data on the index it should have according to the metadata
    nindex = pd.date_range(start=x_range_start, end=period_end,
                           freq=f'{interval_length}min',
                           name='timestamp')
    df = df.reindex(nindex, axis=0)
    return df


def generate_figure(metadata, json_value_response):
    if 'forecast_id' in metadata:
        return generate_forecast_figure(metadata, json_value_response)
    else:
        return generate_observation_figure(metadata, json_value_response)


def generate_forecast_figure(metadata, json_value_response):
    series = json_payload_to_forecast_series(json_value_response)
    if series.empty:
        raise ValueError('No data')
    series = align_index(series, metadata['interval_length'])
    cds = ColumnDataSource(series)
    plot_width = 900
    fig = generate_basic_timeseries(series, metadata['name'],
                                    metadata['variable'], plot_width,
                                    source=cds)
    return components(fig)


def generate_observation_figure(metadata, json_value_response):
    """Creates a bokeh figure from API responses

    Parameters
    ----------
    metadata: dict
        Metadata dictionary used to label the plot. Must include
        a full nested site object. Only works with Metadata for
        types observation, forecast and cdf_forecast.

    json_response: dict
        The json response parsed into a dictionary.

    Raises
    ------
    ValueError
        When the supplied json contains an empty "values" field.
    """
    df = json_payload_to_observation_df(json_value_response)
    if df.empty:
        raise ValueError('No data')
    df = align_index(df, metadata['interval_length'])
    quality_flag = df.pop('quality_flag').astype(int)
    flags = quality_mapping.convert_mask_into_dataframe(quality_flag)
    flags = flags.mask(~flags)
    # need to fill as line needs more than a single point to show up
    if metadata['interval_label'] == ' ending':
        flags.bfill(axis=0, limit=1, inplace=True)
    else:
        # for interval beginning and instantaneous
        flags.ffill(axis=0, limit=1, inplace=True)

    cds = ColumnDataSource(pd.concat([df, flags], axis=1))
    plot_width = 900
    figs = [generate_basic_timeseries(df['value'], metadata['name'],
                                      metadata['variable'], plot_width,
                                      source=cds)]

    figs.extend(generate_quality_bars(flags, plot_width, figs[0].x_range,
                                      cds))
    layout = gridplot(figs,
                      ncols=1,
                      merge_tools=False,
                      sizing_mode='scale_width')
    return components(layout)
