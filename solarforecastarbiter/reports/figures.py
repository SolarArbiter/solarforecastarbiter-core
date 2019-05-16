"""
Plots.
"""

from dataclasses import dataclass

from bokeh.embed import components
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh import palettes

import pandas as pd


PALETTE = palettes.d3['Category10'][6]


def format_variable_name(variable):
    """Make a nice human readable name."""
    caps = ('ac_', 'dc_', 'poa_', 'ghi', 'dni', 'dhi')
    fname = variable
    for cap in caps:
        fname = fname.replace(cap, cap.upper())
    fname = fname.replace('_', ' ')
    return fname


def line_or_step(interval_label):
    if 'instant' in interval_label:
        plot_method = 'line'
        kwargs = dict()
    elif interval_label == 'beginning':
        plot_method = 'step'
        kwargs = dict(mode='before')
    elif interval_label == 'ending':
        plot_method = 'step'
        kwargs = dict(mode='after')
    return plot_method, kwargs


def construct_fx_obs_cds(fx_obs, fx_values, obs_values):
    """
    Construct a tuple of metadata, data for the plot functions.

    Parameters
    ----------
    fx_obs : solarforecastarbiter.datamodel.ForecastObservation
    fx_values : pandas.Series
    obs_values : pandas.Series

    Returns
    -------
    fx_obs : solarforecastarbiter.datamodel.ForecastObservation
    cds : bokeh.models.ColumnDataSource
    """
    data = pd.DataFrame({'observation': obs_values, 'forecast': fx_values})
    data['timestamp'] = data.index
    cds = ColumnDataSource(data)
    return fx_obs, cds


def _obs_name(fx_obs):
    if fx_obs.forecast.name == fx_obs.observation.name:
        return fx_obs.forecast.name + ' Observation'


def _fx_name(fx_obs):
    if fx_obs.forecast.name == fx_obs.observation.name:
        return fx_obs.forecast.name + ' Forecast'


def timeseries(fx_obs_cds, start, end):
    """
    Timeseries plot of one or more forecasts and observations.

    Parameters
    ----------
    obs_fx_cds : tuple of (ForecastObservation, cds) tuples
        ForecastObservation is a datamodel.ForecastObservation object.
        cds is a Bokeh ColumnDataSource with columns
        timestamp, observation, forecast.
    start : pandas.Timestamp
        Report start time
    end : pandas.Timestamp
        Report end time

    Returns
    -------
    fig : bokeh.plotting.figure
    """

    palette = iter(PALETTE)

    fig = figure(
        sizing_mode='scale_width', plot_width=900, plot_height=300,
        x_range=(start, end), x_axis_type='datetime',
        tools='pan,wheel_zoom,reset')

    plotted_objects = []
    for fx_obs, cds in fx_obs_cds:
        if fx_obs.observation in plotted_objects:
            pass
        else:
            plotted_objects.append(fx_obs.observation)
            plot_method, kwargs = line_or_step(
                fx_obs.observation.interval_label)
            name = _obs_name(fx_obs)
            getattr(fig, plot_method)(
                x='timestamp', y='observation', source=cds,
                color='black', legend=name,  **kwargs)
        if fx_obs.forecast in plotted_objects:
            pass
        else:
            plotted_objects.append(fx_obs.forecast)
            plot_method, kwargs = line_or_step(
                fx_obs.forecast.interval_label)
            name = _fx_name(fx_obs)
            getattr(fig, plot_method)(
                x='timestamp', y='forecast', source=cds,
                color=next(palette), legend=name,  **kwargs)

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    fig.xaxis.axis_label = 'Time'
    fig.yaxis.axis_label = format_variable_name(fx_obs.forecast.variable)

    return fig


def scatter():
    raise NotImplementedError


def bar():
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
