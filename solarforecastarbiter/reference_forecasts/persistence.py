# models_persistence.py?
"""
Functions for persistence forecasts.
"""
from functools import partial

import numpy as np
import pandas as pd

from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.io.utils import load_data  # does not exist
from solarforecastarbiter.reference_forecasts import forecast


def persistence(observation, window, data_start, data_end,
                forecast_start, forecast_end, interval_length,
                load_data=load_data):
    """
    Make a persistence forecast for the observation.

    Parameters
    ----------
    observation : datamodel.Observation
    window : pd.Timedelta
        Time period over which to calculate the persistence quantity
    data_start : pd.Timestamp
        Data start
    data_end : pd.Timestamp
        Data end
    forecast_start : pd.Timestamp
        Forecast start
    forecast_end : pd.Timestamp
        Forecast end
    interval_length : str
        Forecast interval length e.g. '5min' or '1h'
    load_data : function
        A function that loads the observation data.

    Returns
    -------
    forecast : pd.Series
    """
    # assumes data (possibly nan) exists from start to end
    obs = load_data(observation, data_start, data_end)
    # only support mean for now
    # np.array strips datetime information in a
    # scalar and Series compatible way
    persistence_quantity = np.array(obs.resample(window).mean())
    fx_index = pd.DatetimeIndex(start=forecast_start, end=forecast_end,
                                freq=interval_length)
    # will raise error if you've supplied incompatible data_start,
    # data_end, window, forecast_start, forecast_end, interval_length
    fx = pd.Series(persistence_quantity, index=fx_index)
    return fx


def index_persistence(observation, window, data_start, data_end,
                      forecast_start, forecast_end, interval_length,
                      load_data=load_data):
    """
    Calculate persistence of clearsky index or AC power index forecast.

    Indicies are calculated using subhourly calculations of clearsky
    irradiance or power that are then resampled to the observation
    window.

    Be careful with start times, end times, window, and interval length.
    For example, if persistence of a scalar quantity is desired,
    data_end - data_start must be strictly less than window.

    Parameters
    ----------
    observation : datamodel.Observation
    window : pd.Timedelta
        Time period over which to calculate the persistence quantity
    data_start : pd.Timestamp
        Data start
    data_end : pd.Timestamp
        Data end
    forecast_start : pd.Timestamp
        Forecast start
    forecast_end : pd.Timestamp
        Forecast end
    interval_length : str
        Forecast interval length e.g. '5min' or '1h'
    load_data : function
        A function that loads the observation data.

    Returns
    -------
    forecast : pd.Series
        Returned forecast has interval label beginning.
    """
    # get observation data for specified range
    obs = load_data(observation, data_start, data_end)

    # for consistency, define resampler function to be used on obs
    # and, later, clearsky irradiance or ac power
    resampler = partial(forecast.resample, freq=window)
    obs_resampled = resampler(obs)

    # partial-up the metadata for clarity below
    site = observation.site
    calc_solpos = partial(pvmodel.calculate_solar_position,
                          site.latitude, site.longitude, site.elevation)
    calc_cs = partial(pvmodel.calculate_clearsky,
                      site.latitude, site.longitude, site.elevation)

    # calculate solar position and clearsky for obs time range
    # minimum time resolution 5 minutes so to reduce errors from
    # changing solar position during persistence window.
    # clear sky or ac power will be resampled below
    freq = min(window, pd.Timedelta('5min'))
    obs_range = pd.DatetimeIndex(start=data_start, end=data_end, freq=freq)
    solar_position_obs = calc_solpos(obs_range)
    clearsky_obs = calc_cs(solar_position_obs['apparent_zenith'])

    # calculate solar position and clearsky for forecast times
    fx_range = pd.DatetimeIndex(start=forecast_start, end=forecast_end,
                                freq=freq)
    solar_position_fx = calc_solpos(fx_range)
    clearsky_fx = calc_cs(solar_position_fx['apparent_zenith'])

    if isinstance(site, datamodel.SolarPowerPlant):
        # no temperature input is ok so long as temperature effects
        # do not push the system above or below AC clip point
        clear_ref = pvmodel.irradiance_to_power(
            site.modeling_parameters, solar_position_obs['apparent_zenith'],
            solar_position_obs['azimuth'], clearsky_obs['ghi'],
            clearsky_obs['dni'], clearsky_obs['dhi']
        )
        clear_fx = pvmodel.irradiance_to_power(
            site.modeling_parameters, solar_position_fx['apparent_zenith'],
            solar_position_fx['azimuth'], clearsky_fx['ghi'],
            clearsky_fx['dni'], clearsky_fx['dhi']
        )
    else:
        clear_ref = clearsky_obs[observation.variable]
        clear_fx = clearsky_fx[observation.variable]

    # np.array strips datetime information in a
    # scalar and Series compatible way
    clear_ref_resampled = resampler(clear_ref)
    pers_index = np.array(obs_resampled) / np.array(clear_ref_resampled)
    clear_fx_resampled = resampler(clear_fx)
    fx = pers_index * np.array(clear_fx_resampled)

    fx_index = pd.DatetimeIndex(start=forecast_start, end=forecast_end,
                                freq=interval_length)
    # If we're persisting a single value, the code below pulls out that value
    # from a 1-length array. If not, it passes the data onto the Series
    # constructor in hopes that you've specified consistent windows
    try:
        fx = fx.item()
    except ValueError:
        # not 1d, hope it's the right length when we make the series!
        pass
    finally:
        # will raise error if you've supplied incompatible data_start,
        # data_end, window, forecast_start, forecast_end, interval_length
        fx = pd.Series(fx, index=fx_index)
    return fx
