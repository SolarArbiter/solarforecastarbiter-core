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
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
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
    If persistence of a scalar quantity is desired, data_end -
    data_start must be less than (interval label = instant) or less than
    or equal (interval label = beginning or ending) to window. If
    persistence of multiple values is desired, data_end - data_start
    must equal to forecast_end - forecast_start.

    Parameters
    ----------
    observation : datamodel.Observation
        Must be AC Power, GHI, DNI, or DHI.
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
        The persistence forecast. The forecast interval label equal to
        the observation interval label.

    Raises
    ------
    ValueError
        If window > data_end - data_start and data_end - data_start !=
        forecast_end - forecast_start.

    ValueError
        If calculated forecast length is inconsistent with desired
        forecast length.
    """
    # consistency checks for labels, data duration, and forecast duration
    closed = datamodel.CLOSED_MAPPING[observation.interval_label]
    gte_or_gt_mapping = {
        'instant': np.greater_equal,
        'instantaneous': np.greater_equal,
        'beginning': np.greater,
        'ending': np.greater
    }
    gte_or_gt = gte_or_gt_mapping[observation.interval_label]
    data_duration = data_end - data_start
    fx_duration = forecast_end - forecast_start
    if gte_or_gt(data_duration, window) and data_duration != fx_duration:
        # data_duration > window (beginning or ending label), or
        # data_duration >= window (instantaneous label), so we will eventually
        # persist multiple values. This requires that we assume that
        # data_duration == fx_duration, but the input data did not follow this
        # rule.
        raise ValueError(
            f'For observations with interval_label '
            f'{observation.interval_label}, if data duration '
            f'{gte_or_gt.__name__} window, data duration must be '
            f'equal to forecast duration. window = {window}, data duration = '
            f'{data_duration}, forecast duration = {fx_duration}')

    # get observation data for specified range
    obs = load_data(observation, data_start, data_end)

    # for consistency, define resampler function to be used on obs
    # and, later, reference clearsky irradiance or clearsky ac power
    resampler = partial(forecast.resample, freq=window, closed=closed)
    obs_resampled = resampler(obs)

    # partial-up the metadata for solar position and
    # clearsky calculation clarity and consistency
    site = observation.site
    calc_solpos = partial(pvmodel.calculate_solar_position,
                          site.latitude, site.longitude, site.elevation)
    calc_cs = partial(pvmodel.calculate_clearsky,
                      site.latitude, site.longitude, site.elevation)

    # Calculate solar position and clearsky for obs time range.
    # minimum time resolution 5 minutes to reduce errors from
    # changing solar position during persistence window.
    # Later, clear sky or ac power will be resampled
    freq = min(window, pd.Timedelta('5min'))
    obs_range = pd.date_range(start=data_start, end=data_end, freq=freq,
                              closed=closed)
    solar_position_obs = calc_solpos(obs_range)
    clearsky_obs = calc_cs(solar_position_obs['apparent_zenith'])

    # Calculate solar position and clearsky for intra hour forecast times.
    # Later, clear sky or ac power will be resampled.
    fx_range = pd.date_range(start=forecast_start, end=forecast_end, freq=freq,
                             closed=closed)
    solar_position_fx = calc_solpos(fx_range)
    clearsky_fx = calc_cs(solar_position_fx['apparent_zenith'])

    # Consider putting the code within each if/else block below into its own
    # function with standard outputs clear_ref and clear_fx. But with only two
    # cases for now, it might be more clear to leave inline.
    if isinstance(site, datamodel.SolarPowerPlant):
        # No temperature input is only OK so long as temperature effects
        # do not push the system above or below AC clip point.
        # It's only a reference forecast!
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
        # assume we are working with ghi, dni, or dhi.
        clear_ref = clearsky_obs[observation.variable]
        clear_fx = clearsky_fx[observation.variable]

    # resample reference quantity (average over window)
    clear_ref_resampled = resampler(clear_ref)
    # calculate persistence index (clear sky index or ac power index)
    # index(t_i) = obs(t_i) / clear(t_i)
    pers_index = obs_resampled / clear_ref_resampled
    # resample forecast clear sky reference (average over window)
    clear_fx_resampled = \
        clear_fx.resample(interval_length, closed=closed).mean()

    # housekeeping for scalar vs. array forecasts. raise exception if
    # calculated forecast is incompatible with desired forecast.
    if len(pers_index) == 1:
        # scalar forecast (data_end - data_start <= window)
        pers_index = pers_index.values[0]
    elif len(pers_index) == len(clear_fx_resampled):
        # array forecast (data_end - data_start >= window)
        pers_index = pers_index.values
    else:
        # should not be reachable thanks to initial restrictions on
        # data length and forecast length.
        # TODO: confirm this is true and remove
        raise ValueError(
            f'Incompatible inputs. '
            f'Data inputs produced forecast of length {len(pers_index)}. '
            f'Forecast inputs require forecast of length '
            f'{len(clear_fx_resampled)}.')

    # finally, make forecast
    # fx(dt + t_i) = index(t_i) * clear(dt + t_i)
    fx = pers_index * clear_fx_resampled

    return fx
