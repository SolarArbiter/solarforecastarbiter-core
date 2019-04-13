# models_persistence.py?
"""
Functions for persistence forecasts.

Two kinds of persistence are supported:

  1. Persistence of observed values in :py:func:`persistence_scalar` and
     :py:func:`persistence_interval`
  2. Persistence of irradiance or power accounting for solar position in
     :py:func:`persistence_scalar_index` and
     :py:func:`persistence_interval_index` (?).

Users of intraday persistence forecasts will typically want to use
:py:func:`persistence_scalar` or :py:func:`persistence_scalar_index`.
Users of day ahead persistence forecasts will typically want to use
:py:func:`persistence_interval`.
:py:func:`persistence_interval_index`?

The functions accept a *load_data* keyword argument that allows users to
change where the functions load the observation data from. This is most
useful for users that would like to provide their own observation data
rather than using the solarforecastarbiter database.
"""
from functools import partial

import numpy as np
import pandas as pd

from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.io.utils import load_data  # does not exist yet
from solarforecastarbiter.reference_forecasts import forecast


def persistence_scalar(observation, data_start, data_end, forecast_start,
                       forecast_end, interval_length, load_data=load_data):
    r"""
    Make a persistence forecast using the mean value of the
    *observation* from *data_start* to *data_end*.

    In the example below, we use GHI to be concrete but the concept
    applies to any kind of observation data. The persistence forecast
    is:

    .. math::

       GHI_{t_f} = \overline{GHI_{t_{start}} \ldots GHI_{t_{end}}}

    where :math:`t_f` is a forecast time, and the overline represents
    the average of all observations that occur between
    :math:`t_{start}` = *data_start* and :math:`t_{end}` = *data_end*.

    Parameters
    ----------
    observation : datamodel.Observation
    data_start : pd.Timestamp
        Observation data start. Forecast is inclusive of this instant if
        observation.interval_label is *beginning* or *instantaneous*.
    data_end : pd.Timestamp
        Observation data end. Forecast is inclusive of this instant if
        observation.interval_label is *ending* or *instantaneous*.
    forecast_start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        observation.interval_label is *beginning* or *instantaneous*.
    forecast_end : pd.Timestamp
        Forecast end. Forecast is inclusive of this instant if
        observation.interval_label is *ending* or *instantaneous*.
    interval_length : pd.Timedelta
        Forecast interval length
    load_data : function, default solarforecastarbiter.io.utils.load_data
        A function that loads the observation data. Must have the same
        signature has the default function.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast. The forecast interval label is the
        same as the observation interval label.
    """
    obs = load_data(observation, data_start, data_end)
    persistence_quantity = obs.mean()
    closed = datamodel.CLOSED_MAPPING[observation.interval_label]
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
                             freq=interval_length, closed=closed)
    fx = pd.Series(persistence_quantity, index=fx_index)
    return fx


def persistence_interval(observation, data_start, data_end, forecast_start,
                         interval_length, load_data=load_data):
    r"""
    Make a persistence forecast for an *observation* using the mean
    values of each *interval_length* bin from *data_start* to
    *data_end*. The forecast starts at *forecast_start* and is of length
    *data_end* - *data_start*. A frequent use of this function is to
    create a day ahead persistence forecast using the previous day's
    observations.

    In the example below, we use GHI to be concrete but the concept
    applies to any kind of observation data. The persistence forecast
    for multiple intervals is:

    .. math::

       GHI_{t_{f_m}} = \overline{GHI_{t_{{start}_m}} \ldots GHI_{t_{{end}_m}}}

    where:

    .. math::

       m &\in \{0, 1, \ldots \frac{\textrm{data end} - \textrm{data start}}{\textrm{interval_length}} - 1\} \\
       t_{start_m} &=  \textrm{data start} + m \times \textrm{interval_length}  \\
       t_{end_m} &= \textrm{data start} + (1 + m) \times \textrm{interval_length} \\
       t_{f_m} &= \textrm{forecast start} + m \times \textrm{interval_length}  \\

    Further, persistence of multiple intervals requires that
    *data_start*, *data_end*, and *forecast_start* are all integer
    multiples of *interval_length*. For example, if *interval_length* =
    60 minutes, *data_start* may be 12:00 or 01:00, but not 12:30.

    Parameters
    ----------
    observation : datamodel.Observation
    data_start : pd.Timestamp
        Observation data start. Forecast is inclusive of this instant if
        observation.interval_label is *beginning* or *instantaneous*.
    data_end : pd.Timestamp
        Observation data end. Forecast is inclusive of this instant if
        observation.interval_label is *ending* or *instantaneous*.
    forecast_start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        observation.interval_label is *beginning* or *instantaneous*.
    interval_length : pd.Timedelta
        Forecast interval length
    load_data : function, default solarforecastarbiter.io.utils.load_data
        A function that loads the observation data. Must have the same
        signature has the default function.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast. The forecast interval label is the
        same as the observation interval label.
    """
    # ensure that we're using times rounded to multiple of interval_length
    [_check_interval_length(t, interval_length) for t in
     (data_start, data_end, forecast_start)]

    # get the data
    obs = load_data(observation, data_start, data_end)

    # average data within bins of length interval_length
    persistence_quantity = obs.resample(interval_length).mean()

    # Make the forecast time index.
    # determine forecast end from forecast start and obs length
    # assumes that load_data returned NaNs if data was missing.
    forecast_end = forecast_start + data_end - data_start
    closed = datamodel.CLOSED_MAPPING[observation.interval_label]
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
                             freq=interval_length, closed=closed)

    # Construct the returned series.
    # Use values to strip the time information from resampled obs.
    # Raises ValueError if len(persistence_quantity) != len(fx_index), but
    # that should never happen given the way we're calculating things here.
    fx = pd.Series(persistence_quantity.values, index=fx_index)
    return fx


def _persistence(observation, window, data_start, data_end,
                 forecast_start, forecast_end, interval_length,
                 load_data=load_data):
    r"""
    Make a persistence forecast for the observation.

    In the examples below, we use GHI to be concrete but the concept
    applies to any kind of observation data. In brief, the
    persistence forecast is:

    .. math::

       GHI(t_f) = GHI(t_0)

    where :math:`t_f` is a forecast time and :math:`t_0` is a reference
    time. However, complications occur when accounting for combinations
    of window, data start, data end, data interval length, and forecast
    interval length.

    Let *data start* and *data end* represent the start and end
    times of the data points that will be used to create the persistence
    forecast. Let *window* represent the time period over which data
    is averaged to create the persistence forecast.

    For persistence of a scalar quantity at all forecast times
    :math:`t_f`, the data averaging *window* should be greater than or
    equal to the data time period:
    *window* >= *data end* - *data start*. If
    *window* >= *data end* - *data start*,

    .. math::

       GHI_{t_f} = \overline{GHI_{t_{start}} \ldots GHI_{t_{end}}}

    where the overline represents the average of all observations that
    occur between :math:`t_{start}` = *data start* and
    :math:`t_{end}` = *data end*.

    For situations such as day ahead persistence forecasts of hourly
    average quantities, *window* < *data end* - *data start*. If
    *window* < *data end* - *data start*:

    .. math::

       GHI_{t_{f_m}} = \overline{GHI_{t_{{start}_m}} \ldots GHI_{t_{{end}_m}}}

    where:

    .. math::

       m &\in \{0, 1, \ldots \frac{\textrm{data end} - \textrm{data start}}{\textrm{window}} - 1\} \\
       t_{start_m} &=  \textrm{data start} + m \times \textrm{window}  \\
       t_{end_m} &= \textrm{data start} + (1 + m) \times \textrm{window} \\
       t_{f_m} &= \textrm{forecast start} + m \times \textrm{window}  \\

    Further, persistence of multiple values requires that:

      * *data end* - *data start* = *forecast end* - *forecast start*.
        The data time period is equal to the forecast time period.
      * (*data end* - *data start*) / *window* :math:`\in \mathbb{Z}`.
        The data time period is an integer multiple of the averaging
        window.
      * *window* = *interval length*. The data averaging window is
        equal to the forecast interval length.

    Further complications arise because of data interval labels
    (*beginning*, *ending*, *instantaneous*). If persistence of a scalar
    quantity is desired, *data end* - *data start* must be strictly less
    than (*interval label* = *instant*) or less than
    or equal (*interval label* = *beginning* or *ending*) to window.

    Parameters
    ----------
    observation : datamodel.Observation
    window : pd.Timedelta
        Time period over which to calculate the persistence quantity
        from the data.
    data_start : pd.Timestamp
        Data start
    data_end : pd.Timestamp
        Data end
    forecast_start : pd.Timestamp
        Forecast start
    forecast_end : pd.Timestamp
        Forecast end
    interval_length : pd.Timedelta
        Forecast interval length
    load_data : function, default solarforecastarbiter.io.utils.load_data
        A function that loads the observation data. Must have the same
        signature has the default function.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast. The forecast interval label is the
        same as the observation interval label.

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
    closed = _check_interval_closure(
        observation.interval_label, window, data_start, data_end,
        forecast_start, forecast_end, interval_length)
    # assumes data (possibly nan) exists from start to end
    obs = load_data(observation, data_start, data_end)
    # only support mean for now
    persistence_quantity = obs.resample(window, closed=closed).mean()
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
                             freq=interval_length, closed=closed)
    # housekeeping for scalar vs. array forecasts. raise exception if
    # calculated forecast is incompatible with desired forecast.
    persistence_quantity = _scalar_or_array_index(persistence_quantity,
                                                  fx_index)
    # will raise error if you've supplied incompatible data_start,
    # data_end, window, forecast_start, forecast_end, interval_length
    fx = pd.Series(persistence_quantity, index=fx_index)
    return fx


def persistence_scalar_index(observation, data_start, data_end, forecast_start,
                             forecast_end, interval_length,
                             load_data=load_data):
    r"""
    Calculate a persistence forecast using the mean value of the
    *observation* clear sky index or AC power index from *data_start* to
    *data_end*.

    In the example below, we use GHI to be concrete but the concept also
    applies to AC power. The persistence forecast is:

    .. math::

       GHI_{t_f} = \frac{\overline{GHI_{t_{start}} \ldots GHI_{t_{end}}}}{\overline{GHI_{{clear}_{t_{start}}} \ldots GHI_{{clear}_{t_{end}}}}} \times GHI_{{clear}_{t_f}}

    where :math:`t_f` is a forecast time, and the overline represents
    the average of all observations or clear sky values that occur
    between :math:`t_{start}` = *data_start* and
    :math:`t_{end}` = *data_end*.

    Parameters
    ----------
    observation : datamodel.Observation
    data_start : pd.Timestamp
        Observation data start. Forecast is inclusive of this instant if
        observation.interval_label is *beginning* or *instantaneous*.
    data_end : pd.Timestamp
        Observation data end. Forecast is inclusive of this instant if
        observation.interval_label is *ending* or *instantaneous*.
    forecast_start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        observation.interval_label is *beginning* or *instantaneous*.
    forecast_end : pd.Timestamp
        Forecast end. Forecast is inclusive of this instant if
        observation.interval_label is *ending* or *instantaneous*.
    interval_length : pd.Timedelta
        Forecast interval length
    load_data : function, default solarforecastarbiter.io.utils.load_data
        A function that loads the observation data. Must have the same
        signature has the default function.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast. The forecast interval label is the
        same as the observation interval label.
    """
    # ensure that we're using times rounded to multiple of interval_length
    [_check_interval_length(t, interval_length) for t in
     (data_start, data_end, forecast_start, forecast_end)]

    # get observation data for specified range
    obs = load_data(observation, data_start, data_end)

    # partial-up the metadata for solar position and
    # clearsky calculation clarity and consistency
    site = observation.site
    calc_solpos = partial(pvmodel.calculate_solar_position,
                          site.latitude, site.longitude, site.elevation)
    calc_cs = partial(pvmodel.calculate_clearsky,
                      site.latitude, site.longitude, site.elevation)

    # Calculate solar position and clearsky for obs time range.
    # minimum time resolution 5 minutes to reduce errors from
    # changing solar position during persistence data range.
    # Later, modeled clear sky or ac power will be averaged over the data range
    freq = min(interval_length, pd.Timedelta('5min'))
    closed = datamodel.CLOSED_MAPPING[observation.interval_label]
    obs_range = pd.date_range(start=data_start, end=data_end, freq=freq,
                              closed=closed)
    solar_position_obs = calc_solpos(obs_range)
    clearsky_obs = calc_cs(solar_position_obs['apparent_zenith'])

    # Calculate solar position and clearsky for the forecast times.
    # Use 5 minute or better frequency to minimize solar position errors.
    # Later, modeled clear sky or ac power will be resampled to interval_length
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

    # calculate persistence index (clear sky index or ac power index)
    # avg{index_{t_start}...index_{t_end}} =
    #   avg{obs_{t_start}...obs_{t_end}} / avg{clear_{t_start}...clear_{t_end}}
    # clear_ref is calculated at high temporal resolution, so this is accurate
    # for any observation interval length
    pers_index = obs.mean() / clear_ref.mean()

    # average instantaneous clear forecasts over interval_length windows
    # resample operation should be safe due to
    # _check_interval_length calls above
    clear_fx_resampled = \
        clear_fx.resample(interval_length, closed=closed).mean()

    # finally, make forecast
    # fx_t_f = avg{index_{t_start}...index_{t_end}} * clear_t_f
    fx = pers_index * clear_fx_resampled

    return fx


def _index_persistence(observation, window, data_start, data_end,
                       forecast_start, forecast_end, interval_length,
                       load_data=load_data):
    r"""
    Calculate persistence of clearsky index or AC power index forecast.

    In brief,

    .. math::

       GHI(t_0 + \Delta_t) = \frac{GHI(t_0)}{GHI_{clear}(t_0)} GHI_{clear}(t_0 + \Delta_t)

       AC(t_0 + \Delta_t) = \frac{AC(t_0)}{AC_{clear}(t_0)} AC_{clear}(t_0 + \Delta_t)

    however, complications exist when accounting for combinations of
    data interval length, data interval label, forecast interval length,
    and forecast interval label.

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
        from the data.
    data_start : pd.Timestamp
        Data start
    data_end : pd.Timestamp
        Data end
    forecast_start : pd.Timestamp
        Forecast start
    forecast_end : pd.Timestamp
        Forecast end
    interval_length : pd.Timedelta
        Forecast interval length
    load_data : function, default solarforecastarbiter.io.utils.load_data
        A function that loads the observation data. Must have the same
        signature has the default function.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast. The forecast interval label is the
        same as the observation interval label.

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
    closed = _check_interval_closure(
        observation.interval_label, window, data_start, data_end,
        forecast_start, forecast_end, interval_length)

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
    pers_index = _scalar_or_array_index(pers_index, clear_fx_resampled)

    # finally, make forecast
    # fx(dt + t_i) = index(t_i) * clear(dt + t_i)
    fx = pers_index * clear_fx_resampled

    return fx


def _check_interval_length(atime, interval_length):
    if atime.timestamp() % interval_length.total_seconds():
        raise ValueError('time must be integer multiple of interval_length')


def _check_interval_closure(interval_label, window, data_start, data_end,
                            forecast_start, forecast_end, interval_length):
    """Ensures valid input.

    Returns
    -------
    closed : None, 'left', or 'right'
        Parameter to be used with pandas DatetimeIndex operations

    Raises
    ------
    ValueError
    """
    closed = datamodel.CLOSED_MAPPING[interval_label]
    gte_or_gt_mapping = {
        'instant': np.greater_equal,
        'instantaneous': np.greater_equal,
        'beginning': np.greater,
        'ending': np.greater
    }
    gte_or_gt = gte_or_gt_mapping[interval_label]
    data_duration = data_end - data_start
    fx_duration = forecast_end - forecast_start
    if (gte_or_gt(data_duration, window) and (
            data_duration != fx_duration or window != interval_length)):
        # data_duration > window (beginning or ending label), or
        # data_duration >= window (instantaneous label), so we will eventually
        # persist multiple values. This requires that we assume that
        # data_duration == fx_duration, but the input data did not follow this
        # rule.
        raise ValueError(
            f'For observations with interval_label '
            f'{interval_label}, if data duration '
            f'{gte_or_gt.__name__} window, data duration must be '
            f'equal to forecast duration and window must be equal to '
            f'interval length. window = {window}, data duration = '
            f'{data_duration}, forecast duration = {fx_duration} '
            f'interval length = {interval_length}')

    return closed


def _scalar_or_array_index(pers_index, clear_fx_resampled):
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
    return pers_index
