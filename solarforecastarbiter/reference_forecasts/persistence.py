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

import pandas as pd

from solarforecastarbiter import datamodel, pvmodel


def persistence_scalar(observation, data_start, data_end, forecast_start,
                       forecast_end, interval_length, interval_label,
                       load_data):
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
        observation.interval_label is *beginning* or *instant*.
    data_end : pd.Timestamp
        Observation data end. Forecast is inclusive of this instant if
        observation.interval_label is *ending* or *instant*.
    forecast_start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* or *instant*.
    forecast_end : pd.Timestamp
        Forecast end. Forecast is inclusive of this instant if
        interval_label is *ending* or *instant*.
    interval_length : pd.Timedelta
        Forecast interval length
    interval_label : str
        instant, beginning, or ending
    load_data : function
        A function that loads the observation data. Must have the
        signature load_data(observation, data_start, data_end) and
        properly account for observation interval label.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast.
    """
    obs = load_data(observation, data_start, data_end)
    persistence_quantity = obs.mean()
    closed = datamodel.CLOSED_MAPPING[interval_label]
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
                             freq=interval_length, closed=closed)
    fx = pd.Series(persistence_quantity, index=fx_index)
    return fx


# Different signature than the scalar functions because forecast_end is
# determined by the forecast start and data length. Could make them the
# same for function call consistency, but readability counts.
def persistence_interval(observation, data_start, data_end, forecast_start,
                         interval_length, interval_label, load_data):
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

       m &\in \{0, 1, \ldots \frac{\textrm{data end} - \textrm{data start}}
           {\textrm{interval_length}} - 1\} \\
       t_{start_m} &=  \textrm{data start} +
           m \times \textrm{interval_length} \\
       t_{end_m} &= \textrm{data start} +
           (1 + m) \times \textrm{interval_length} \\
       t_{f_m} &= \textrm{forecast start} +
           m \times \textrm{interval_length} \\

    Further, persistence of multiple intervals requires that
    *data_start*, *data_end*, and *forecast_start* are all integer
    multiples of *interval_length*. For example, if *interval_length* =
    60 minutes, *data_start* may be 12:00 or 01:00, but not 12:30.

    Parameters
    ----------
    observation : datamodel.Observation
    data_start : pd.Timestamp
        Observation data start. Forecast is inclusive of this instant if
        observation.interval_label is *beginning* or *instant*.
    data_end : pd.Timestamp
        Observation data end. Forecast is inclusive of this instant if
        observation.interval_label is *ending* or *instant*.
    forecast_start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* or *instant*.
    interval_length : pd.Timedelta
        Forecast interval length
    interval_label : str
        instant, beginning, or ending
    load_data : function
        A function that loads the observation data. Must have the
        signature load_data(observation, data_start, data_end) and
        properly account for observation interval label.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast.
    """
    # determine forecast end from forecast start and obs length
    # assumes that load_data will return NaNs if data is missing.
    forecast_end = forecast_start + (data_end - data_start)

    # ensure that we're using times rounded to multiple of interval_length
    _check_intervals_times(observation.interval_label, data_start, data_end,
                           forecast_start, forecast_end,
                           observation.interval_length)

    # get the data
    obs = load_data(observation, data_start, data_end)

    # average data within bins of length interval_length
    closed_obs = datamodel.CLOSED_MAPPING[observation.interval_label]
    persistence_quantity = obs.resample(interval_length,
                                        closed=closed_obs).mean()

    # Make the forecast time index.
    closed = datamodel.CLOSED_MAPPING[interval_label]
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
                             freq=interval_length, closed=closed)

    # Construct the returned series.
    # Use values to strip the time information from resampled obs.
    # Raises ValueError if len(persistence_quantity) != len(fx_index), but
    # that should never happen given the way we're calculating things here.
    fx = pd.Series(persistence_quantity.values, index=fx_index)
    return fx


def persistence_scalar_index(observation, data_start, data_end, forecast_start,
                             forecast_end, interval_length, interval_label,
                             load_data):
    r"""
    Calculate a persistence forecast using the mean value of the
    *observation* clear sky index or AC power index from *data_start* to
    *data_end*.

    In the example below, we use GHI to be concrete but the concept also
    applies to AC power. The persistence forecast is:

    .. math::

       GHI_{t_f} = \overline{
           \frac{ GHI_{t_{start}} }{ GHI_{{clear}_{t_{start}}} } \ldots
           \frac{ GHI_{t_{end}} }{ GHI_{{clear}_{t_{end}}} } }

    where :math:`t_f` is a forecast time, and the overline represents
    the average of all observations or clear sky values that occur
    between :math:`t_{start}` = *data_start* and
    :math:`t_{end}` = *data_end*. All :math:`GHI_{t}/GHI_{{clear}_t}`
    ratios are restricted to the range [0, 2] before the average is
    computed.

    Parameters
    ----------
    observation : datamodel.Observation
    data_start : pd.Timestamp
        Observation data start. Forecast is inclusive of this instant if
        observation.interval_label is *beginning* or *instant*.
    data_end : pd.Timestamp
        Observation data end. Forecast is inclusive of this instant if
        observation.interval_label is *ending* or *instant*.
    forecast_start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* or *instant*.
    forecast_end : pd.Timestamp
        Forecast end. Forecast is inclusive of this instant if
        interval_label is *ending* or *instant*.
    interval_length : pd.Timedelta
        Forecast interval length
    interval_label : str
        instant, beginning, or ending
    load_data : function
        A function that loads the observation data. Must have the
        signature load_data(observation, data_start, data_end) and
        properly account for observation interval label.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast. The forecast interval label is the
        same as the observation interval label.
    """
    # ensure that we're using times rounded to multiple of interval_length
    _check_intervals_times(observation.interval_label, data_start, data_end,
                           forecast_start, forecast_end,
                           observation.interval_length)

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
    # if data is instantaneous, calculate at the obs time.
    # else (if data is interval average), calculate at 1 minute resolution to
    # reduce errors from changing solar position during persistence data range.
    # Later, modeled clear sky or ac power will be averaged over the data range
    closed = datamodel.CLOSED_MAPPING[observation.interval_label]
    if closed is None:
        freq = observation.interval_length
    else:
        freq = pd.Timedelta('1min')
    obs_range = pd.date_range(start=data_start, end=data_end, freq=freq,
                              closed=closed)
    solar_position_obs = calc_solpos(obs_range)
    clearsky_obs = calc_cs(solar_position_obs['apparent_zenith'])

    # Calculate solar position and clearsky for the forecast times.
    # Use 5 minute or better frequency to minimize solar position errors.
    # Later, modeled clear sky or ac power will be resampled to interval_length
    closed_fx = datamodel.CLOSED_MAPPING[interval_label]
    fx_range = pd.date_range(start=forecast_start, end=forecast_end, freq=freq,
                             closed=closed_fx)
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

    # resample sub-interval reference clear sky to observation intervals
    clear_ref_resampled = clear_ref.resample(
        observation.interval_length, closed=closed, label=closed).mean()
    # calculate persistence index (clear sky index or ac power index)
    # avg{index_{t_start}...index_{t_end}} =
    #   avg{obs_{t_start}/clear_{t_start}...obs_{t_end}/clear_{t_end}}
    # clear_ref is calculated at high temporal resolution, so this is accurate
    # for any observation interval length
    # apply clip to the clear sky index array before computing the average.
    # this prevents outliers from distorting the mean, a common occurance
    # near sunrise and sunset.
    pers_index = (obs / clear_ref_resampled).clip(lower=0, upper=2).mean()

    # average instantaneous clear forecasts over interval_length windows
    # resample operation should be safe due to
    # _check_interval_length calls above
    clear_fx_resampled = clear_fx.resample(
        interval_length, closed=closed_fx, label=closed_fx).mean()

    # finally, make forecast
    # fx_t_f = avg{index_{t_start}...index_{t_end}} * clear_t_f
    fx = pers_index * clear_fx_resampled

    return fx


def _check_intervals_times(interval_label, data_start, data_end,
                           forecast_start, forecast_end, interval_length):
    """Ensures valid input.

    Raises
    ------
    ValueError
    """
    interval_length_sec = int(interval_length.total_seconds())
    data_start_mod = data_start.timestamp() % interval_length_sec == 0
    forecast_start_mod = forecast_start.timestamp() % interval_length_sec == 0
    data_end_mod = data_end.timestamp() % interval_length_sec == 0
    forecast_end_mod = forecast_end.timestamp() % interval_length_sec == 0
    strvals = (
        f'interval_label={interval_label}, data_start={data_start}, '
        f'data_end={data_end}, forecast_start={forecast_start}, '
        f'forecast_end={forecast_end}, interval_length={interval_length_sec}s')
    if 'instant' in interval_label:
        # two options allowed:
        # 1. start times % int length == 0 and NOT end times % int. length == 0
        # 2. NOT start times % int length == 0 and end times % int. length == 0
        # everything else fails
        if data_start_mod and not data_end_mod:
            pass
        elif not data_start_mod and data_end_mod:
            pass
        else:
            raise ValueError('For observations with interval_label '
                             'instant, data_start OR data_end '
                             'must be must be divisible by interval_length.' +
                             strvals)
    elif interval_label in ['ending', 'beginning']:
        if not all((data_start_mod, forecast_start_mod,
                    data_end_mod, forecast_end_mod)):
            raise ValueError('For observations with interval_label beginning '
                             'or ending, all of data_start, forecast_start, '
                             'data_end, and forecast_end must be divisible by '
                             'interval_length. ' + strvals)
    else:
        raise ValueError('invalid interval_label')


def persistence_dayofweek(observation, forecast_start, forecast_end,
                          interval_length, interval_label, load_data):
    r"""
    Make a persistence forecast for an *observation* using the mean values of
    each *interval_length* bin from the same day of the week from the prior
    week, e.g., use data from Saturday April 4th 2020 to predict Saturday April
    11th 2020.

    This type of persistence forecast is useful as a baseline for load
    forecasting as most regions exhibit clear trends in load for each day of
    the week. For example, the load on a Monday tends to look more similar to
    the load from the prior Monday than it does to the load from either the
    prior day (Sunday) or the next day (Tuesday).

    Parameters
    ----------
    observation : datamodel.Observation
    forecast_start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* or *instant*.
    forecast_end : pd.Timestamp
        Forecast end. Forecast is inclusive of this instant if
        interval_label is *ending* or *instant*.
    interval_length : pd.Timedelta
        Forecast interval length
    interval_label : str
        instant, beginning, or ending
    load_data : function
        A function that loads the observation data. Must have the
        signature load_data(observation, data_start, data_end) and
        properly account for observation interval label.

    Returns
    -------
    forecast : pd.Series
        The persistence forecast.

    See Also
    --------
    :py:func:`solarforecastarbiter.reference_forecasts.persistence.persistence_interval`

    Notes
    -----
    This function requires that observation data exists from 1 week (7 days)
    prior to the *forecast_start*. It therefore is not suitable for forecast
    horizons greater than 7 days. Additionally, this function does not correct
    for shorter timescale trends and therefore will be less accurate in
    situations where there are signficiant changes week-to-week, e.g., a week
    of moderate temperature followed by a heatwave that drives up demand from
    increased A/C usage.

    """

    # use data from the same day of week from the prior week
    data_start = forecast_start - pd.Timedelta("7D")
    data_end = forecast_end - pd.Timedelta("7D")
    print(type(data_start), type(data_end), type(forecast_start))

    fx = persistence_interval(observation, data_start, data_end,
                              forecast_start, interval_length, interval_label,
                              load_data)
    return fx
