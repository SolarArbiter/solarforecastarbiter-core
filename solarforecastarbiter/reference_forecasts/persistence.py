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

from statsmodels.distributions.empirical_distribution import ECDF


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

    closed_obs = datamodel.CLOSED_MAPPING[observation.interval_label]
    # get the data
    obs = load_data(observation, data_start, data_end)
    obs_index = pd.date_range(start=data_start, end=data_end,
                              freq=observation.interval_length,
                              closed=closed_obs)
    # put in nans if appropriate
    obs = obs.reindex(obs_index)
    # average data within bins of length interval_length
    persistence_quantity = obs.resample(interval_length,
                                        closed=closed_obs).mean()

    # Make the forecast time index.
    closed = datamodel.CLOSED_MAPPING[interval_label]
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
                             freq=interval_length, closed=closed)

    # Construct the returned series.
    # Use values to strip the time information from resampled obs.
    # Raises ValueError if len(persistence_quantity) != len(fx_index), but
    # that should never happen given the way we're calculating things here
    # now that nans are insterted into obs for missing data.
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
    if (
            isinstance(site, datamodel.SolarPowerPlant) and
            observation.variable == 'ac_power'
    ):
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


def persistence_probabilistic(observation, data_start, data_end,
                              forecast_start, forecast_end,
                              interval_length, interval_label,
                              load_data, axis, constant_values):
    r"""
    Make a probabilistic persistence forecast using the *observation* from
    *data_start* to *data_end*. In the forecast literature, this method is
    typically referred to as Persistence Ensemble (PeEn). [1]_ [2]_ [3]_

    The function handles forecasting either constant variable values or
    constant percentiles. In the examples below, we use GHI to be concrete but
    the concepts also apply to other variables (AC power, net load, etc.).

    If forecasting constant variable values (e.g. forecast the probability of
    GHI being less than or equal to 500 W/m^2), the persistence forecast is:

    .. math::

       F_n(x) = ECDF(GHI_{t_{start}}, ..., GHI_{t_{end}})
       Prob(GHI_{t_f} <= 100 W/m^2) = F_n(100 W/m^2)

    where :math:`t_f` is a forecast time and :math:`F_n` is the empirical CDF
    (ECDF) function computed from the *n* observations between
    :math:`t_{start}` = *data_start* and :math:`t_{end}` = *data_end*, which
    maps from variable values to probabilities.

    If forecasting constant probabilities (e.g. forecast the GHI value that has
    a 50% probability), the persistence forecast is:

    .. math::

       F_n(x) = ECDF(GHI_{t_{start}}, ..., GHI_{t_{end}})
       Q_n(p) = \inf {x \in \mathrf{R} : p \leq F_n(x) }
       p_{t_f} = Q_n(50%)

    where :math:`Q_n` is the quantile function based on the *n* observations
    between :math:`t_{start}` = *data_start* and :math:`t_{end}` = *data_end*,
    which maps from probabilities to variable values.

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
    axis : {'x', 'y'}
        The axis on which the constant values of the CDF is specified. The axis
        can be either *x* (constant variable values) or *y* (constant
        percentiles).
    constant_values : array_like
        The variable values or percentiles.

    Returns
    -------
    forecasts : list of pd.Series
        The persistence forecasts, returned in the same order as
        *constant_values*. If *axis* is *x*, the forecast values are
        percentiles (e.g. 25%). If instead *axis* is *y*, the forecasts values
        have the same units as the observation data (e.g. MW).

    Raises
    ------
    ValueError
        If the **axis** parameter is invalid.

    References
    ----------
    .. [1] Allessandrini et al. (2015) "An analog ensemble for short-term
       probabilistic solar power forecast", Appl. Energy 157, pp. 95-110.
       doi: 10.1016/j.apenergy.2015.08.011
    .. [2] Yang (2019) "A universal benchmarking method for probabilistic
       solar irradiance forecasting", Solar Energy 184, pp. 410-416.
       doi: 10.1016/j.solener.2019.04.018
    .. [3] Doubleday, Van Scyoc Herndandez and Hodge (2020) "Benchmark
       probabilistic solar forecasts: characteristics and recommendations",
       Solar Energy 206, pp. 52-67. doi: 10.1016/j.solener.2020.05.051

    """
    closed = datamodel.CLOSED_MAPPING[interval_label]
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
                             freq=interval_length, closed=closed)

    # observation data resampled to match forecast sampling
    obs = load_data(observation, data_start, data_end)
    obs = obs.resample(interval_length, closed=closed).mean()

    if obs.empty:
        return [pd.Series(None, dtype=float, index=fx_index)
                for _ in constant_values]

    if axis == "x":
        cdf = ECDF(obs)
        forecasts = []
        for constant_value in constant_values:
            fx_prob = cdf(constant_value) * 100.0
            forecasts.append(pd.Series(fx_prob, index=fx_index))
    elif axis == "y":   # constant_values=percentiles, fx=variable
        forecasts = []
        for constant_value in constant_values:
            fx = np.percentile(obs, constant_value)
            forecasts.append(pd.Series(fx, index=fx_index))
    else:
        raise ValueError(f"Invalid axis parameter: {axis}")

    return forecasts


def persistence_probabilistic_timeofday(observation, data_start, data_end,
                                        forecast_start, forecast_end,
                                        interval_length, interval_label,
                                        load_data, axis, constant_values):
    r"""
    Make a probabilistic persistence forecast using the *observation* from
    *data_start* to *data_end*, matched by time of day (e.g. to forecast 9am,
    only use observations from 9am on days between *data_start* and
    *data_end*). This is a common variant of the Persistence Ensemble (PeEn)
    method. [1]_ [2]_ [3]_

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
    axis : {'x', 'y'}
        The axis on which the constant values of the CDF is specified. The axis
        can be either *x* (constant variable values) or *y* (constant
        percentiles).
    constant_values : array_like
        The variable values or percentiles.

    Returns
    -------
    forecasts : list of pd.Series
        The persistence forecasts, returned in the same order as
        *constant_values*. If *axis* is *x*, the forecast values are
        percentiles (e.g. 25%). If instead *axis* is *y*, the forecasts values
        have the same units as the observation data (e.g. MW).

    Raises
    ------
    ValueError
        If there is insufficient data for matching by time of day or the
        **axis** parameter is invalid.

    Notes
    -----
    Assumes that there is at least 20 days of *observation* data available
    based on [1]_, [2]_, [3]_.

    References
    ----------
    .. [1] Allessandrini et al. (2015) "An analog ensemble for short-term
       probabilistic solar power forecast", Appl. Energy 157, pp. 95-110.
       doi: 10.1016/j.apenergy.2015.08.011
    .. [2] Yang (2019) "A universal benchmarking method for probabilistic
       solar irradiance forecasting", Solar Energy 184, pp. 410-416.
       doi: 10.1016/j.solener.2019.04.018
    .. [3] Doubleday, Van Scyoc Herndandez and Hodge (2020) "Benchmark
       probabilistic solar forecasts: characteristics and recommendations",
       Solar Energy 206, pp. 52-67. doi: 10.1016/j.solener.2020.05.051

    See also
    --------
    :py:func:`solarforecastarbiter.reference_forecasts.persistence.persistence_probabilistic`

    """
    # ensure that we're using times rounded to multiple of interval_length
    _check_intervals_times(observation.interval_label, data_start, data_end,
                           forecast_start, forecast_end,
                           observation.interval_length)

    closed = datamodel.CLOSED_MAPPING[interval_label]
    fx_index = pd.date_range(start=forecast_start, end=forecast_end,
                             freq=interval_length, closed=closed)

    # observation data resampled to match forecast sampling
    obs = load_data(observation, data_start, data_end)
    obs = obs.resample(interval_length, closed=closed).mean()
    if obs.empty:
        raise ValueError("Insufficient data to match by time of day")

    # time of day: minutes past midnight (e.g. 0=12:00am, 75=1:15am)
    if obs.index.tzinfo is not None:
        if fx_index.tzinfo is not None:
            obs = obs.tz_convert(fx_index.tzinfo)
        else:
            fx_index = fx_index.tz_localize(obs.index.tzinfo)
    else:
        if fx_index.tzinfo is not None:
            obs = obs.tz_localize(fx_index.tzinfo)

    obs_timeofday = (obs.index.hour * 60 + obs.index.minute).astype(int)
    fx_timeofday = (fx_index.hour * 60 + fx_index.minute).astype(int)

    # confirm sufficient data for matching by time of day
    if obs.last_valid_index() - obs.first_valid_index() < pd.Timedelta("20D"):
        raise ValueError("Insufficient data to match by time of day")

    if axis == "x":
        forecasts = []
        for constant_value in constant_values:
            fx = pd.Series(np.nan, index=fx_index)
            for tod in fx_timeofday:
                data = obs[obs_timeofday == tod]
                cdf = ECDF(data)
                fx[fx_timeofday == tod] = cdf(constant_value) * 100.0
            forecasts.append(fx)
    elif axis == "y":
        forecasts = []
        for constant_value in constant_values:
            fx = pd.Series(np.nan, index=fx_index)
            for tod in fx_timeofday:
                data = obs[obs_timeofday == tod]
                fx[fx_timeofday == tod] = np.percentile(data, constant_value)
            forecasts.append(fx)
    else:
        raise ValueError(f"Invalid axis parameter: {axis}")

    return forecasts
