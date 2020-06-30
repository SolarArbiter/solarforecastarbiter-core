import numpy as np
import pandas as pd
import pytz


from solarforecastarbiter.io import utils as io_utils


def get_issue_times(forecast, start_from):
    """Return a list of the issue times for a given Forecast starting
    from the date of `start_from` until the first issue time of the next day.
    The output timestamps are localized to the timezone of `start_from`.

    Parameters
    ----------
    forecast : datamodel.Forecast
        Forecast object that contains the time information
    start_from : pandas.Timestamp
        Return issue times from this same day in the same timezone

    Returns
    -------
    list
        pandas.Timestamp objects with the issues times for the particular day
        including the first issue time for the next day.
    """
    start_time = forecast.issue_time_of_day
    if start_time.tzinfo is None:
        start_time = pytz.utc.localize(start_time)
    # work for forecasts over 1d run length
    dayadj = pd.Timedelta(forecast.run_length).ceil('1d')
    # make broad range of times that should cover start_from and next time
    # even after timezone conversion
    earliest_start = pd.Timestamp.combine(
        (start_from - dayadj).date(), start_time)
    possible_times = []
    for i in range(3):
        start = earliest_start + i * dayadj
        end = (start + dayadj).floor('1d')
        possible_times.extend(list(
            pd.date_range(start=start, end=end, freq=forecast.run_length)))
    possible_times = pd.DatetimeIndex(possible_times).tz_convert(
        start_from.tz).drop_duplicates()
    # then slice the broad range based on start_from day
    startloc = possible_times.get_loc(start_from.floor('1d'), method='bfill')
    endloc = possible_times.get_loc(
        (start_from + pd.Timedelta('1d')).floor('1d'), method='bfill') + 1
    return list(possible_times[startloc:endloc])


def get_next_issue_time(forecast, run_time):
    """Determine the next issue time from a forecast and run time
    """
    issue_times = get_issue_times(forecast, run_time)
    idx = np.searchsorted(issue_times, run_time)
    return issue_times[idx]


def get_init_time(run_time, fetch_metadata):
    """Determine the most recent init time for which all forecast data is
    available."""
    run_finish = (pd.Timedelta(fetch_metadata['delay_to_first_forecast']) +
                  pd.Timedelta(fetch_metadata['avg_max_run_length']))
    freq = fetch_metadata['update_freq']
    init_time = (run_time - run_finish).floor(freq=freq)
    return init_time


def get_forecast_start_end(forecast, issue_time,
                           adjust_for_interval_label=False):
    """
    Get absolute forecast start from *forecast* object parameters and
    absolute *issue_time*.

    Parameters
    ----------
    forecast : datamodel.Forecast
    issue_time : pd.Timestamp
    adjust_for_interval_label : boolean
        If True, adds or subtracts a nanosecond from the start or end
        time based value of forecast.interval_label

    Returns
    -------
    forecast_start : pd.Timestamp
        Start time of forecast issued at issue_time
    forecast_end : pd.Timestamp
        End time of forecast issued at issue_time

    Raises
    ------
    ValueError if forecast and issue_time are incompatible
    """
    issue_times = get_issue_times(forecast, issue_time)
    if issue_time not in issue_times:
        raise ValueError(
            ('Incompatible forecast.issue_time_of_day %s, '
             'forecast.run_length %s, and issue_time %s') % (
             forecast.issue_time_of_day, forecast.run_length, issue_time))
    forecast_start = issue_time + forecast.lead_time_to_start
    forecast_end = forecast_start + forecast.run_length
    if adjust_for_interval_label:
        forecast_start, forecast_end = \
            io_utils.adjust_start_end_for_interval_label(
                forecast.interval_label, forecast_start, forecast_end, True)
    return forecast_start, forecast_end


def find_next_issue_time_from_last_forecast(forecast, last_forecast_time):
    """
    Find the next issue time for *forecast* based on the timestamp of the
    last forecast value. If *last_forecast_time* is not the end of a forecast
    run, the issue time returned will be the issue time that overwrites
    *last_forecast_time* with a full length forecast.

    Parameters
    ----------
    forecast : datamodel.Forecast
    last_forecast_time : pd.Timestamp
        Last timestamp available for the forecast

    Returns
    -------
    pd.Timestamp
        The next issue time for the forecast
    """
    last_probable_issue_time = (last_forecast_time -
                                forecast.run_length -
                                forecast.lead_time_to_start)
    # for beginning & instantaneous labels, last_probable_issue_time
    # is currently e.g. 13:55 - 60 min - 0 min = 12:55, so add the
    # interval_length to make last_probable_issue_time = 13:00
    if forecast.interval_label != 'ending':
        last_probable_issue_time += forecast.interval_length
    next_issue_time = get_next_issue_time(
        forecast, last_probable_issue_time + pd.Timedelta('1ns'))
    return next_issue_time


def _is_intraday(forecast):
    """Is the forecast intraday?"""
    # intra day persistence and "day ahead" persistence require
    # fairly different parameters.
    # is this a sufficiently robust way to distinguish?
    return forecast.run_length < pd.Timedelta('1d')


def _intraday_start_end(observation, forecast, run_time):
    """
    Time range of data to be used for intra-day persistence forecast.

    Parameters
    ----------
    observation : datamodel.Observation
    forecast : datamodel.Forecast
    run_time : pd.Timestamp

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp
    """
    _check_intraday_compatibility(observation, forecast)
    # no longer than 1 hour
    window = min(forecast.run_length, pd.Timedelta('1hr'))
    data_end = run_time
    data_start = data_end - window
    return data_start, data_end


def _dayahead_start_end(issue_time, forecast):
    """
    Time range of data to be used for day-ahead persistence forecast.

    Parameters
    ----------
    issue_time : pd.Timestamp
    forecast : datamodel.Forecast

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp

    Notes
    -----
    Day-ahead persistence: uses the most recently available data that
    maintains same times in forecast and observation data,
    but shifts observation period by a number of days to end before
    issue time.
    """
    # data_end = last forecast time for next issue of run - 1 day
    data_end = issue_time + forecast.lead_time_to_start + forecast.run_length
    # data end should end before, not at issue time, so add the extra ns
    data_end -= (data_end - issue_time + pd.Timedelta('1ns')).ceil('1d')
    data_start = data_end - forecast.run_length
    return data_start, data_end


def _weekahead_start_end(issue_time, forecast):
    """
    Time range of data to be used for week-ahead persistence, aka, day of week
    persistence.

    Parameters
    ----------
    issue_time : pd.Timestamp
    lead_time : pd.Timedelta

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp

    """
    data_start = issue_time + forecast.lead_time_to_start - pd.Timedelta('7d')
    data_end = data_start + forecast.run_length
    return data_start, data_end


def _adjust_for_instant_obs(data_start, data_end, observation, forecast):
    # instantaneous observations require care.
    # persistence models return forecasts with same closure as obs
    if 'instant' in forecast.interval_label:
        data_end -= pd.Timedelta('1s')
    elif forecast.interval_label == 'beginning':
        data_end -= pd.Timedelta('1s')
    else:
        data_start += pd.Timedelta('1s')
    return data_start, data_end


def get_data_start_end(observation, forecast, run_time, issue_time):
    """
    Determine the data start and data end times for a persistence
    forecast. For non-intraday persistence, the data start/end
    only rely on the issue time and forecast parameters to ensure
    that one can reason about what data was used for a particular
    forecast instead of also having to know when the forecast was
    made.


    Parameters
    ----------
    observation : datamodel.Observation
    forecast : datamodel.Forecast
    run_time : pd.Timestamp
    issue_time : pd.Timestamp

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp
    """

    if _is_intraday(forecast):
        data_start, data_end = _intraday_start_end(observation, forecast,
                                                   run_time)
    elif forecast.variable == 'net_load':
        data_start, data_end = _weekahead_start_end(
            issue_time, forecast)
    else:
        data_start, data_end = _dayahead_start_end(issue_time, forecast)

    _check_instant_compatibility(observation, forecast)
    # to ensure that each observation data point contributes to the correct
    # forecast, the data_end and data_start values may need to be nudged
    if 'instant' in observation.interval_label:
        data_start, data_end = _adjust_for_instant_obs(data_start, data_end,
                                                       observation, forecast)
    return data_start, data_end


def _check_instant_compatibility(observation, forecast):
    if 'instant' in forecast.interval_label:
        if 'instant' not in observation.interval_label:
            raise ValueError('Instantaneous forecast cannot be made from '
                             'interval average observations')

        if forecast.interval_length != observation.interval_length:
            raise ValueError('Instantaneous forecast requires instantaneous '
                             'observation with identical interval length.')


def _check_intraday_compatibility(observation, forecast):
    # time window over which observation data will be used to create
    # persistence forecast.
    if (observation.interval_length > forecast.run_length or
            observation.interval_length > pd.Timedelta('1h')):
        raise ValueError(
            'Intraday persistence requires observation.interval_length '
            '<= forecast.run_length and observation.interval_length <= 1h')


def check_persistence_compatibility(observation, forecast, index):
    """
    Checks if the Observation is compatible with the Forecast to
    generate a persistence forecast.

    Parameters
    ----------
    observation : datamodel.Observation
        The metadata of the observation to be used to create the
        forecast.
    forecast : datamodel.Forecast
        The metadata of the desired forecast.
    index : bool
        If the persistence forecast will persist a clear sky or
        AC power index (True), or use the observed value (False).

    Raises
    ------
    ValueError
        If an intraday forecast is to be made and the
        observation interval length is too long.
    ValueError
        If an instantaneous forecast is to be made and the
        observation is not also instantaneous with the same
        interval length.
    ValueError
        If the forecast run length is greater than one day
        and an index persistence forecast was to be made.
    """
    intraday = _is_intraday(forecast)
    if intraday:
        _check_intraday_compatibility(observation, forecast)
    else:
        if index:
            raise ValueError('index=True not supported for forecasts'
                             ' with run_length >= 1day')
    _check_instant_compatibility(observation, forecast)
