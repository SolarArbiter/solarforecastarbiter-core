import numpy as np
import pandas as pd


from solarforecastarbiter.io import utils as io_utils


def get_issue_times(forecast, start_from=None):
    """
    Return a list of the issue times for a given Forecast.

    Parameters
    ----------
    forecast : datamodel.Forecast
        Forecast object that contains the time information
    start_from : pandas.Timestamp or None
        If a timestamp, return the issues times for the same days as
        `start_from`. If None, return datetime.time objects.

    Returns
    -------
    list
        Either of datetime.time objects indicating the possible issue times, or
        pandas.Timestamp objects with the issues times for the particular day
        including the first issue time for the next day.
    """
    if start_from is None:
        issue = pd.Timestamp.combine(pd.Timestamp(0).date(),
                                     forecast.issue_time_of_day)
    else:
        issue = pd.Timestamp.combine(start_from.date(),
                                     forecast.issue_time_of_day).tz_localize(
                                         start_from.tz)
    next_day = (issue + pd.Timedelta('1d')).floor('1d')
    # works even for midnight issue
    out = [issue]
    while issue < next_day:
        issue += forecast.run_length
        out.append(issue)
    if start_from is None:
        out = [o.time() for o in out]
    return out


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


def find_next_issue_time_from_last_forecast(forecast,  last_forecast_time):
    """
    Find the next issue time for *forecast* based on the timestamp of the
    last forecast value. If *last_forecast_time* is not the end of a forecast
    run, the issue time returned will be the issue time that overwrites
    *last_forecast_time* with a full length forecast.

    Parameters
    ----------
    forecast : datamodel.Forecast
    last_forecast_time : pd.Timestamp
        Last timestamp avaible for the forecast

    Returns
    pd.Timestamp
        The next issue time for the forecast
    """
    last_probable_issue_time = (last_forecast_time -
                                forecast.run_length -
                                forecast.lead_time_to_start)
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


def _check_midnight_to_midnight(forecast_start, forecast_end):
    if (forecast_start.round('1d') != forecast_start or
            forecast_end - forecast_start > pd.Timedelta('1d')):
        raise ValueError(
            'Day ahead persistence requires midnight to midnight periods')


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

    # time window over which observation data will be used to create
    # persistence forecast.
    if (observation.interval_length > forecast.run_length or
            observation.interval_length > pd.Timedelta('1h')):
        raise ValueError(
            'Intraday persistence requires observation.interval_length '
            '<= forecast.run_length and observation.interval_length <= 1h')
    # no longer than 1 hour
    window = min(forecast.run_length, pd.Timedelta('1hr'))
    data_end = run_time
    data_start = data_end - window
    return data_start, data_end


def _dayahead_start_end(run_time):
    """
    Time range of data to be used for day-ahead persistence forecast.

    Parameters
    ----------
    run_time : pd.Timestamp

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp

    Notes
    -----
    Day-ahead persistence: tomorrow's forecast is equal to yesterday's
    observations. So, forecast always uses obs > 24 hr old at each valid
    time. Smarter approach might be to use today's observations up
    until issue_time, and use yesterday's observations for issue_time
    until end of day. So, forecast *never* uses obs > 24 hr old at each
    valid time. Arguably too much for a reference forecast.
    """

    data_end = run_time.floor('1d')
    data_start = data_end - pd.Timedelta('1d')
    return data_start, data_end


def _weekahead_start_end(run_time):
    """
    Time range of data to be used for week-ahead persistence, aka, day of week
    persistence.

    Parameters
    ----------
    run_time : pd.Timestamp
    run_length : pd.Timedelta

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp

    """
    data_end = run_time.ceil('1d') - pd.Timedelta('6d')
    data_start = data_end - pd.Timedelta('1d')
    return data_start, data_end


def _adjust_for_instant_obs(data_start, data_end, observation, forecast):
    # instantaneous observations require care.
    # persistence models return forecasts with same closure as obs
    if 'instant' in forecast.interval_label:
        if forecast.interval_length != observation.interval_length:
            raise ValueError('Instantaneous forecast requires instantaneous '
                             'observation with identical interval length.')
        else:
            data_end -= pd.Timedelta('1s')
    elif forecast.interval_label == 'beginning':
        data_end -= pd.Timedelta('1s')
    else:
        data_start += pd.Timedelta('1s')
    return data_start, data_end


def get_data_start_end(observation, forecast, run_time):
    """
    Determine the data start and data end times for a persistence
    forecast.

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

    if _is_intraday(forecast):
        data_start, data_end = _intraday_start_end(observation, forecast,
                                                   run_time)
    elif forecast.variable == 'net_load':
        data_start, data_end = _weekahead_start_end(run_time)
    else:
        data_start, data_end = _dayahead_start_end(run_time)

    # to ensure that each observation data point contributes to the correct
    # forecast, the data_end and data_start values may need to be nudged
    if 'instant' in observation.interval_label:
        data_start, data_end = _adjust_for_instant_obs(data_start, data_end,
                                                       observation, forecast)
    else:
        if 'instant' in forecast.interval_label:
            raise ValueError('Instantaneous forecast cannot be made from '
                             'interval average observations')

    return data_start, data_end
