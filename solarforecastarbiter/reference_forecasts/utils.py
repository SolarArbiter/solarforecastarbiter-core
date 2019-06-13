import pandas as pd


from solarforecastarbiter.io import utils as io_utils


def issue_times(forecast, start_from=None):
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
    out = []
    while issue < next_day:
        if start_from is None:
            out.append(issue.time())
        else:
            out.append(issue)
        issue += forecast.run_length
    return out


def get_init_time(run_time, fetch_metadata):
    """Determine the most recent init time for which all forecast data is
    available."""
    run_finish = (pd.Timedelta(fetch_metadata['delay_to_first_forecast']) +
                  pd.Timedelta(fetch_metadata['avg_max_run_length']))
    freq = fetch_metadata['update_freq']
    init_time = (run_time - run_finish).floor(freq=freq)
    return init_time


def get_forecast_start_end(forecast, issue_time):
    """
    Get absolute forecast start from *forecast* object parameters and
    absolute *issue_time*.

    Parameters
    ----------
    forecast : datamodel.Forecast
    issue_time : pd.Timestamp

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
    first_issue_time = pd.Timestamp.combine(issue_time.floor('1D'),
                                            forecast.issue_time_of_day)
    issue_times = pd.date_range(start=first_issue_time,
                                end=first_issue_time+pd.Timedelta('1d'),
                                freq=forecast.run_length)
    if issue_time not in issue_times:
        raise ValueError(
            ('Incompatible forecast.issue_time_of_day %s, '
             'forecast.run_length %s, and issue_time %s') % (
             forecast.issue_time_of_day, forecast.run_length, issue_time))
    forecast_start = issue_time + forecast.lead_time_to_start
    forecast_end = forecast_start + forecast.run_length
    return io_utils.adjust_start_end_for_interval_label(
        forecast.interval_label, forecast_start, forecast_end, True)


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
    # day ahead persistence: tomorrow's forecast is equal to yesterday's
    # observations. So, forecast always uses obs > 24 hr old at each valid
    # time. Smarter approach might be to use today's observations up
    # until issue_time, and use yesterday's observations for issue_time
    # until end of day. So, forecast *never* uses obs > 24 hr old at each
    # valid time. Arguably too much for a reference forecast.
    data_end = run_time.floor('1d')
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

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp
    """
    if _is_intraday(forecast):
        data_start, data_end = _intraday_start_end(observation, forecast,
                                                   run_time)
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
