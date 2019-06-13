import pandas as pd


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
