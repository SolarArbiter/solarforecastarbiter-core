import pandas as pd


from solarforecastarbiter.datamodel import ALLOWED_VARIABLES, COMMON_NAMES


def format_variable_name(variable):
    """Make a human readable name for the variable"""
    return f'{COMMON_NAMES[variable]} ({ALLOWED_VARIABLES[variable]})'


def align_index(df, interval_length, limit=None):
    """
    Align the index to the specified interval_length inserting NaNs
    as appropriate.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        Pandas object with a datetime index that will be reindexed
    interval_length : pd.Timedelta
        Interval to conform input index to
    limit : pd.Timedelta, default None
        Restricts the output index to `end - limit:end`

    Returns
    -------
    pandas.DataFrame or pandas.Series
       Same type as input `df` with an index with frequency `interval_length`
    """
    # If there is more than 3 days of data, limit the default x_range
    # to display only the most recent 3 day. Enable scrolling back in future
    # release.
    period_end = df.index[-1]
    if limit is not None:
        period_start = df.index[
            df.index.get_loc(period_end - limit, method='bfill')]
    else:
        period_start = df.index[0]
    # align the data on the index it should have according to the metadata
    nindex = pd.date_range(start=period_start, end=period_end,
                           freq=interval_length,
                           name='timestamp')
    df = df.reindex(nindex, axis=0)
    return df


def line_or_step(interval_label):
    """
    For a given interval_label, determine the plot_method of the data,
    any kwargs for that plot method, and kwargs for adding a hovertool
    for the data.
    """
    if 'instant' in interval_label:
        plot_method = 'line'
        plot_kwargs = dict()
        hover_kwargs = dict(line_policy='nearest',
                            attachment='horizontal')
    elif interval_label == 'beginning':
        plot_method = 'step'
        plot_kwargs = dict(mode='after')
        hover_kwargs = dict(line_policy='prev',
                            attachment='left',
                            add_line=True)
    elif interval_label == 'ending':
        plot_method = 'step'
        plot_kwargs = dict(mode='before')
        hover_kwargs = dict(line_policy='next',
                            attachment='right',
                            add_line=True)
    else:
        raise ValueError(
            'interval_label must be one of "instant", "beginning", '
            'or "ending"')

    return plot_method, plot_kwargs, hover_kwargs
