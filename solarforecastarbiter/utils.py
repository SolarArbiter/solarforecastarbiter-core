from contextlib import contextmanager
from hashlib import sha256
import logging
import warnings


import numpy as np
import pandas as pd
from sentry_sdk.integrations import logging as sentry_logging


from solarforecastarbiter import datamodel


def _observation_valid(index, obs_id, aggregate_observations):
    """
    Indicates where the observation data is valid. For now,
    effective_from and effective_until are inclusive, so data missing
    at those times is marked as missing in the aggregate.
    """
    nindex = pd.DatetimeIndex([], tz=index.tz)
    for aggobs in aggregate_observations:
        if aggobs['observation_id'] == obs_id:
            if aggobs['observation_deleted_at'] is None:
                locs = index.slice_locs(aggobs['effective_from'],
                                        aggobs['effective_until'])
                nindex = nindex.union(index[locs[0]:locs[1]])
            elif (
                    aggobs['effective_until'] is None or
                    aggobs['effective_until'] >= index[0]
            ):
                raise ValueError(
                    'Deleted Observation data cannot be retrieved'
                    ' to include in Aggregate')
            else:  # observation deleted and effective_until before index
                return pd.Series(False, index=index)
    return pd.Series(1, index=nindex).reindex(index).fillna(0).astype(bool)


def _make_aggregate_index(data, interval_length, interval_label,
                          timezone):
    """
    Compute the aggregate the index should have based on the min and
    max timestamps in the data, the interval length, label, and timezone.
    """
    # first, find limits for a new index
    start = pd.Timestamp('20380119T031407Z')
    end = pd.Timestamp('19700101T000001Z')
    for df in data.values():
        start = min(start, min(df.index))
        end = max(end, max(df.index))
    # adjust start, end to nearest interval
    # hard to understand what this interval should be for
    # odd (e.g. 52min) intervals, so required that interval
    # is a divisor of one day
    if 86400 % pd.Timedelta(interval_length).total_seconds() != 0:
        raise ValueError(
            'interval_length must be a divisor of one day')
    if interval_label == 'ending':
        start = start.ceil(interval_length)
        end = end.ceil(interval_length)
    elif interval_label == 'beginning':
        start = start.floor(interval_length)
        end = end.floor(interval_length)
    else:
        raise ValueError(
            'interval_label must be beginning or ending for aggregates')
    # raise the error if unlocalized
    start = start.tz_convert(timezone)
    end = end.tz_convert(timezone)
    return pd.date_range(
        start, end, freq=interval_length, tz=timezone)


def compute_aggregate(data, interval_length, interval_label,
                      timezone, agg_func, aggregate_observations,
                      new_index=None):
    """
    Computes an aggregate quantity according to agg_func of the data.
    This function assumes the data has an interval_value_type of
    interval_mean or instantaneous and that the data interval_length
    is less than or equal to the aggregate interval_length.
    NaNs in the output are the result of missing data from an
    underyling observation of the aggregate.

    Parameters
    ----------
    data : dict of pandas.DataFrames
        With keys 'observation_id' corresponding to observation in
        aggregate_observations. DataFrames must have 'value' and 'quality_flag'
        columns.
    interval_length : str or pandas.Timedelta
        The time between timesteps in the aggregate result.
    interval_label : str
        Whether the timestamps in the aggregated output represent the beginning
        or ending of the interval
    timezone : str
        The IANA timezone for the output index
    agg_func : str
        The aggregation function (e.g 'sum', 'mean', 'min') to create the
        aggregate
    aggregate_observations : tuple of dicts
        Each dict should have 'observation_id' (string),
        'effective_from' (timestamp), 'effective_until' (timestamp or None),
        and 'observation_deleted_at' (timestamp or None) fields.
    new_index : pandas.DatetimeIndex
        The index to resample data to. Will attempt to infer an index if not
        provided.

    Returns
    -------
    pandas.DataFrame
        - Index is a DatetimeIndex that adheres to interval_length and
          interval_label

        - Columns are 'value', for the aggregated value according to agg_func,
          and 'quality_flag', the bitwise or of all flags in the aggregate for
          the interval.

        - A 'value' of NaN means that data from one or more
          observations was missing in that interval.

    Raises
    ------
    KeyError
        If data is missing a key for an observation in aggregate_obsevations

        + Or, if any DataFrames in data do not have 'value' or 'quality_flag'
          columns

    ValueError
        If interval_length is not a divisor of one day and an index is not
        provided.

        + Or, if an observation has been deleted but the data is required for
          the aggregate
        + Or, if interval_label is not beginning or ending
        + Or, if data is empty and an index is provided.

    """
    if new_index is None:
        new_index = _make_aggregate_index(
            data, interval_length, interval_label, timezone)
    unique_ids = {ao['observation_id'] for ao in aggregate_observations}
    valid_mask = {obs_id: _observation_valid(
        new_index, obs_id, aggregate_observations) for obs_id in unique_ids}
    expected_observations = {k for k, v in valid_mask.items() if v.any()}

    # Raise an exception if no observations are valid
    if len(expected_observations) == 0:
        raise ValueError(
            'No effective observations in data')

    missing_from_data_dict = expected_observations - set(data.keys())

    if missing_from_data_dict:
        raise KeyError(
            'Cannot aggregate data with missing keys '
            f'{", ".join(missing_from_data_dict)}')

    value_is_missing = pd.Series(False, index=new_index)
    value = {}
    qf = {}
    closed = datamodel.CLOSED_MAPPING[interval_label]
    for obs_id, df in data.items():
        resampler = df.resample(interval_length, closed=closed, label=closed)
        new_val = resampler['value'].mean().reindex(new_index)
        # data is missing when the resampled value is NaN and the data
        # should be valid according to effective_from/until
        valid = valid_mask[obs_id]
        missing = new_val.isna() & valid
        if missing.any():
            warnings.warn('Values missing for one or more observations')
            value_is_missing[missing] = True
        value[obs_id] = new_val[valid]
        qf[obs_id] = resampler['quality_flag'].apply(np.bitwise_or.reduce)
    final_value = pd.DataFrame(value).reindex(new_index).aggregate(
        agg_func, axis=1)
    final_value[value_is_missing] = np.nan
    # have to fill in nans and convert to int to do bitwise_or
    # only works with pandas >= 0.25.0
    final_qf = pd.DataFrame(qf).reindex(new_index).fillna(0).astype(
        int).aggregate(np.bitwise_or.reduce, axis=1)
    out = pd.DataFrame({'value': final_value, 'quality_flag': final_qf})
    return out


def sha256_pandas_object_hash(obj):
    """
    Compute a hash for a pandas object. No sorting of the
    object is performed, so an object with the same data in
    in a different order returns a different hash.

    Parameters
    ----------
    obj: pandas.Series or pandas.DataFrame

    Returns
    -------
    str
       Hex digest of the SHA-256 hash of the individual object row hashes
    """
    return sha256(
        pd.util.hash_pandas_object(obj).values.tobytes()
    ).hexdigest()


class ListHandler(logging.Handler):
    """
    A logger handler that appends each log record to a list.
    """
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def export_records(self, level=logging.WARNING):
        """
        Convert each log record in the records list with level
        greater than or equal to `level` to a
        :py:class:`solarforecastarbiter.datamodel.ReportMessage`
        and return the tuple of messages.
        """
        out = []
        for rec in self.records:
            if rec.levelno >= level:
                out.append(
                    datamodel.ReportMessage(
                        message=rec.getMessage(),
                        step=rec.name,
                        level=rec.levelname,
                        function=rec.funcName
                    )
                )
        return tuple(out)


def _get_children(name):
    return {k for k in logging.getLogger(name).manager.loggerDict.keys()
            if k.startswith(name + '.')}


@contextmanager
def hijack_loggers(loggers, level=logging.INFO):
    """
    Context manager to temporarily set the handler
    of each logger in `loggers`.

    Parameters
    ----------
    loggers: list of str
        Loggers to change
    level: logging LEVEL int
        Level to set the temporary handler to

    Returns
    -------
    ListHandler
        The handler that will be temporarily assigned
        to each logger.

    Notes
    -----
    This may not capture all records when used in a
    distributed or multiprocessing workflow
    """
    handler = ListHandler()
    handler.setLevel(level)

    all_loggers = set()
    for name in loggers:
        all_loggers.add(name)
        all_loggers |= _get_children(name)

    logger_info = {}
    for name in all_loggers:
        logger = logging.getLogger(name)
        logger_info[name] = (logger.handlers, logger.propagate)
        logger.handlers = [handler]
        sentry_logging.ignore_logger(name)
        logger.propagate = False
    yield handler
    for name in all_loggers:
        logger = logging.getLogger(name)
        hnd, prop = logger_info[name]
        logger.handlers = hnd
        logger.propagate = prop
        try:
            sentry_logging._IGNORED_LOGGERS.remove(name)
        except Exception:
            pass
    del handler


def _unique_key(try_key, keys, i=0):
    if try_key in keys:
        try_key += str(i)
        i += 1
        return _unique_key(try_key, keys, i)
    else:
        return try_key


def generate_continuous_chunks(data, freq):
    """
    Generator to split data into continuous chunks with spacing of freq.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Data to apply func to. Must have a DatetimeIndex.
    freq : pd.Timedelta
        Expected frequency to split data into continuous chunks

    Yields
    ------
    continuous_data : same as data
        Each continuous chunk that conforms to freq

    Raises
    ------
    TypeError
        If data is not a pandas Series or DataFrame, or
        does not have a DatetimeIndex
    ValueError
        If freq cannot be converted to a pandas.Timedelta

    Examples
    --------
    The following code would post two forecast series ignoring the missing
    period in the middle.

    .. testsetup::

        import pandas as pd
        from solarforecastarbiter.io import api

    >>> series = pd.Series(
    ...     [1.0, 2.0, 3.0, 7.0, 8.0],
    ...     index=[
    ...         pd.Timestamp('2020-07-01T01:00Z'),
    ...         pd.Timestamp('2020-07-01T02:00Z'),
    ...         pd.Timestamp('2020-07-01T03:00Z'),
    ...         pd.Timestamp('2020-07-01T07:00Z'),
    ...         pd.Timestamp('2020-07-01T08:00Z'),
    ...     ])
    >>> session = api.APISession('token')
    >>> for cser in generate_continuous_chunks(series, pd.Timedelta('1h')):
    ...     session.post_forecast_values('forecast_id', cser)
    """
    if (
            not isinstance(data, (pd.Series, pd.DataFrame)) or
            not isinstance(data.index, pd.DatetimeIndex)
    ):
        raise TypeError(
            'data must be a pandas Series or DataFrame with DatetimeIndex')
    if not isinstance(freq, pd.Timedelta):
        freq = pd.Timedelta(freq)

    # first value is NaT, rest are timedeltas
    index_ser = data.index.to_series()
    delta = index_ser.diff()
    if len(delta) < 2:
        yield data
        return

    flags = delta != freq

    group_key = '_cid'
    if isinstance(data, pd.DataFrame):
        keys = data.columns
        group_key = _unique_key(group_key, keys)
        data[group_key] = flags.cumsum()
    elif isinstance(data, pd.Series):
        keys = data.name
        data = pd.DataFrame(
            {keys: data, group_key: flags.cumsum()})

    for _, group in data.groupby(group_key):
        yield group[keys]


def merge_ranges(ranges):
    """Generator to merge the ranges like (min_val, max_val) removing any overlap.
    Results will be sorted in ascending order. The type of values in each range
    set should have well defined behaviour with the comparison operators, namely
    >, >=, <, <=.

    Parameters
    ----------
    ranges: iterable

    Yields
    ------
    next_value: same type as ranges[0]

    Raises
    ------
    ValueError
        If any range is not properly sorted
    TypeError
        If any range values cannot be compared

    Examples
    --------
    .. testsetup::
        import pandas as pd
        from solarforecastarbiter.utils import merge_ranges

    >>> list(merge_ranges([[0, 1], [9, 15], [-1, 3]]))
    [[-1, 3], [9, 15]]

    >>> list(merge_ranges([
    ...         (pd.Timestamp('2020-01-01T00:00Z'), pd.Timestamp('2020-01-05T12:00Z')),
    ...         (pd.Timestamp('2020-01-02T00:00Z'), pd.Timestamp('2020-01-03T12:00Z')),
    ... ]))
    [(pd.Timestamp('2020-01-01T00:00Z'), pd.Timestamp('2020-01-05T12:00Z'))]
    """  # NOQA
    if len(ranges) == 0:
        return ranges
    type_ = type(ranges[0])
    ranges = sorted(ranges)
    last = list(ranges[0])
    for rset in ranges:
        rset = list(rset)
        if rset[1] < rset[0]:
            raise ValueError(
                'All ranges must be properly sorted like (min, max)')
        if not (rset[0] < rset[1] or rset[0] > rset[1] or rset[0] == rset[1]):
            raise TypeError(
                f'Cannot properly compare ({rset[0]}, {rset[1]})')
        if rset[0] <= last[1]:
            last[1] = max(rset[1], last[1])
        else:
            yield type_(last)
            last = rset
    yield type_(last)
