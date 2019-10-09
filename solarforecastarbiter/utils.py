import warnings


import numpy as np
import pandas as pd


from solarforecastarbiter import datamodel


def _observation_valid(index, obs_id, aggregate_observations):
    """
    Indicates where the observation data is valid. For now,
    effective_from and effective_until are inclusive, so data missing
    at those times is marked as missing in the aggregate.

    Other option is to use aggregate interval_label to determine
    which side should be open/closed.
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
                      timezone, agg_func, aggregate_observations):
    """
    Perform the aggregation. Nans remain.
    First, resample possibly ignoring nans.
    Then aggregate and keep nans.
    Assumptions: data is interval_mean or instantaneous
    interval_length <= agg interval_length


    Parameters
    ----------
    data : dict of pandas.DataFrames
       keys are observation_id and df With values, quality_flag columns.
    """
    new_index = _make_aggregate_index(
        data, interval_length, interval_label, timezone)
    unique_ids = {ao['observation_id'] for ao in aggregate_observations}
    valid_mask = {obs_id: _observation_valid(
        new_index, obs_id, aggregate_observations) for obs_id in unique_ids}

    missing_from_data = {
        ao['observation_id'] for ao in aggregate_observations
        if ao['observation_deleted_at'] is None
        } - set(data.keys())

    if missing_from_data:
        raise KeyError(
            'Cannot aggregate data with missing keys '
            f'{", ".join(missing_from_data)}')

    data_missing = pd.Series(False, index=new_index)
    value = {}
    qf = {}
    closed = datamodel.CLOSED_MAPPING[interval_label]
    for obs_id, df in data.items():
        resampled = df.resample(interval_length, closed=closed, label=closed)
        new_val = resampled.value.mean().reindex(new_index)
        # data is missing when the resampled value is NaN and the data
        # should be valid according to effective_from/until
        valid = valid_mask[obs_id]
        missing = new_val.isna() & valid
        if missing.any():
            warnings.warn('Values missing for one or more observations')
            data_missing[missing] = True
        value[obs_id] = new_val[valid]
        qf[obs_id] = resampled.quality_flag.apply(np.bitwise_or.reduce)
    final_value = pd.DataFrame(value).reindex(new_index).aggregate(
        agg_func, axis=1)
    final_value[data_missing] = np.nan
    # have to fill in nans and convert to int to do bitwise_or
    final_qf = pd.DataFrame(qf).reindex(new_index).fillna(0).astype(
        int).aggregate(np.bitwise_or.reduce, axis=1)
    out = pd.DataFrame({'value': final_value, 'quality_flag': final_qf})
    return out
