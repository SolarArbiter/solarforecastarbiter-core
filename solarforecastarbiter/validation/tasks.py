import logging


import numpy as np
import pandas as pd
from pvlib.irradiance import get_extra_radiation


from solarforecastarbiter import pvmodel, datamodel
from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.validation import validator, quality_mapping


logger = logging.getLogger(__name__)


def _validate_timestamp(observation, values):
    return validator.check_timestamp_spacing(
        values.index, observation.interval_length, _return_mask=True)


def _validate_stale_interpolated(observation, values):
    window = validator.stale_interpolated_window(observation.interval_length)
    stale_flag = validator.detect_stale_values(values, window=window,
                                               _return_mask=True)
    interpolation_flag = validator.detect_interpolation(values, window=window,
                                                        _return_mask=True)
    return stale_flag, interpolation_flag


# three functions to handle solar position and nighttime flags.
# 1. _solpos_night dispatches to one of
# 2. _solpos_night_instantaneous for instantaneous observations or
# 3. _solpos_night_resample for interval average observations

def _solpos_night(observation, values):
    closed = datamodel.CLOSED_MAPPING[observation.interval_label]
    if closed is None:
        return _solpos_night_instantaneous(observation, values)
    else:
        return _solpos_night_resample(observation, values)


def _solpos_night_instantaneous(observation, values):
    solar_position = pvmodel.calculate_solar_position(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, values.index)
    night_flag = validator.check_day_night(solar_position['zenith'],
                                           _return_mask=True)
    return solar_position, night_flag


def _resample_date_range(interval_length, closed, freq, values):
    # consider moving this to utils
    data_start, data_end = values.index[0], values.index[-1]
    if closed == 'left':
        data_end += interval_length
    elif closed == 'right':
        data_start -= interval_length
    else:
        raise ValueError("closed must be left or right")  # pragma: no cover
    obs_range = pd.date_range(start=data_start, end=data_end, freq=freq,
                              closed=closed)
    return obs_range


def _solpos_night_resample(observation, values):
    # similar approach as in persistence_scalar_index.
    # Calculate solar position and clearsky at 1 minute resolution to
    # reduce errors from changing solar position during interval.
    # Later, nighttime bools will be resampled over the interval
    closed = datamodel.CLOSED_MAPPING[observation.interval_label]
    freq = pd.Timedelta('1min')
    interval_length = observation.interval_length
    # need to calculate solar position for instants before or after the
    # first/last labels depending on the interval label convention.
    obs_range = _resample_date_range(interval_length, closed, freq, values)
    # could add logic to remove points from obs_range where there are
    # gaps in values. that would reduce computation time in some situations
    solar_position = pvmodel.calculate_solar_position(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, obs_range
    )
    # get the night flag as bitmask
    night_flag = validator.check_day_night_interval(
        solar_position['zenith'],
        closed,
        interval_length,
        solar_zenith_interval_length=freq,
        _return_mask=True
    )
    # Better to use average solar position in downstream functions
    # Best to return high res solar position and adapt downstream functions
    # but that is left for future work
    solar_position = solar_position.resample(
        interval_length, closed=closed, label=closed
    ).mean()
    # return series with same index as input values
    # i.e. put any gaps back in the data
    try:
        night_flag = night_flag.loc[values.index]
        solar_position = solar_position.loc[values.index]
    except KeyError:
        raise KeyError(
            'Missing times when reindexing averaged flag or solar position to '
            'original data. Check that observation.interval_length is '
            'consistent with observation_values.index.')
    return solar_position, night_flag


def _solpos_dni_extra(observation, values):
    solar_position, night_flag = _solpos_night(observation, values)
    dni_extra = get_extra_radiation(values.index)
    timestamp_flag = _validate_timestamp(observation, values)
    return solar_position, dni_extra, timestamp_flag, night_flag


def validate_ghi(observation, values):
    """
    Run validation checks on a GHI observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    ghi_limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_ghi_limits_QCRad`
    ghi_clearsky_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_ghi_clearsky`
    cloud_free_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_clearsky_ghi`
    """
    solar_position, dni_extra, timestamp_flag, night_flag = _solpos_dni_extra(
        observation, values)
    clearsky = pvmodel.calculate_clearsky(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, solar_position['apparent_zenith'])

    ghi_limit_flag = validator.check_ghi_limits_QCRad(
        values, solar_position['zenith'], dni_extra,
        _return_mask=True)
    ghi_clearsky_flag = validator.check_ghi_clearsky(values, clearsky['ghi'],
                                                     _return_mask=True)
    cloud_free_flag = validator.detect_clearsky_ghi(values, clearsky['ghi'],
                                                    _return_mask=True)
    return (timestamp_flag, night_flag, ghi_limit_flag,
            ghi_clearsky_flag, cloud_free_flag)


def validate_dni(observation, values):
    """
    Run validation checks on a DNI observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    dni_limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_dni_limits_QCRad`
    """
    solar_position, dni_extra, timestamp_flag, night_flag = _solpos_dni_extra(
        observation, values)
    dni_limit_flag = validator.check_dni_limits_QCRad(values,
                                                      solar_position['zenith'],
                                                      dni_extra,
                                                      _return_mask=True)
    return timestamp_flag, night_flag, dni_limit_flag


def validate_dhi(observation, values):
    """
    Run validation checks on a DHI observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    dhi_limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_dhi_limits_QCRad`
    """
    solar_position, dni_extra, timestamp_flag, night_flag = _solpos_dni_extra(
        observation, values)
    dhi_limit_flag = validator.check_dhi_limits_QCRad(values,
                                                      solar_position['zenith'],
                                                      dni_extra,
                                                      _return_mask=True)
    return timestamp_flag, night_flag, dhi_limit_flag


def validate_poa_global(observation, values):
    """
    Run validation checks on a POA observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    poa_clearsky_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_poa_clearsky`
    """
    solar_position, dni_extra, timestamp_flag, night_flag = _solpos_dni_extra(
        observation, values)
    clearsky = pvmodel.calculate_clearsky(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, solar_position['apparent_zenith'])
    aoi_func = pvmodel.aoi_func_factory(observation.site.modeling_parameters)
    poa_clearsky = pvmodel.calculate_poa_effective(
        aoi_func=aoi_func, apparent_zenith=solar_position['apparent_zenith'],
        azimuth=solar_position['azimuth'], ghi=clearsky['ghi'],
        dni=clearsky['dni'], dhi=clearsky['dhi'])
    poa_clearsky_flag = validator.check_poa_clearsky(values, poa_clearsky,
                                                     _return_mask=True)
    return timestamp_flag, night_flag, poa_clearsky_flag


def validate_air_temperature(observation, values):
    """
    Run validation checks on an air temperature observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_temperature_limits`
    """
    timestamp_flag, night_flag = validate_defaults(observation, values)
    limit_flag = validator.check_temperature_limits(
        values, _return_mask=True)
    return timestamp_flag, night_flag, limit_flag


def validate_wind_speed(observation, values):
    """
    Run validation checks on a wind speed observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.wind_limit_flag`
    """
    timestamp_flag, night_flag = validate_defaults(observation, values)
    limit_flag = validator.check_wind_limits(values, _return_mask=True)
    return timestamp_flag, night_flag, limit_flag


def validate_relative_humidity(observation, values):
    """
    Run validation checks on a relative humidity observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_rh_limits`
    """
    timestamp_flag, night_flag = validate_defaults(observation, values)
    limit_flag = validator.check_rh_limits(values, _return_mask=True)
    return timestamp_flag, night_flag, limit_flag


def validate_ac_power(observation, values):
    """
    Run a number of validation checks on a daily timeseries of AC power.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_ac_power_limits`
    """
    solar_position, dni_extra, timestamp_flag, night_flag = _solpos_dni_extra(
        observation, values)
    day_night = \
        ~quality_mapping.convert_mask_into_dataframe(night_flag)['NIGHTTIME']
    limit_flag = validator.check_ac_power_limits(
        values, day_night,
        observation.site.modeling_parameters.ac_capacity, _return_mask=True)
    return timestamp_flag, night_flag, limit_flag


def validate_dc_power(observation, values):
    """
    Run a number of validation checks on a daily timeseries of DC power.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_dc_power_limits`
    """
    solar_position, dni_extra, timestamp_flag, night_flag = _solpos_dni_extra(
        observation, values)
    day_night = \
        ~quality_mapping.convert_mask_into_dataframe(night_flag)['NIGHTTIME']
    dc_limit_flag = validator.check_dc_power_limits(
        values, day_night,
        observation.site.modeling_parameters.dc_capacity, _return_mask=True)
    return timestamp_flag, night_flag, dc_limit_flag


def validate_defaults(observation, values):
    """
    Run default validation checks on an observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    """
    timestamp_flag = _validate_timestamp(observation, values)
    _, night_flag = _solpos_night(observation, values)
    return timestamp_flag, night_flag


def validate_daily_ghi(observation, values):
    """
    Run validation on a daily timeseries of GHI. First,
    all checks of `validate_ghi` are run in addition to
    detecting stale values and interpolation

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    *ghi_flags
        Bitmasks from :py:func:`.tasks.validate_ghi`
    stale_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_stale_values`
    interpolation_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_interpolation`
    """
    ghi_flags = validate_ghi(observation, values)
    stale_flag, interpolation_flag = _validate_stale_interpolated(observation,
                                                                  values)
    return (*ghi_flags, stale_flag, interpolation_flag)


def validate_daily_dc_power(observation, values):
    """
    Run validation on a daily timeseries of DC power.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_dc_power_limits`
    stale_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_stale_values`
    interpolation_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_interpolation`
    """
    timestamp_flag, night_flag, dc_limit_flag = validate_dc_power(observation,
                                                                  values)
    stale_flag, interpolation_flag = _validate_stale_interpolated(observation,
                                                                  values)
    return (timestamp_flag, night_flag, dc_limit_flag, stale_flag,
            interpolation_flag)


def validate_daily_ac_power(observation, values):
    """
    Run a number of validation checks on a daily timeseries of AC power.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    timestamp_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_timestamp_spacing`
    night_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_day_night` or
        :py:func:`.validator.check_day_night_interval`
    limit_flag : pandas.Series
        Bitmask from :py:func:`.validator.check_ac_power_limits`
    stale_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_stale_values`
    interpolation_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_interpolation`
    """
    timestamp_flag, night_flag, ac_limit_flag = validate_ac_power(observation,
                                                                  values)
    stale_flag, interpolation_flag = _validate_stale_interpolated(observation,
                                                                  values)
    clipping_flag = validator.detect_clipping(values, _return_mask=True)
    return (timestamp_flag, night_flag, ac_limit_flag, stale_flag,
            interpolation_flag, clipping_flag)


def validate_daily_defaults(observation, values):
    """
    Run default daily validation checks on an observation.
    Applies the validation for the observation's variable and then
    the stale and interpolated validation. :py:func:`validate_defaults`
    is used if the Observation variable does not have a defined validation
    function.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    *variable_immediate_flags
        Bitmasks from :py:func:`.tasks.validate_{variable}`
    stale_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_stale_values`
    interpolation_flag : pandas.Series
        Bitmask from :py:func:`.validator.detect_interpolation`
    """
    immediate_func = IMMEDIATE_VALIDATION_FUNCS.get(
        observation.variable, validate_defaults)
    immediate_flags = immediate_func(observation, values)
    stale_flag, interpolation_flag = _validate_stale_interpolated(observation,
                                                                  values)
    return (*immediate_flags, stale_flag, interpolation_flag)


IMMEDIATE_VALIDATION_FUNCS = {
    'air_temperature': validate_air_temperature,
    'wind_speed': validate_wind_speed,
    'ghi': validate_ghi,
    'dni': validate_dni,
    'dhi': validate_dhi,
    'poa_global': validate_poa_global,
    'relative_humidity': validate_relative_humidity,
    'ac_power': validate_ac_power,
    'dc_power': validate_dc_power
}
DAILY_VALIDATION_FUNCS = {
    'ghi': validate_daily_ghi,
    'dc_power': validate_daily_dc_power,
    'ac_power': validate_daily_ac_power,
    # no stale/interpolated
    'event': validate_defaults,
    'availability': validate_defaults,
    'curtailment': validate_defaults,
}


def apply_immediate_validation(observation, observation_values):
    """
    Apply the appropriate validation functions to the observation_values.

    Only the USER_FLAGGED flag is propagated if the series has been
    previously validated.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
    observation_values : pandas.DataFrame
        Must have 'value' and 'quality_flag' columns

    Returns
    -------
    pandas.DataFrame
        With the same index as the input and 'quality_flag' updated
        appropriately
    """
    value_series = observation_values['value'].astype(float)
    quality_flags = observation_values['quality_flag'].copy() & 1

    validation_func = IMMEDIATE_VALIDATION_FUNCS.get(
        observation.variable, validate_defaults)
    validation_flags = validation_func(observation, value_series)

    for flag in validation_flags:
        quality_flags |= flag
    quality_flags |= quality_mapping.LATEST_VERSION_FLAG

    quality_flags.name = 'quality_flag'
    observation_values.update(quality_flags)
    return observation_values


def apply_daily_validation(observation, observation_values):
    """
    Apply the appropriate daily validation functions to the observation_values.

    Only the USER_FLAGGED flag is propagated if the series has been previously
    validated.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
    observation_values : pandas.DataFrame
        Must have 'value' and 'quality_flag' columns

    Returns
    -------
    pandas.DataFrame
        With the same index as the input and 'quality_flag' updated
        appropriately

    Raises
    ------
    IndexError
        If there are not enough valid points to perform daily validation

    """
    validated = observation_values.sort_index()
    value_series = validated['value'].astype(float)
    if len(value_series.dropna()) < 10:
        raise IndexError(
            'Data series does not have at least 10 datapoints to validate')
    quality_flags = validated['quality_flag'].copy() & 1

    # if the variable has a daily check, run that, else run the
    # immediate validation, else validate timestamps
    validation_func = DAILY_VALIDATION_FUNCS.get(
        observation.variable, validate_daily_defaults)
    validation_flags = validation_func(observation, value_series)

    for flag in validation_flags:
        quality_flags |= flag
    quality_flags |= quality_mapping.DAILY_VALIDATION_FLAG
    quality_flags |= quality_mapping.LATEST_VERSION_FLAG

    quality_flags.name = 'quality_flag'
    validated.update(quality_flags)
    return validated


def apply_validation(observation, observation_values):
    """
    Applies the appropriate daily or immediate validation functions to the
    observation_values depending on the length of the data. If an Aggregate
    object is passed, a warning is logged and the observation_values are
    returned.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
    observation_values : pandas.DataFrame
        Must have 'value' and 'quality_flag' columns

    Returns
    -------
    pandas.DataFrame
        With the same index as the input and 'quality_flag' updated
        appropriately

    Raises
    ------
    TypeError
        If the supplied observations_values is not a DataFrame with a
        DatetimeIndex
    """
    if isinstance(observation, datamodel.Aggregate):
        logger.warning('Cannot apply validation to an Aggregate')
        return observation_values
    data = observation_values.sort_index()
    if (
            not isinstance(data, pd.DataFrame) or
            not isinstance(data.index, pd.DatetimeIndex)
    ):
        raise TypeError('Expect observation_values to have a DatetimeIndex')
    if data.empty:
        return data
    if (
            (data.index[-1] - data.index[0]) >= pd.Timedelta('1d') and
            (len(data['value'].dropna()) > 10)
    ):
        return apply_daily_validation(observation, data)
    else:
        return apply_immediate_validation(observation, data)


def _group_continuous_week_post(session, observation, observation_values):
    # observation_values expected to be sorted
    # observation values already have uneven frequency checked
    gid = quality_mapping.check_if_series_flagged(
        observation_values['quality_flag'], 'UNEVEN FREQUENCY').cumsum()
    # make series of week + year integers to further
    # split data to post at most one week at a time
    # ~10,000 pts of 1min data
    week_int = (gid.index.week + gid.index.year).values
    # combine the continuous groups with groups of weeks
    # gid is unique for each group since week_int and cumsum
    # increase monotonically and are positive
    gid += week_int
    observation_values['gid'] = gid
    for _, group in observation_values.groupby('gid'):
        session.post_observation_values(observation.observation_id,
                                        group[['value', 'quality_flag']],
                                        params='donotvalidate')


def _validate_post(session, observation, start, end):
    logger.info('Validating data for %s from %s to %s',
                observation.name, start, end)
    observation_values = session.get_observation_values(
        observation.observation_id, start, end)
    validated = apply_validation(observation, observation_values)
    return _group_continuous_week_post(
        session, observation, validated)


def _find_unvalidated_time_ranges(session, observation, min_start, max_end):
    """Find the time ranges where the observation data needs to have
    daily validation applied. Extend to next day midnight so daily
    validation can be applied even since it requires >= 1 day of data
    """
    tz = observation.site.timezone
    dates = session.get_observation_values_not_flagged(
        observation_id=observation.observation_id,
        start=min_start,
        end=max_end,
        flag=(
            quality_mapping.DAILY_VALIDATION_FLAG |
            quality_mapping.LATEST_VERSION_FLAG
        ),
        timezone=tz)
    if len(dates) == 0:
        return
    sorted_dates = np.array(sorted(dates))

    def first_last(prev, ind):
        first = pd.Timestamp(dates[prev]).tz_localize(tz)
        last = (pd.Timestamp(dates[ind])
                .tz_localize(tz) + pd.Timedelta('1D'))
        return first, last

    prev = 0
    # find the difference between each date, as integer days
    # subtract one to then use nonzero to find those
    # dates that are not continuous
    breaks = np.diff(sorted_dates).astype('timedelta64[D]') - 1
    discontinuities = np.nonzero(breaks)[0]

    for ind in discontinuities:
        first, last = first_last(prev, ind)
        yield first, last
        prev = ind + 1

    first, last = first_last(prev, -1)
    yield first, last


def _split_validation(session, observation, start, end, only_missing):
    if not only_missing:
        return _validate_post(session, observation, start, end)

    for _start, _end in _find_unvalidated_time_ranges(
            session, observation, start, end):
        _validate_post(session, observation, _start, _end)


def fetch_and_validate_observation(access_token, observation_id, start, end,
                                   only_missing=False, base_url=None):
    """Task that will run immediately after Observation values are
    uploaded to the API to validate the data. If over a day of data is
    present, daily validation will be applied.

    For the last day of a multiday series that only has a partial day's
    worth of data, if `only_missing` is False, the data is evaluated as
    one series and daily validation is applied. If `only_missing` is True,
    any discontinuous periods of data with less than one day of data will
    only have immediate validation applied. If the period is longer than
    a day, the full daily validation is applied.

    Parameters
    ----------
    access_token : str
        Token to access the API
    observation_id : str
        ID of the observation to fetch values and validate
    start : datetime-like
        Start time to limit observation fetch
    end : datetime-like
        End time to limit observation fetch
    only_missing : boolean, default False
        If True, only periods that have not had daily validation applied
        are fetched and validated. Otherwise all data between start and end
        is validated.
    base_url : str, default None
        URL for the API to fetch and post data
    """
    session = APISession(access_token, base_url=base_url)
    observation = session.get_observation(observation_id)
    _split_validation(session, observation, start, end, only_missing)


def fetch_and_validate_all_observations(access_token, start, end,
                                        only_missing=True, base_url=None):
    """
    Run the observation validation for all observations that the user
    has access to in their organization. See further discussion in
    :py:func:`solarforecastarbiter.validation.tasks.fetch_and_validate_all_observations`

    Parameters
    ----------
    access_token : str
        Token to access the API
    start : datetime-like
        Start time to limit observation fetch
    end : datetime-like
        End time to limit observation fetch
    only_missing : boolean, default True
        If True, only periods that have not had daily validation applied
        are fetched and validated. Otherwise all data between start and end
        is validated.
    base_url : str, default None
        URL for the API to fetch and post data

    """
    session = APISession(access_token, base_url=base_url)
    user_info = session.get_user_info()
    observations = [obs for obs in session.list_observations()
                    if obs.provider == user_info['organization']]
    for observation in observations:
        _split_validation(session, observation, start, end, only_missing)
