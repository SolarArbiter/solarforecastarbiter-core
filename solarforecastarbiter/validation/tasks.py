import logging


import pandas as pd
from pvlib.irradiance import get_extra_radiation


from solarforecastarbiter import pvmodel
from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.validation import validator, quality_mapping


logger = logging.getLogger(__name__)


def _validate_timestamp(observation, values):
    return validator.check_timestamp_spacing(
        values.index, observation.interval_length, _return_mask=True)


def _solpos_night(observation, values):
    solar_position = pvmodel.calculate_solar_position(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, values.index)
    night_flag = validator.check_irradiance_day_night(solar_position['zenith'],
                                                      _return_mask=True)
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
    timestamp_flag, night_flag, ghi_limit_flag, ghi_clearsky_flag, cloud_free_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.check_ghi_limits_QCRad`,
        :py:func:`.validator.check_ghi_clearsky`,
        :py:func:`.validator.detect_clearsky_ghi` respectively
    """  # NOQA
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
    timestamp_flag, night_flag, dni_limit_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.check_dni_limits_QCRad` respectively
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
    timestamp_flag, night_flag, dhi_limit_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.check_dhi_limits_QCRad` respectively
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
    timestamp_flag, night_flag, poa_clearsky_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.check_poa_clearsky` respectively
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
    timestamp_flag, night_flag, temp_limit_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.check_temperature_limits` respectively
    """
    timestamp_flag, night_flag = validate_defaults(observation, values)
    temp_limit_flag = validator.check_temperature_limits(
        values, _return_mask=True)
    return timestamp_flag, night_flag, temp_limit_flag


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
    timestamp_flag, night_flag, wind_limit_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.check_wind_limits` respectively
    """
    timestamp_flag, night_flag = validate_defaults(observation, values)
    wind_limit_flag = validator.check_wind_limits(values,
                                                  _return_mask=True)
    return timestamp_flag, night_flag, wind_limit_flag


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
    timestamp_flag, night_flag, rh_limit_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.check_rh_limits` respectively
    """
    timestamp_flag, night_flag = validate_defaults(observation, values)
    rh_limit_flag = validator.check_rh_limits(values, _return_mask=True)
    return timestamp_flag, night_flag, rh_limit_flag


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
    timestamp_flag, night_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night` respectively
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
    *ghi_flags, stale_flag, interpolation_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validate_ghi`,
        :py:func:`.validator.detect_stale_values`,
        :py:func:`.validator.detect_interpolation`
    """
    ghi_flags = validate_ghi(observation, values)
    stale_flag = validator.detect_stale_values(values, _return_mask=True)
    interpolation_flag = validator.detect_interpolation(values,
                                                        _return_mask=True)
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
    timestamp_flag, night_flag, stale_flag, interpolation_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.detect_stale_values`,
        :py:func:`.validator.detect_interpolation`
    """
    timestamp_flag, night_flag = validate_defaults(observation, values)
    stale_flag = validator.detect_stale_values(values, _return_mask=True)
    interpolation_flag = validator.detect_interpolation(values,
                                                        _return_mask=True)
    return (timestamp_flag, night_flag, stale_flag, interpolation_flag)


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
    timestamp_flag, night_flag, stale_flag, interpolation_flag, clipping_flag : pandas.Series
        Integer bitmask series from
        :py:func:`.validator.check_timestamp_spacing`,
        :py:func:`.validator.check_irradiance_day_night`,
        :py:func:`.validator.detect_stale_values`,
        :py:func:`.validator.detect_interpolation`,
        :py:func:`.validator.detect_clipping`
    """  # NOQA
    timestamp_flag, night_flag = validate_defaults(observation, values)
    stale_flag = validator.detect_stale_values(values, _return_mask=True)
    interpolation_flag = validator.detect_interpolation(values,
                                                        _return_mask=True)
    clipping_flag = validator.detect_clipping(values, _return_mask=True)
    return (timestamp_flag, night_flag, stale_flag, interpolation_flag,
            clipping_flag)


IMMEDIATE_VALIDATION_FUNCS = {
    'air_temperature': validate_air_temperature,
    'wind_speed': validate_wind_speed,
    'ghi': validate_ghi,
    'dni': validate_dni,
    'dhi': validate_dhi,
    'poa_global': validate_poa_global,
    'relative_humidity': validate_relative_humidity
}


def immediate_observation_validation(access_token, observation_id, start, end,
                                     base_url=None):
    """
    Task that will run immediately after Observation values are uploaded to the
    API to validate the data.
    """
    session = APISession(access_token, base_url=base_url)
    observation = session.get_observation(observation_id)
    observation_values = session.get_observation_values(observation_id, start,
                                                        end)
    value_series = observation_values['value'].astype(float)
    quality_flags = observation_values['quality_flag'].copy()

    validation_func = IMMEDIATE_VALIDATION_FUNCS.get(
        observation.variable, validate_defaults)
    validation_flags = validation_func(observation, value_series)

    for flag in validation_flags:
        quality_flags |= flag

    quality_flags.name = 'quality_flag'
    observation_values.update(quality_flags)
    session.post_observation_values(observation_id, observation_values,
                                    params='donotvalidate')


DAILY_VALIDATION_FUNCS = {
    'ghi': validate_daily_ghi,
    'dc_power': validate_daily_dc_power,
    'ac_power': validate_daily_ac_power
}


def _daily_validation(session, observation, start, end, base_url):
    logger.info('Validating data for %s from %s to %s',
                observation.name, start, end)
    observation_values = session.get_observation_values(
        observation.observation_id, start, end).sort_index()
    value_series = observation_values['value'].astype(float)
    if len(value_series.dropna()) < 10:
        raise IndexError(
            'Data series does not have at least 10 datapoints to validate')
    quality_flags = observation_values['quality_flag'].copy()

    # if the variable has a daily check, run that, else run the
    # immediate validation, else validate timestamps
    validation_func = DAILY_VALIDATION_FUNCS.get(
        observation.variable, IMMEDIATE_VALIDATION_FUNCS.get(
            observation.variable, validate_defaults))
    validation_flags = validation_func(observation, value_series)

    for flag in validation_flags:
        quality_flags |= flag

    quality_flags.name = 'quality_flag'
    observation_values.update(quality_flags)
    return _group_continuous_week_post(
        session, observation, observation_values)


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


def daily_single_observation_validation(access_token, observation_id, start,
                                        end, base_url=None):
    """
    Task that expects a longer, likely daily timeseries of Observation values
    that will be validated.
    """
    session = APISession(access_token, base_url=base_url)
    observation = session.get_observation(observation_id)
    try:
        _daily_validation(session, observation, start, end, base_url)
    except IndexError:
        logger.warning(
            'Daily validation for %s failed: not enough values',
            observation.name)


def daily_observation_validation(access_token, start, end, base_url=None):
    """
    Run the daily observation validation for all observations that the user
    has access to in their organization.
    """
    session = APISession(access_token, base_url=base_url)
    user_info = session.get_user_info()
    observations = [obs for obs in session.list_observations()
                    if obs.provider == user_info['organization']]
    for observation in observations:
        try:
            _daily_validation(session, observation, start, end, base_url)
        except IndexError:
            logger.warning(('Skipping daily validation of %s '
                            'not enough values'), observation.name)
            continue
