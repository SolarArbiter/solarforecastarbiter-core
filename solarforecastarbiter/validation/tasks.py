from pvlib.irradiance import get_extra_radiation


from solarforecastarbiter import pvmodel
from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.validation import validator


def _validate_timestamp(observation, values):
    return validator.check_timestamp_spacing(
        values.index, observation.interval_length, _return_mask=True)


def _solpos_dni_extra(observation, values):
    solar_position = pvmodel.calculate_solar_position(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, values.index)
    dni_extra = get_extra_radiation(values.index)
    timestamp_flag = _validate_timestamp(observation, values)
    night_flag = validator.check_irradiance_day_night(solar_position['zenith'],
                                                      _return_mask=True)
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
    tuple
        Tuple of integer bitmask series of flags from the following tests, in
        order,
        `validator.check_timestamp_spacing`,
        `validator.check_irradiance_day_night`,
        `validator.check_ghi_limits_QCRad`,
        `validator.check_ghi_clearsky`
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
    return timestamp_flag, night_flag, ghi_limit_flag, ghi_clearsky_flag


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
    tuple
        Tuple of integer bitmask series of flags from the following tests, in
        order,
        `validator.check_timestamp_spacing`,
        `validator.check_irradiance_day_night`,
        `validator.check_dni_limits_QCRad`
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
    tuple
        Tuple of integer bitmask series of flags from the following tests, in
        order,
        `validator.check_timestamp_spacing`,
        `validator.check_irradiance_day_night`,
        `validator.check_dhi_limits_QCRad`
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
    tuple
        Tuple of integer bitmask series of flags from the following tests, in
        order,
        `validator.check_timestamp_spacing`,
        `validator.check_irradiance_day_night`,
        `validator.check_poa_clearsky`
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
    tuple
        Tuple of integer bitmask series of flags from the following tests, in
        order,
        `validator.check_timestamp_spacing`,
        `validator.check_temperature_limits`
    """
    timestamp_flag = _validate_timestamp(observation, values)
    temp_limit_flag = validator.check_temperature_limits(
        values, _return_mask=True)
    return timestamp_flag, temp_limit_flag


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
    tuple
        Tuple of integer bitmask series of flags from the following tests, in
        order,
        `validator.check_timestamp_spacing`,
        `validator.check_wind_limits`
    """
    timestamp_flag = _validate_timestamp(observation, values)
    wind_limit_flag = validator.check_wind_limits(values,
                                                  _return_mask=True)
    return timestamp_flag, wind_limit_flag


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
    tuple
        Tuple of integer bitmask series of flags from the following tests, in
        order,
        `validator.check_timestamp_spacing`,
        `validator.check_rh_limits`
    """
    timestamp_flag = _validate_timestamp(observation, values)
    rh_limit_flag = validator.check_rh_limits(values, _return_mask=True)
    return timestamp_flag, rh_limit_flag


def validate_timestamp(observation, values):
    """
    Run validation checks on an observation.

    Parameters
    ----------
    observation : solarforecastarbiter.datamodel.Observation
       Observation object that the data is associated with
    values : pandas.Series
       Series of observation values

    Returns
    -------
    tuple
        Tuple of integer bitmask series of flags from the following tests, in
        order,
        `validator.check_timestamp_spacing`
    """
    return (_validate_timestamp(observation, values),)


IMMEDIATE_VALIDATION_FUNCS = {
    'air_temperature': validate_air_temperature,
    'wind_speed': validate_wind_speed,
    'ghi': validate_ghi,
    'dni': validate_dni,
    'dhi': validate_dhi,
    'poa_global': validate_poa_global,
    'relative_humidity': validate_relative_humidity
}


def immediate_observation_validation(access_token, observation_id, start, end):
    """
    Task that will run immediately after Observation values are uploaded to the
    API to validate the data.
    """
    session = APISession(access_token)
    observation = session.get_observation(observation_id)
    observation_values = session.get_observation_values(observation_id, start,
                                                        end)
    value_series = observation_values['value']
    quality_flags = observation_values['quality_flag'].copy()

    validation_func = IMMEDIATE_VALIDATION_FUNCS.get(
        observation.variable, validate_timestamp)
    validation_flags = validation_func(observation, value_series)

    for flag in validation_flags:
        quality_flags |= flag

    quality_flags.name = 'quality_flag'
    observation_values.update(quality_flags)
    session.post_observation_values(observation_id, observation_values)
