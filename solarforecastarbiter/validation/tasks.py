import pandas as pd
from pvlib.irradiance import get_extra_radiation


from solarforecastarbiter import pvmodel
from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.validation import validator, quality_mapping


def _validate_timestamp(observation, values):
    return validator.check_timestamp_spacing(
        values.index, observation.interval_length, return_bool=False)


def _solpos_dni_extra(observation, values):
    solar_position = pvmodel.calculate_solar_position(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, values.index)
    dni_extra = get_extra_radiation(values.index)
    new_flags = _validate_timestamp(observation, values)
    new_flags |= validator.check_irradiance_day_night(solar_position['zenith'],
                                                      return_bool=False)
    return solar_position, dni_extra, new_flags


def validate_ghi(observation, values):
    solar_position, dni_extra, new_flags = _solpos_dni_extra(observation,
                                                             values)
    clearsky = pvmodel.calculate_clearsky(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, solar_position['apparent_zenith'])

    new_flags |= validator.check_ghi_limits_QCRad(values,
                                                  solar_position['zenith'],
                                                  dni_extra,
                                                  return_bool=False)
    new_flags |= validator.check_ghi_clearsky(values, clearsky['ghi'],
                                              return_bool=False)
    return new_flags


def validate_dni(observation, values):
    solar_position, dni_extra, new_flags = _solpos_dni_extra(observation,
                                                             values)
    new_flags |= validator.check_dni_limits_QCRad(values,
                                                  solar_position['zenith'],
                                                  dni_extra,
                                                  return_bool=False)
    return new_flags


def validate_dhi(observation, values):
    solar_position, dni_extra, new_flags = _solpos_dni_extra(observation,
                                                             values)
    new_flags |= validator.check_dhi_limits_QCRad(values,
                                                  solar_position['zenith'],
                                                  dni_extra,
                                                  return_bool=False)
    return new_flags


def validate_poa_global(observation, values):
    solar_position, dni_extra, new_flags = _solpos_dni_extra(observation,
                                                             values)
    clearsky = pvmodel.calculate_clearsky(
        observation.site.latitude, observation.site.longitude,
        observation.site.elevation, solar_position['apparent_zenith'])
    aoi_func = pvmodel.aoi_func_factory(observation.site.modeling_parameters)
    poa_clearsky = pvmodel.calculate_poa_effective(
        aoi_func=aoi_func, apparent_zenith=solar_position['apparent_zenith'],
        azimuth=solar_position['azimuth'], ghi=clearsky['ghi'],
        dni=clearsky['dni'], dhi=clearsky['dhi'])
    new_flags |= validator.check_poa_clearsky(values, poa_clearsky,
                                              return_bool=False)
    return new_flags


def validate_air_temperature(observation, values):
    new_flags = _validate_timestamp(observation, values)
    new_flags |= validator.check_temperature_limits(values, return_bool=False)
    return new_flags


def validate_wind_speed(observation, values):
    new_flags = _validate_timestamp(observation, values)
    new_flags |= validator.check_wind_limits(values, return_bool=False)
    return new_flags


def validate_relative_humidity(observation, values):
    new_flags = _validate_timestamp(observation, values)
    new_flags |= validator.check_rh_limits(values, return_bool=False)
    return new_flags


def validate_nothing(observation, values):
    @quality_mapping.mask_flags('OK', invert=False)
    def noop():
        return pd.Series(1, index=values.index)
    return noop(return_bool=False)


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
        observation.variable, validate_nothing)

    quality_flags |= validation_func(observation.variable)(
        observation, value_series)

    quality_flags.name = 'quality_flag'
    observation_values.update(quality_flags)
    session.post_observation_values(observation_id, observation_values)
