import pandas as pd
from pvlib.irradiance import get_extra_radiation


from solarforecastarbiter import pvmodel
from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.validation import validator


def _solpos_dni_extra(site, times):
    solar_position = pvmodel.calculate_solar_position(
        site.latitude, site.longitude, site.elevation, times)
    dni_extra = get_extra_radiation(times)
    return solar_position, dni_extra


def validate_ghi(site, values):
    solar_position, dni_extra = _solpos_dni_extra(site, values.index)
    clearsky = pvmodel.calculate_clearsky(
        site.latitude, site.longitude, site.elevation,
        solar_position['apparent_zenith'])

    new_flags = validator.check_irradiance_day_night(solar_position['zenith'],
                                                     return_bool=False)
    new_flags |= validator.check_ghi_limits_QCRad(values,
                                                  solar_position['zenith'],
                                                  dni_extra)
    new_flags |= validator.check_ghi_clearsky(values, clearsky['ghi'])
    return new_flags


def validate_dni(site, values):
    solar_position, dni_extra = _solpos_dni_extra(site, values.index)
    new_flags = validator.check_irradiance_day_night(solar_position['zenith'],
                                                     return_bool=False)
    new_flags |= validator.check_dni_limits_QCRad(values,
                                                  solar_position['zenith'],
                                                  dni_extra)
    return new_flags


def validate_dhi(site, values):
    solar_position, dni_extra = _solpos_dni_extra(site, values.index)
    new_flags = validator.check_irradiance_day_night(solar_position['zenith'],
                                                     return_bool=False)
    new_flags |= validator.check_dhi_limits_QCRad(values,
                                                  solar_position['zenith'],
                                                  dni_extra)
    return new_flags


VARIABLE_VALIDATION_FUNCS = {
    'ghi': validate_ghi,
    'dni': validate_dni,
    'dhi': validate_dhi,
}


def immediate_observation_validation(access_token, observation_id, start, end):
    session = APISession(access_token)
    observation = session.get_observation(observation_id)
    observation_values = session.get_observation_values(observation_id, start,
                                                        end)
    value_series = observation_values['value']
    quality_flags = observation_values['quality_flag'].copy()

    quality_flags |= validator.check_timestamp_spacing(
        value_series.index, observation.interval_length, return_bool=False)

    validation_func = VARIABLE_VALIDATION_FUNCS.get(
        observation.variable,
        lambda x, y: pd.Series(0, index=value_series.index))

    quality_flags |= validation_func(observation.variable)(
        observation.site, value_series)

    # session.post
