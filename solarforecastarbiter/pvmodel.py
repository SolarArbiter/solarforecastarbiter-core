"""
Calculate AC power and modeling intermediates from system metadata,
times, and weather data.

Steps are:

1. Calculate solar position using solar_position
2. If not already known, calculate 3 irradiance components from measured
   GHI using irradiance_components or modeled clear sky using clearsky.
3. calculate_poa_effective
4. calculate_power

Steps 3 and 4 are bundled in :py:func:`irradiance_to_power`
"""

from functools import partial

import pvlib

from solarforecastarbiter import datamodel


def calculate_solar_position(latitude, longitude, elevation, times):
    """
    Calculates solar position using pvlib's implementation of NREL SPA.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    times : pd.DatetimeIndex

    Returns
    -------
    solar_position : pd.DataFrame
        The DataFrame will have the following columns: apparent_zenith
        (degrees), zenith (degrees), apparent_elevation (degrees),
        elevation (degrees), azimuth (degrees),
        equation_of_time (minutes).
    """
    solpos = pvlib.solarposition.get_solarposition(times, latitude,
                                                   longitude,
                                                   altitude=elevation,
                                                   method='nrel_numpy')
    return solpos


def complete_irradiance_components(ghi, zenith):
    """
    Uses the Erbs model to calculate DNI and DHI from GHI.

    Parameters
    ----------
    ghi : pd.Series
    zenith : pd.Series
        Solar zenith (not-refraction corrected)

    Returns
    -------
    dni : pd.Series, dhi : pd.Series
    """
    dni_dhi = pvlib.irradiance.erbs(ghi, zenith, ghi.index)
    return dni_dhi['dni'], dni_dhi['dhi']


def calculate_clearsky(latitude, longitude, elevation, apparent_zenith):
    """
    Calculates clear sky irradiance using the Ineichen model and the SoDa
    climatological turbidity data set.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    apparent_zenith : pd.Series
        Solar apparent zenith

    Returns
    -------
    cs : pd.DataFrame
        Columns are ghi, dni, dhi.
    """
    airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
    pressure = pvlib.atmosphere.alt2pres(elevation)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    tl = pvlib.clearsky.lookup_linke_turbidity(apparent_zenith.index,
                                               latitude,
                                               longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(apparent_zenith.index)
    cs = pvlib.clearsky.ineichen(apparent_zenith, am_abs, tl,
                                 dni_extra=dni_extra,
                                 altitude=elevation)
    return cs


def _system_tilt_azimuth_aoi(modeling_parameters, apparent_zenith, azimuth):
    """
    Unpack modeling parameters attributes into function calls.

    Parameters
    ----------
    modeling_parameters : datamodel.FixedTiltModelingParameters or
                          datamodel.SingleAxisModelingParameters
    apparent_zenith : pd.Series
        Solar apparent zenith
    azimuth : pd.Series
        Solar azimuth

    Returns
    -------
    surface_tilt : pd.Series, surface_azimuth : pd.Series, aoi : pd.Series

    Raises
    ------
    TypeError if modeling_parameters is invalid.
    """
    if isinstance(modeling_parameters,
                  datamodel.FixedTiltModelingParameters):
        return _aoi_fixed(
            modeling_parameters.surface_tilt,
            modeling_parameters.surface_azimuth,
            apparent_zenith, azimuth
        )
    elif isinstance(modeling_parameters,
                    datamodel.SingleAxisModelingParameters):
        return _aoi_tracking(
            modeling_parameters.axis_tilt,
            modeling_parameters.axis_azimuth,
            modeling_parameters.maximum_rotation_angle,
            modeling_parameters.backtrack,
            modeling_parameters.ground_coverage_ratio,
            apparent_zenith,
            azimuth
        )
    else:
        raise TypeError('Invalid modeling_parameters type %s' %
                        type(modeling_parameters))


def aoi_func_factory(modeling_parameters):
    """
    Create a function to calculate AOI, surface tilt, and surface
    azimuth from system modeling_parameters.

    Parameters
    ----------
    modeling_parameters : datamodel.FixedTiltModelingParameters or
                          datamodel.SingleAxisModelingParameters

    Returns
    -------
    function
        Function that accepts two arguments (apparent_zenith, azimuth)
        and returns three series (surface_tilt, surface_azimuth, aoi)

    Raises
    ------
    TypeError if modeling_parameters is invalid.
    """
    if isinstance(modeling_parameters,
                  datamodel.FixedTiltModelingParameters):
        return partial(
            _aoi_fixed,
            modeling_parameters.surface_tilt,
            modeling_parameters.surface_azimuth
        )
    elif isinstance(modeling_parameters,
                    datamodel.SingleAxisModelingParameters):
        return partial(
            _aoi_tracking,
            modeling_parameters.axis_tilt,
            modeling_parameters.axis_azimuth,
            modeling_parameters.maximum_rotation_angle,
            modeling_parameters.backtrack,
            modeling_parameters.ground_coverage_ratio
        )
    else:
        raise TypeError('Invalid modeling_parameters type %s' %
                        type(modeling_parameters))


def _aoi_fixed(surface_tilt, surface_azimuth, apparent_zenith, azimuth):
    """
    Calculate AOI for fixed system, bundle return with tilt, azimuth for
    consistency with similar tracker function.

    Parameters
    ----------
    surface_tilt : float
    surface_azimuth : float
    apparent_zenith : pd.Series
        Solar apparent zenith
    azimuth : pd.Series
        Solar azimuth

    Returns
    -------
    surface_tilt : pd.Series, surface_azimuth : pd.Series, aoi : pd.Series
    """
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               apparent_zenith, azimuth)
    return surface_tilt, surface_azimuth, aoi


def _aoi_tracking(axis_tilt, axis_azimuth, maximum_rotation_angle, backtrack,
                  ground_coverage_ratio, apparent_zenith, azimuth):
    """
    Calculate AOI, surface tilt, and surface azimuth for tracking system.

    Parameters
    ----------
    axis_tilt : float
    axis_azimuth : float
    maximum_rotation_angle : float
    backtrack : bool
    ground_coverage_ratio : float
    apparent_zenith : pd.Series
        Solar apparent zenith
    azimuth : pd.Series
        Solar azimuth

    Returns
    -------
    surface_tilt : pd.Series, surface_azimuth : pd.Series, aoi : pd.Series
    """
    tracking = pvlib.tracking.singleaxis(
        apparent_zenith,
        azimuth,
        axis_tilt=axis_tilt,
        axis_azimuth=axis_azimuth,
        max_angle=maximum_rotation_angle,
        backtrack=backtrack,
        gcr=ground_coverage_ratio
    )
    surface_tilt = tracking['surface_tilt']
    surface_azimuth = tracking['surface_azimuth']
    aoi = tracking['aoi']
    return surface_tilt, surface_azimuth, aoi


def calculate_poa_effective(surface_tilt, surface_azimuth, aoi,
                            apparent_zenith, azimuth, ghi, dni, dhi):
    """
    Calculate effective plane of array irradiance from system metadata,
    solar position, and irradiance components. Accounts for AOI losses.

    Parameters
    ----------
    surface_tilt : float or pd.Series
    surface_azimuth : float or pd.Series
    aoi : pd.Series
    apparent_zenith : pd.Series
        Solar apparent zenith
    azimuth : pd.Series
        Solar azimuth
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series

    Returns
    -------
    poa_effective : pd.Series
    """
    dni_extra = pvlib.irradiance.get_extra_radiation(apparent_zenith.index)
    poa_sky_diffuse = pvlib.irradiance.haydavies(
        surface_tilt, surface_azimuth,
        dhi, dni, dni_extra,
        solar_zenith=apparent_zenith,
        solar_azimuth=azimuth,
        dni_extra=dni_extra, model='haydavies')
    # assumes albedo = 0.25
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(surface_tilt, ghi)
    aoi_modifier = pvlib.pvsystem.physicaliam(aoi)
    beam_component = dni * pvlib.tools.cosd(aoi)
    beam_component = beam_component.clip(lower=0) * aoi_modifier
    poa_effective = beam_component + poa_sky_diffuse + poa_ground_diffuse
    return poa_effective


def calculate_poa_effective_func(aoi_func, apparent_zenith, azimuth,
                                 ghi, dni, dhi):
    """
    Calculate effective plane of array irradiance from system metadata,
    solar position, and irradiance components. Accounts for AOI losses.

    Parameters
    ----------
    aoi_func : function
        Function with arguments (apparent_zenith, azimuth) and returns
        surface_tilt, surface_azimuth, aoi
    apparent_zenith : pd.Series
        Solar apparent zenith
    azimuth : pd.Series
        Solar azimuth
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series

    Returns
    -------
    poa_effective : pd.Series
    """
    surface_tilt, surface_azimuth, aoi = aoi_func(apparent_zenith, azimuth)
    dni_extra = pvlib.irradiance.get_extra_radiation(apparent_zenith.index)
    poa_sky_diffuse = pvlib.irradiance.haydavies(
        surface_tilt, surface_azimuth,
        dhi, dni, dni_extra,
        solar_zenith=apparent_zenith,
        solar_azimuth=azimuth,
        dni_extra=dni_extra, model='haydavies')
    # assumes albedo = 0.25
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(surface_tilt, ghi)
    aoi_modifier = pvlib.pvsystem.physicaliam(aoi)
    beam_component = dni * pvlib.tools.cosd(aoi)
    beam_component = beam_component.clip(lower=0) * aoi_modifier
    poa_effective = beam_component + poa_sky_diffuse + poa_ground_diffuse
    return poa_effective


def calculate_power(dc_capacity, dc_loss_factor, ac_capacity, ac_loss_factor,
                    poa_effective, temp_air=20, wind_speed=1):
    """
    Calcuate AC power from system metadata, plane of array irradiance,
    and weather data using the PVWatts model.

    Parameters
    ----------
    dc_capacity : float
    dc_loss_factor : float
    ac_capacity : float
    ac_loss_factor : float
    poa_effective : pd.Series
    temp_air : pd.Series, default 20
    wind_speed : pd.Series, default 1

    Returns
    -------
    ac_power : pd.Series
    """
    pvtemps = pvlib.pvsystem.sapm_celltemp(poa_effective,
                                           wind_speed, temp_air)
    dc = pvlib.pvsystem.pvwatts_dc(poa_effective, pvtemps['temp_cell'],
                                   dc_capacity)
    dc *= (1 - dc_loss_factor / 100)
    ac = pvlib.pvsystem.pvwatts_ac(dc, dc_capacity)
    ac = ac.clip(upper=ac_capacity)
    ac *= (1 - ac_loss_factor / 100)
    return ac


def irradiance_to_power(modeling_parameters, apparent_zenith, azimuth, ghi,
                        dni, dhi, temp_air=20, wind_speed=1):
    """
    Calcuate AC power from system metadata, solar position, and
    ghi, dni, dhi.

    Parameters
    ----------
    modeling_parameters : datamodel.FixedTiltModelingParameters or
                          datamodel.SingleAxisModelingParameters
    apparent_zenith : pd.Series
        Solar apparent zenith
    azimuth : pd.Series
        Solar azimuth
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series
    temp_air : pd.Series, default 20
    wind_speed : pd.Series, default 1

    Returns
    -------
    ac_power : pd.Series
    """
    surface_tilt, surface_azimuth, aoi = _system_tilt_azimuth_aoi(
        modeling_parameters, apparent_zenith, azimuth)
    poa_effective = calculate_poa_effective(
        surface_tilt, surface_azimuth, aoi, apparent_zenith, azimuth,
        ghi, dni, dhi)
    ac = calculate_power(
        modeling_parameters.dc_capacity,
        modeling_parameters.dc_loss_factor,
        modeling_parameters.ac_capacity,
        modeling_parameters.ac_loss_factor,
        poa_effective,
        temp_air=temp_air,
        wind_speed=wind_speed)
    return ac


def irradiance_to_power2(modeling_parameters, apparent_zenith, azimuth, ghi,
                         dni, dhi, temp_air=20, wind_speed=1):
    """
    Calcuate AC power from system metadata, solar position, and
    ghi, dni, dhi.

    Parameters
    ----------
    modeling_parameters : datamodel.FixedTiltModelingParameters or
                          datamodel.SingleAxisModelingParameters
    apparent_zenith : pd.Series
        Solar apparent zenith
    azimuth : pd.Series
        Solar azimuth
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series
    temp_air : pd.Series, default 20
    wind_speed : pd.Series, default 1

    Returns
    -------
    ac_power : pd.Series
    """
    aoi_func = aoi_func_factory(modeling_parameters)
    poa_effective = calculate_poa_effective_func(
        aoi_func, apparent_zenith, azimuth, ghi, dni, dhi)
    ac = calculate_power(
        modeling_parameters.dc_capacity,
        modeling_parameters.dc_loss_factor,
        modeling_parameters.ac_capacity,
        modeling_parameters.ac_loss_factor,
        poa_effective,
        temp_air=temp_air,
        wind_speed=wind_speed)
    return ac
