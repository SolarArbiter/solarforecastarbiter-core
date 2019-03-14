"""
Calculate AC power and modeling intermediates from system metadata,
times, and weather data.

Steps are:

1. Calculate solar position using solar_position
2. If not already known, calculate 3 irradiance components from measured
   GHI using irradiance_components or modeled clear sky using clearsky.
3. calculate_poa_effective
4. calculate_power
"""

import pvlib

from solarforecastarbiter import datamodel


def solar_position(site, times):
    """
    Calculates solar position using pvlib's implementation of NREL SPA.

    Parameters
    ----------
    site : datamodel.Site
    times : pd.DatetimeIndex

    Returns
    -------
    solar_position : pd.DataFrame
        The DataFrame will have the following columns: apparent_zenith
        (degrees), zenith (degrees), apparent_elevation (degrees),
        elevation (degrees), azimuth (degrees),
        equation_of_time (minutes).
    """
    solpos = pvlib.solarposition.get_solarposition(times, site.latitude,
                                                   site.longitude,
                                                   altitude=site.elevation,
                                                   method='nrel_numpy')
    return solpos


def irradiance_components(ghi, solar_position):
    """
    Uses the Erbs model to calculate DNI and DHI from GHI.

    Parameters
    ----------
    ghi : pd.Series
    solar_position : pd.DataFrame

    Returns
    -------
    irrads : pd.DataFrame
        Columns are ghi, dni, dhi.
    """
    dni_dhi = pvlib.irradiance.erbs(ghi, solar_position['zenith'], ghi.index)
    dni_dhi['ghi'] = ghi
    # erbs also returns kt, so get subset of what we care about
    irrads = dni_dhi[['ghi', 'dni', 'dhi']]
    return irrads


def clearsky(site, solar_position):
    """
    Calculates clear sky irradiance using the Ineichen model and the SoDa
    climatological turbidity data set.

    Parameters
    ----------
    site : datamodel.Site
    solar_position : pd.DataFrame

    Returns
    -------
    cs : pd.DataFrame
        Columns are ghi, dni, dhi.
    """
    airmass = pvlib.atmosphere.get_relative_airmass(
        solar_position['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(site.elevation)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    tl = pvlib.clearsky.lookup_linke_turbidity(solar_position.index,
                                               site.latitude,
                                               site.longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(solar_position.index)
    cs = pvlib.clearsky.ineichen(solar_position['apparent_zenith'], am_abs, tl,
                                 dni_extra=dni_extra,
                                 altitude=site.elevation)
    return cs


def system_tilt_azimuth_aoi(system, solar_position):
    """
    Lookup or calculate system surface tilt, surface azimuth, and AOI
    for fixed tilt or single axis tracking systems.

    Parameters
    ----------
    system : datamodel.SolarPowerPlant
    solar_position : pd.DataFrame

    Returns
    -------
    surface_tilt, surface_azimuth, aoi

    Raises
    ------
    TypeError if system.modeling_parameters is invalid.
    """
    solar_zenith = solar_position['apparent_zenith']
    solar_azimuth = solar_position['azimuth']
    if isinstance(system.modeling_parameters,
                  datamodel.FixedTiltModelingParameters):
        surface_tilt = system.modeling_parameters.surface_tilt
        surface_azimuth = system.modeling_parameters.surface_azimuth
        aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                                   solar_zenith, solar_azimuth)
    elif isinstance(system.modeling_parameters,
                    datamodel.SingleAxisModelingParameters):
        tracking = pvlib.tracking.singleaxis(
            solar_zenith,
            solar_azimuth,
            axis_tilt=system.modeling_parameters.axis_tilt,
            axis_azimuth=system.modeling_parameters.axis_azimuth,
            max_angle=system.modeling_parameters.maximum_rotation_angle,
            backtrack=system.modeling_parameters.backtrack,
            gcr=system.modeling_parameters.ground_coverage_ratio
        )
        surface_tilt = tracking['surface_tilt']
        surface_azimuth = tracking['surface_azimuth']
        aoi = tracking['aoi']
    else:
        raise TypeError('Invalid system.modeling_parameters type %s' %
                        type(system.modeling_parameters))
    return surface_tilt, surface_azimuth, aoi


def calculate_poa_effective(system, solar_position, irradiance):
    """
    Calculate effective plane of array irradiance from system metadata,
    solar position, and irradiance components. Accounts for AOI losses.

    Parameters
    ----------
    system : datamodel.SolarPowerPlant
    solar_position : pd.DataFrame
    irradiance : pd.DataFrame

    Returns
    -------
    poa_effective : pd.Series
    """
    surface_tilt, surface_azimuth, aoi = system_tilt_azimuth_aoi(
        system, solar_position)

    dni_extra = pvlib.irradiance.get_extra_radiation(solar_position.index)

    poa_sky_diffuse = pvlib.irradiance.haydavies(
        surface_tilt, surface_azimuth,
        irradiance['dhi'], irradiance['dni'], dni_extra,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni_extra=dni_extra, model='haydavies')
    # assumes albedo = 0.25
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(surface_tilt,
                                                             irradiance['ghi'])
    aoi_modifier = pvlib.pvsystem.physicaliam(aoi)
    beam_component = irradiance['dni'] * pvlib.tools.cosd(aoi)
    beam_component = beam_component.clip(lower=0) * aoi_modifier
    poa_effective = beam_component + poa_sky_diffuse + poa_ground_diffuse
    return poa_effective


def calculate_power(system, poa_effective, temp_air=20, wind_speed=1):
    """
    Calcuate AC power from system metadata, plane of array irradiance,
    and weather data using the PVWatts model.

    Parameters
    ----------
    system : datamodel.SolarPowerPlant
    poa_effective : pd.Series
    temp_air : pd.Series
    wind_speed : pd.Series

    Returns
    -------
    ac_power : pd.Series
    """
    pvtemps = pvlib.pvsystem.sapm_celltemp(poa_effective,
                                           wind_speed, temp_air)
    dc = pvlib.pvsystem.pvwatts_dc(poa_effective, pvtemps['temp_cell'],
                                   system.modeling_parameters.dc_capacity)
    dc *= (1 - system.modeling_parameters.dc_loss_factor / 100)
    ac = pvlib.pvsystem.pvwatts_ac(dc, system.modeling_parameters.dc_capacity)
    ac = ac.clip(upper=system.modeling_parameters.ac_capacity)
    ac *= (1 - system.modeling_parameters.ac_loss_factor / 100)
    return ac
