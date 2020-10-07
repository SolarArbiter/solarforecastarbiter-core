# maybe rename nwp.py or models_nwp.py
"""
Default processing functions for data from NOAA weather models.

All public functions in this module have the same signature.

The functions accept:

  * latitude : float
  * longitude : float
  * elevation : float
  * init_time : pd.Timestamp
      Full datetime of a model initialization
  * start : pd.Timestamp
  * end : pd.Timestamp
  * load_forecast : function
      A function that accepts the arguments above and returns the
      correct data. Enables users to supply their own data independently
      of the `solarforecastarbiter.io` module.
  * __model : str
      The NWP model that the processing function is associated with.

The functions return a tuple of:

  * ghi : pd.Series
  * dni : pd.Series
  * dhi : pd.Series
  * air_temperature : pd.Series
  * wind_speed : pd.Series
  * resampler : function
      A function that resamples data to the appropriate frequency.
      This function is to be applied to all forecast variables after
      power is calculated.
  * solar_position_calculator : function
      A function that returns solar position at the ghi forecast times
      (after internal interpolation, before external resampling). The
      function will return results immediately if solar position is
      already known or will call a solar position calculation algorithm
      and then return.

Most of the model functions return forecast data interpolated to 5
minute frequency. Interpolation to 5 minutes reduces the errors
associated with solar position and irradiance to power models (these
models assume instantaneous inputs). Why 5 minutes? It's a round number
that produces about 10 data points per hour, so it's reasonable for hour
average calculations. It is expected that after calculating power, users
will apply the `resampler` function to both the weather and power
forecasts. These functions may be most useful if the user would like to
understand the performance of a NWP model with modest post-processing.

Several model functions return instantaneous data that is coincident
with the NWP model forecast time (15 minutes or hourly, depending on the
NWP model). The resamplers returned by these functions do not modify the
data (though they do define the frequency attribute of the data's
DatetimeIndex). These functions may most useful if the user would like
to understand the raw performance of a NWP model.

The functions in this module accept primitives (floats, strings, etc.)
rather than objects defined in :py:mod:`solarforecastarbiter.datamodel`
because we anticipate that these functions may be of more general use
and that functions that accept primitives may be easier to maintain in
the long run.
"""
from functools import partial
import inspect


from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.io.nwp import load_forecast
from solarforecastarbiter.io.utils import adjust_start_end_for_interval_label
from solarforecastarbiter.reference_forecasts import forecast

import pandas as pd


def get_nwp_model(func):
    """Get the NWP model string from a modeling function"""
    return inspect.signature(func).parameters['__model'].default


def _resample_using_cloud_cover(latitude, longitude, elevation,
                                cloud_cover, air_temperature, wind_speed,
                                start, end, interval_label,
                                fill_method, solar_position=None):
    """
    Calculate all irradiance components from cloud cover.

    Cloud cover from GFS is an interval average with ending label.
    Air temperature and wind speed are instantaneous values.
    Intervals are 1, 3, or 6 hours in length.

    Cloud cover, air temperature, and wind speed from RAP and NAM are
    instantaneous.

    To accurately convert from cloud cover to irradiance, we need to
    interpolate this data to subhourly resolution because solar position
    and PV power calculations assume instantaneous inputs at each time.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    cloud_cover : pd.Series
    air_temperature : pd.Series
    wind_speed : pd.Series
    start : pd.Timestamp
    end : pd.Timestamp
    interval_label : str
        beginning, ending, or instant
    fill_method : str
        'bfill' (recommended for GFS), 'interpolate' (recommended for
        NAM/RAP), or any other method of pd.Series.
    solar_position : pd.DataFrame or None
        Provide a DataFrame to avoid unnecessary recomputation for e.g.
        GEFS members. If None, solar position is computed.

    Returns
    -------
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series
    air_temperature : pd.Series
    wind_speed : pd.Series
    resampler : function
    sol_pos_calculator : function
        When called, immediatedly returns pre-computed solar position.
    """
    # Resample cloud cover, temp, and wind to higher temporal resolution
    # because solar position and PV power calculations assume instantaneous
    # inputs. Why 5 minutes? It's a round number that produces order 10 data
    # points per hour, so it's reasonable for hour average calculations.
    # Cloud cover should be filled backwards for GFS because model output
    # represents average over the previous hour and interpolated for RAP and
    # NAM because model outputs represent instantaneous values. Air temperature
    # and wind are interpolated because model output represents
    # instantaneous values.
    freq = '5min'
    start_adj, end_adj = adjust_start_end_for_interval_label(interval_label,
                                                             start, end)
    cloud_cover = forecast.reindex_fill_slice(
        cloud_cover, freq=freq, start=start, end=end,
        start_slice=start_adj, end_slice=end_adj,
        fill_method=fill_method)
    resample_fill_slicer = partial(
        forecast.reindex_fill_slice, freq=freq, start=start, end=end,
        start_slice=start_adj, end_slice=end_adj, fill_method='interpolate')
    air_temperature, wind_speed = [
        resample_fill_slicer(v) for v in (air_temperature, wind_speed)
    ]
    if solar_position is None:
        solar_position = pvmodel.calculate_solar_position(
            latitude, longitude, elevation, cloud_cover.index)
    ghi, dni, dhi = forecast.cloud_cover_to_irradiance(
        latitude, longitude, elevation, cloud_cover,
        solar_position['apparent_zenith'], solar_position['zenith'])

    label = datamodel.CLOSED_MAPPING[interval_label]
    resampler = partial(forecast.resample, freq='1h', label=label)

    def solar_pos_calculator(): return solar_position

    return (ghi, dni, dhi, air_temperature, wind_speed,
            resampler, solar_pos_calculator)


def _ghi_to_dni_dhi(latitude, longitude, elevation, ghi):
    """
    Calculate DNI, DHI from GHI and calculated solar position.
    """
    solar_position = pvmodel.calculate_solar_position(
        latitude, longitude, elevation, ghi.index)
    dni, dhi = pvmodel.complete_irradiance_components(
        ghi, solar_position['zenith'])

    def solar_pos_calculator(): return solar_position
    return dni, dhi, solar_pos_calculator


def hrrr_subhourly_to_subhourly_instantaneous(latitude, longitude, elevation,
                                              init_time, start, end,
                                              interval_label,
                                              load_forecast=load_forecast,
                                              *, __model='hrrr_subhourly'):
    """
    Subhourly (15 min) instantantaneous HRRR forecast.
    GHI, DNI, DHI directly from model.
    Max forecast horizon 18 or 36 hours (0Z, 6Z, 12Z, 18Z).

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this point.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this point.
    interval_label : str
        Must be instant
    """
    start_adj, end_adj = adjust_start_end_for_interval_label(
        interval_label, start, end, limit_instant=True)
    ghi, dni, dhi, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start_adj, end_adj, __model)
    # resampler takes 15 min instantaneous in, retuns 15 min instantaneous out
    # still want to call resample, rather than pass through lambda x: x
    # so that DatetimeIndex has well-defined freq attribute
    resampler = partial(forecast.resample, freq='15min')
    solar_pos_calculator = partial(
        pvmodel.calculate_solar_position, latitude, longitude, elevation,
        ghi.index)
    return (ghi, dni, dhi, air_temperature, wind_speed,
            resampler, solar_pos_calculator)


def hrrr_subhourly_to_hourly_mean(latitude, longitude, elevation,
                                  init_time, start, end, interval_label,
                                  load_forecast=load_forecast,
                                  *, __model='hrrr_subhourly'):
    """
    Hourly mean HRRR forecast.
    GHI, DNI, DHI directly from model, resampled.
    Max forecast horizon 18 or 36 hours (0Z, 6Z, 12Z, 18Z).

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* and exclusive of this instant if
        interval_label is *ending*.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this instant if
        interval_label is *beginning* and inclusive of this instant if
        interval_label is *ending*.
    interval_label : str
        Must be *beginning* or *ending*
    """
    ghi, dni, dhi, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, __model)
    # Interpolate irrad, temp, wind data to 5 min to
    # minimize weather to power errors. Either start or end is outside of
    # forecast, but is needed for subhourly interpolation. After
    # interpolation, we slice the extra point out of the interpolated
    # output.
    start_adj, end_adj = adjust_start_end_for_interval_label(interval_label,
                                                             start, end)
    resample_interpolate_slicer = partial(forecast.reindex_fill_slice,
                                          freq='5min', start_slice=start_adj,
                                          end_slice=end_adj)
    ghi, dni, dhi, air_temperature, wind_speed = [
        resample_interpolate_slicer(v) for v in
        (ghi, dni, dhi, air_temperature, wind_speed)
    ]
    # weather (and optionally power) will eventually be resampled
    # to hourly average using resampler defined below
    label = datamodel.CLOSED_MAPPING[interval_label]
    resampler = partial(forecast.resample, freq='1h', label=label)
    solar_pos_calculator = partial(
        pvmodel.calculate_solar_position, latitude, longitude, elevation,
        ghi.index)
    return (ghi, dni, dhi, air_temperature, wind_speed,
            resampler, solar_pos_calculator)


def rap_ghi_to_instantaneous(latitude, longitude, elevation,
                             init_time, start, end, interval_label,
                             load_forecast=load_forecast,
                             *, __model='rap'):
    """
    Hourly instantantaneous RAP forecast.
    GHI directly from NWP model. DNI, DHI computed.
    Max forecast horizon 21 or 39 (3Z, 9Z, 15Z, 21Z) hours.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this point.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this point.
    interval_label : str
        Must be instant
    """
    # ghi dni and dhi not in RAP output available from g2sub service
    ghi, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, __model,
        variables=('ghi', 'air_temperature', 'wind_speed'))
    dni, dhi, solar_pos_calculator = _ghi_to_dni_dhi(
        latitude, longitude, elevation, ghi)
    # hourly instant in, hourly instant out
    resampler = partial(forecast.resample, freq='1h')
    return (ghi, dni, dhi, air_temperature, wind_speed,
            resampler, solar_pos_calculator)


def rap_cloud_cover_to_hourly_mean(latitude, longitude, elevation,
                                   init_time, start, end, interval_label,
                                   load_forecast=load_forecast,
                                   *, __model='rap'):
    """
    Take hourly RAP instantantaneous cloud cover and convert it to
    hourly average forecasts.
    GHI from NWP model cloud cover. DNI, DHI computed.
    Max forecast horizon 21 or 39 (3Z, 9Z, 15Z, 21Z) hours.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* and exclusive of this instant if
        interval_label is *ending*.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this instant if
        interval_label is *beginning* and inclusive of this instant if
        interval_label is *ending*.
    interval_label : str
        Must be *beginning* or *ending*
    """
    cloud_cover, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, __model,
        variables=('cloud_cover', 'air_temperature', 'wind_speed'))
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, air_temperature,
                                       wind_speed, start, end, interval_label,
                                       'interpolate')


def gfs_quarter_deg_3hour_to_hourly_mean(latitude, longitude, elevation,
                                         init_time, start, end, interval_label,
                                         load_forecast=load_forecast,
                                         *, __model='gfs_3h'):
    """
    Take 3 hr GFS and convert it to hourly average data.
    GHI from NWP model cloud cover. DNI, DHI computed.
    Max forecast horizon 240 hours.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* and exclusive of this instant if
        interval_label is *ending*.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this instant if
        interval_label is *beginning* and inclusive of this instant if
        interval_label is *ending*.
    interval_label : str
        Must be *beginning* or *ending*
    """
    start_floored, end_ceil = _adjust_gfs_start_end(start, end)
    cloud_cover_mixed, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start_floored, end_ceil, __model,
        variables=('cloud_cover', 'air_temperature', 'wind_speed'))
    cloud_cover = forecast.unmix_intervals(cloud_cover_mixed)
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, air_temperature,
                                       wind_speed, start, end, interval_label,
                                       'bfill')


def gfs_quarter_deg_hourly_to_hourly_mean(latitude, longitude, elevation,
                                          init_time, start, end,
                                          interval_label,
                                          load_forecast=load_forecast,
                                          *, __model='gfs_0p25'):
    """
    Take 1 hr GFS and convert it to hourly average data.
    GHI from NWP model cloud cover. DNI, DHI computed.
    Max forecast horizon 120 hours.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* and exclusive of this instant if
        interval_label is *ending*.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this instant if
        interval_label is *beginning* and inclusive of this instant if
        interval_label is *ending*.
    interval_label : str
        Must be *beginning* or *ending*
    """
    start_floored, end_ceil = _adjust_gfs_start_end(start, end)
    cloud_cover_mixed, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start_floored, end_ceil, __model,
        variables=('cloud_cover', 'air_temperature', 'wind_speed'))
    cloud_cover = forecast.unmix_intervals(cloud_cover_mixed)
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, air_temperature,
                                       wind_speed, start, end, interval_label,
                                       'bfill')


def gfs_quarter_deg_to_hourly_mean(latitude, longitude, elevation,
                                   init_time, start, end, interval_label,
                                   load_forecast=load_forecast,
                                   *, __model='gfs_0p25'):
    """
    Hourly average forecasts derived from GFS 1, 3, and 12 hr frequency
    output. GHI from NWP model cloud cover. DNI, DHI computed.
    Max forecast horizon 384 hours.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* and exclusive of this instant if
        interval_label is *ending*.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this instant if
        interval_label is *beginning* and inclusive of this instant if
        interval_label is *ending*.
    interval_label : str
        Must be *beginning* or *ending*
    """
    start_floored, end_ceil = _adjust_gfs_start_end(start, end)
    cloud_cover_mixed, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start_floored, end_ceil, __model,
        variables=('cloud_cover', 'air_temperature', 'wind_speed'))
    cloud_cover = _unmix_various_gfs_intervals(
        init_time, start_floored, end_ceil, cloud_cover_mixed)
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, air_temperature,
                                       wind_speed, start, end, interval_label,
                                       'bfill')


def _adjust_gfs_start_end(start, end):
    """
    Adjusts the GFS start and end times so that we always load a full
    period of the mixed intervals average cycle.
    """
    start_floored = start.floor('6h') + pd.Timedelta('1h')
    if start_floored > start:
        start_floored -= pd.Timedelta('6h')
    end_ceil = end.ceil('6h')
    return start_floored, end_ceil


def _unmix_various_gfs_intervals(init_time, start_floored, end_ceil,
                                 cloud_cover_mixed):
    """unmix intervals for each kind of interval length in GFS forecast"""
    end_1h = init_time + pd.Timedelta('120hr')
    end_3h = init_time + pd.Timedelta('240hr')

    cloud_covers = []

    if start_floored < end_1h:
        cloud_cover_1h_mixed = cloud_cover_mixed.loc[start_floored:end_1h]
        cloud_covers.append(forecast.unmix_intervals(cloud_cover_1h_mixed))

    if end_ceil > end_1h and start_floored < end_3h:
        cloud_cover_3h_mixed = cloud_cover_mixed.loc[
            end_1h+pd.Timedelta('3hr'):end_3h]
        cloud_covers.append(forecast.unmix_intervals(cloud_cover_3h_mixed))

    if end_ceil > end_3h:
        cloud_cover_12h = cloud_cover_mixed.loc[
            end_3h+pd.Timedelta('12hr'):end_ceil]
        cloud_covers.append(cloud_cover_12h)

    cloud_cover = pd.concat(cloud_covers)
    return cloud_cover


def gefs_half_deg_to_hourly_mean(latitude, longitude, elevation,
                                 init_time, start, end, interval_label,
                                 load_forecast=load_forecast,
                                 *, __model='gefs'):
    """
    Hourly average forecasts derived from GEFS 3, 6, and
    12 hr frequency output. GHI from NWP model cloud cover. DNI, DHI
    computed. Max forecast horizon 384 hours.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* and exclusive of this instant if
        interval_label is *ending*.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this instant if
        interval_label is *beginning* and inclusive of this instant if
        interval_label is *ending*.
    interval_label : str
        Must be *beginning* or *ending*

    Returns
    -------
    The columns of the DataFrames correspond to the GEFS members.

    ghi : pd.DataFrame
    dni : pd.DataFrame
    dhi : pd.DataFrame
    air_temperature : pd.DataFrame
    wind_speed : pd.DataFrame
    resample_sort : function
        Resamples, then sorts the above DataFrames.
    solar_position_calculator : function

    Notes
    -----
    Returned values are hourly averages sorted from smallest to largest
    at each time stamp. Each variable is sorted independently.
    This describes a ProbabilisticForecast with ``axis='y'`` and
    ``constant_values=[0, 5, ...95, 100]``.
    """
    start_floored, end_ceil = _adjust_gfs_start_end(start, end)

    def _load_gefs_member(member, solar_position):
        cloud_cover_mixed, air_temperature, wind_speed = load_forecast(
            latitude, longitude, init_time, start_floored, end_ceil, member,
            variables=('cloud_cover', 'air_temperature', 'wind_speed'))
        cloud_cover = _unmix_various_gefs_intervals(
            init_time, start_floored, end_ceil, cloud_cover_mixed)
        return _resample_using_cloud_cover(
            latitude, longitude, elevation, cloud_cover, air_temperature,
            wind_speed, start, end, interval_label, 'bfill',
            solar_position=solar_position)

    # load and process control forecast, then load and process
    # permutations. for efficiency, use control's solar position.
    # accumulate results in dicts so they can be easily converted to
    # DataFrames.
    ghi, dni, dhi, air_temperature, wind_speed, resampler, sol_pos_calc = \
        _load_gefs_member('gefs_c00', None)
    solar_position = sol_pos_calc()
    ghi_ens = {'c00': ghi}
    dni_ens = {'c00': dni}
    dhi_ens = {'c00': dhi}
    air_temperature_ens = {'c00': air_temperature}
    wind_speed_ens = {'c00': wind_speed}
    for member in range(1, 21):
        key = f'p{member:02d}'
        ghi, dni, dhi, air_temperature, wind_speed, _, _ = \
            _load_gefs_member(f'gefs_{key}', solar_position)
        ghi_ens[key] = ghi
        dni_ens[key] = dni
        dhi_ens[key] = dhi
        air_temperature_ens[key] = air_temperature
        wind_speed_ens[key] = wind_speed

    ghi_ens = pd.DataFrame(ghi_ens)
    dni_ens = pd.DataFrame(dni_ens)
    dhi_ens = pd.DataFrame(dhi_ens)
    air_temperature_ens = pd.DataFrame(air_temperature_ens)
    wind_speed_ens = pd.DataFrame(wind_speed_ens)

    def resample_sort(fx):
        resampled = resampler(fx)
        sorted_ = forecast.sort_gefs_frame(resampled)
        return sorted_

    return (ghi_ens, dni_ens, dhi_ens, air_temperature_ens, wind_speed_ens,
            resample_sort, sol_pos_calc)


def _unmix_various_gefs_intervals(init_time, start_floored, end_ceil,
                                  cloud_cover_mixed):
    """unmix intervals for each kind of interval length in GEFS forecast"""
    # 195hr is only in index for post Sept 2020 GEFS
    if init_time + pd.Timedelta('195hr') in cloud_cover_mixed.index:
        end_3h = init_time + pd.Timedelta('240hr')
    else:
        end_3h = init_time + pd.Timedelta('192hr')

    cloud_covers = []

    if start_floored < end_3h:
        cloud_cover_3h_mixed = cloud_cover_mixed.loc[start_floored:end_3h]
        cloud_covers.append(forecast.unmix_intervals(cloud_cover_3h_mixed))

    if end_ceil > end_3h:
        cloud_cover_6_or_12h = cloud_cover_mixed.loc[
            end_3h+pd.Timedelta('6hr'):end_ceil]
        cloud_covers.append(cloud_cover_6_or_12h)

    cloud_cover = pd.concat(cloud_covers)
    return cloud_cover


def nam_12km_hourly_to_hourly_instantaneous(latitude, longitude, elevation,
                                            init_time, start, end,
                                            interval_label,
                                            load_forecast=load_forecast,
                                            *, __model='nam_12km'):
    """
    Hourly instantantaneous forecast.
    GHI directly from NWP model. DNI, DHI computed.
    Max forecast horizon 36 hours.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this point.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this point.
    interval_label : str
        Must be instant
    """
    start_adj, end_adj = adjust_start_end_for_interval_label(
        interval_label, start, end, limit_instant=True)
    ghi, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start_adj, end_adj, __model,
        variables=('ghi', 'air_temperature', 'wind_speed'))
    dni, dhi, solar_pos_calculator = _ghi_to_dni_dhi(
        latitude, longitude, elevation, ghi)
    # hourly instant in, hourly instant out
    resampler = partial(forecast.resample, freq='1h')
    return (ghi, dni, dhi, air_temperature, wind_speed,
            resampler, solar_pos_calculator)


def nam_12km_cloud_cover_to_hourly_mean(latitude, longitude, elevation,
                                        init_time, start, end, interval_label,
                                        load_forecast=load_forecast,
                                        *, __model='nam_12km'):
    """
    Hourly average forecast.
    GHI from NWP model cloud cover. DNI, DHI computed.
    Max forecast horizon 72 hours.

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    init_time : pd.Timestamp
        Full datetime of a model initialization
    start : pd.Timestamp
        Forecast start. Forecast is inclusive of this instant if
        interval_label is *beginning* and exclusive of this instant if
        interval_label is *ending*.
    end : pd.Timestamp
        Forecast end. Forecast is exclusive of this instant if
        interval_label is *beginning* and inclusive of this instant if
        interval_label is *ending*.
    interval_label : str
        Must be *beginning* or *ending*
    """
    cloud_cover, air_temperature, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, __model,
        variables=('cloud_cover', 'air_temperature', 'wind_speed'))
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, air_temperature,
                                       wind_speed, start, end, interval_label,
                                       'interpolate')
