from pathlib import Path


import numpy as np
import xarray as xr


BASE_PATH = ''


def set_base_path(new_path):
    global BASE_PATH
    if new_path is not None:
        BASE_PATH = new_path


CF_MAPPING = {
    't2m': 'air_temperature',
    'si10': 'wind_speed',
    'dswrf': 'ghi',
    'vbdsf': 'dni',
    'vddsf': 'dhi',
    'tcdc': 'cloud_cover'
}


def _load_pnt(ds, latitude, longitude, limit):
    lats = ds.latitude.values
    lons = ds.longitude.values
    if lats.ndim == 1:
        # ds uses lat and lon as primary coordinates. We still want to use
        # tunnel_fast to determine distance so that we can check that
        # the query point is not too far out of the domain.
        # tunnel_fast requires 2D grid
        lons, lats = np.meshgrid(lons, lats)
    iy_min, ix_min = tunnel_fast(lats, lons, latitude, longitude,
                                 limit=limit)
    # could avoid this if statement if we only use positional indexing
    # like ds[iy_min, ix_min] but this seems safer
    if ds.latitude.ndim == 1:
        pnt = ds.isel(latitude=iy_min, longitude=ix_min)
    else:
        pnt = ds.isel(y=iy_min, x=ix_min)
    return pnt


def load_forecast(
        latitude, longitude, init_time, start, end, model,
        variables=('ghi', 'dni', 'dhi', 'air_temperature', 'wind_speed'),
        base_path=None):
    """Load NWP model data.

    Parameters
    ----------
    latitude : float

    longitude : float

    init_time : pd.Timestamp
        Full datetime of a model initialization

    start : pd.Timestamp

    end : pd.Timestamp

    model : str
        Name of model. Must be one of:

            * 'hrrr_hourly'
            * 'hrrr_subhourly'
            * 'gfs_0p25'
            * 'rap'
            * 'nam_12km'
            * 'gefs_p{num}' where num is '01' to '20', 'gefs_c00',
              'gefs_avg', or 'gefs_spr'

    variables : list of str
        The variables to load.

    Returns
    -------
    By default:

      * ghi : pd.Series
      * dni : pd.Series
      * dhi : pd.Series
      * air_temperature : pd.Series
      * wind_speed : pd.Series

    Raises
    ------
    ValueError : Raised if the requested variable is not found.
    """
    base_path = base_path if base_path is not None else BASE_PATH
    if 'gefs' in model:
        # account for slightly different file layout for gefs
        model_path = 'gefs'
    else:
        model_path = model
    filepath = (Path(base_path) / model_path /
                init_time.strftime('%Y/%m/%d/%H') / (model + '.nc'))
    if not filepath.is_file():
        raise FileNotFoundError(f'{filepath} does not exist')
    mapping_subset = {k: v for k, v in CF_MAPPING.items() if v in variables}

    limit = 500  # maximum distance from point to closest grid point
    # Time in file are stored unlocalized but are UTC
    # so convert localized start/end to UTC then drop tz
    if start.tzinfo is not None:
        start = start.tz_convert('UTC').tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert('UTC').tz_localize(None)
    with xr.open_dataset(filepath) as ds:
        pnt = _load_pnt(ds, latitude, longitude, limit)
        pnt = pnt.sel(time=slice(start, end))
        pnt = pnt.rename(mapping_subset)
        pnt['air_temperature'] -= 273.15  # convert Kelvin to deg C
        series = [pnt[variable].to_series().tz_localize('UTC')
                  for variable in variables]
        return series


def tunnel_fast(latvar, lonvar, lat0, lon0, limit=None):
    """
    Find closest point in a set of (lat, lon) points to specified point.

    Returns iy, ix such that the square of the tunnel distance
    between (latval[it, ix], lonval[iy, ix]) and (lat0, lon0)
    is minimum.

    Parameters
    ----------
    latvar : numpy.array
        Model latitudes.
    lonvar : numpy.array
        Model longitudes.
    lat0 : float
        Latitude of desired point.
    lon0 : float
        Longitude of desired point.
    limit : None or float
        Maximum distance allowed in units of km.

    Returns
    -------
    iy : int
        Index of point in latitude array that minimizes distance to
        desired point.
    ix : int
        Index of point in longitude array that minimizes distance to
        desired point.

    Raises
    ------
    ValueError
        If closest point exceeds minimum distance requirement.

    References
    ----------
    https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
    """
    rad_factor = np.pi/180.0
    latvals = latvar * rad_factor
    lonvals = lonvar * rad_factor
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    clat, clon = np.cos(latvals), np.cos(lonvals)
    slat, slon = np.sin(latvals), np.sin(lonvals)
    delX = np.cos(lat0_rad)*np.cos(lon0_rad) - clat*clon
    delY = np.cos(lat0_rad)*np.sin(lon0_rad) - clat*slon
    delZ = np.sin(lat0_rad) - slat
    dist_sq = delX**2 + delY**2 + delZ**2
    if limit is not None:
        dist = 6378.1 * np.sqrt(dist_sq.min())
        if dist > limit:
            raise ValueError('Maximum distance limit exceeded')
    minindex_1d = dist_sq.argmin()  # 1D index of minimum element
    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
    return iy_min, ix_min
