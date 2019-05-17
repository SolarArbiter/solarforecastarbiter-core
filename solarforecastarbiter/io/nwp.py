from pathlib import Path

import numpy as np
import xarray as xr

BASE_PATH = ''

CF_MAPPING = {
    't2m': 'temp_air',
    'si10': 'wind_speed',
    'dswrf': 'ghi',
    'vbdsf': 'dni',
    'vddsf': 'dhi',
    'tcdc': 'cloud_cover'
}


def load_forecast(latitude, longitude, init_time, start, end, model,
                  variables=('ghi', 'dni', 'dhi', 'temp_air', 'wind_speed'),
                  base_path=BASE_PATH):
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

    variables : list of str
        The variables to load.

    Returns
    -------
    By default:

      * ghi : pd.Series
      * dni : pd.Series
      * dhi : pd.Series
      * temp_air : pd.Series
      * wind_speed : pd.Series

    Raises
    ------
    ValueError : Raised if the requested variable is not found.
    """

    filepath = (Path(base_path) / model /
                init_time.strftime('%Y/%m/%d/%H') / (model + '.nc'))

    mapping_subset = {k: v for k, v in CF_MAPPING.items() if v in variables}

    with xr.open_dataset(filepath) as ds:
        # might want to check for points that are (too far) out of the domain
        try:
            pnt = ds.sel(latitude=latitude, longitude=longitude,
                         method='nearest')
        except ValueError:
            iy_min, ix_min = tunnel_fast(ds.latitude, ds.longitude, latitude,
                                         longitude)
            pnt = ds.sel(y=iy_min, x=ix_min, method='nearest')
        pnt = pnt.rename(mapping_subset)
        return [pnt[variable].to_series() for variable in variables]


def tunnel_fast(latvar, lonvar, lat0, lon0):
    """
    Find closest point in a set of (lat, lon) points to specified point.

    Returns iy,ix such that the square of the tunnel distance
    between (latval[it,ix],lonval[iy,ix]) and (lat0,lon0)
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

    Returns
    -------
    iy_min : int
        Index of point in latitude array that minimizes distance to
        desired point.
    ix_min : int
        Index of point in longitude array that minimizes distance to
        desired point.

    References
    ----------
    https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
    """
    rad_factor = np.pi/180.0
    latvals = latvar * rad_factor
    lonvals = lonvar * rad_factor
    ny, nx = latvals.shape
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    clat, clon = np.cos(latvals), np.cos(lonvals)
    slat, slon = np.sin(latvals), np.sin(lonvals)
    delX = np.cos(lat0_rad)*np.cos(lon0_rad) - clat*clon
    delY = np.cos(lat0_rad)*np.sin(lon0_rad) - clat*slon
    delZ = np.sin(lat0_rad) - slat
    dist_sq = delX**2 + delY**2 + delZ**2
    minindex_1d = dist_sq.argmin()  # 1D index of minimum element
    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
    return iy_min, ix_min
