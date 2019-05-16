from pathlib import Path

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
        pnt = ds.sel(latitude=latitude, longitude=longitude, method='nearest')
        pnt = pnt.rename(mapping_subset)
        return [pnt[variable] for variable in variables]

    # with nc4.Dataset(filepath, mode='r') as store:
    #     for variable in variables:
    #         store.variables[variable][0]
    #     timevar = store.variables['time'][0]
    #     timeindex = pd.to_datetime(np.ma.compressed(timevar), utc=True,
    #         unit='s')
