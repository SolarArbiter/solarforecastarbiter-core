from pathlib import Path

import netCDF4 as nc4

BASE_PATH = ''


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

            * 'hrrr'
            * 'hrrr_subhourly'
            * 'gfs'  # assume 0.25 degree or specify?
            * 'rap'
            * 'nam'  # assume 12 km or specify?

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

    filepath = Path(base_path) / model / init_time.strftime('%Y/%m/%d/%H')
    nc4.Dataset(filepath, mode='r')
