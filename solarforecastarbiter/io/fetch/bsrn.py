"""Parse data from the BSRN, fetch data from NASA LARC.

Modified from pvlib python pvlib/iotools/bsrn.py.
See LICENSES/PVLIB-PYTHON_LICENSE
"""
from io import StringIO
import logging

import pandas as pd
import requests

logger = logging.getLogger('bsrn')

COL_SPECS = [(0, 3), (4, 9), (10, 16), (16, 22), (22, 27), (27, 32), (32, 39),
             (39, 45), (45, 50), (50, 55), (55, 64), (64, 70), (70, 75)]

BSRN_COLUMNS = ['day', 'minute',
                'ghi', 'ghi_std', 'ghi_min', 'ghi_max',
                'dni', 'dni_std', 'dni_min', 'dni_max',
                'empty', 'empty', 'empty', 'empty', 'empty',
                'dhi', 'dhi_std', 'dhi_min', 'dhi_max',
                'lwd', 'lwd_std', 'lwd_min', 'lwd_max',
                'temp_air', 'relative_humidity', 'pressure']


def parse_bsrn(fbuf):
    """
    Parse a buffered BSRN station-to-archive file into a DataFrame.

    The BSRN (Baseline Surface Radiation Network) is a world wide network
    of high-quality solar radiation monitoring stations as described in [1]_.
    The function only parses the basic measurements (LR0100), which include
    global, diffuse, direct and downwelling long-wave radiation [2]_. Future
    updates may include parsing of additional data and meta-data.

    BSRN files are freely available and can be accessed via FTP [3]_. Required

    username and password are easily obtainable as described in the BSRN's
    Data Release Guidelines [4]_.

    Parameters
    ----------
    fbuf: io.StringIO
        A buffer containing the data to be parsed.

    Returns
    -------
    data: DataFrame
        A DataFrame with the columns as described below. For more extensive
        description of the variables, consult [2]_.

    Notes
    -----
    The data DataFrame includes the following fields:

    =======================  ======  ==========================================
    Key                      Format  Description
    =======================  ======  ==========================================
    day                      int     Day of the month 1-31
    minute                   int     Minute of the day 0-1439
    ghi                      float   Mean global horizontal irradiance [W/m^2]
    ghi_std                  float   Std. global horizontal irradiance [W/m^2]
    ghi_min                  float   Min. global horizontal irradiance [W/m^2]
    ghi_max                  float   Max. global horizontal irradiance [W/m^2]
    dni                      float   Mean direct normal irradiance [W/m^2]
    dni_std                  float   Std. direct normal irradiance [W/m^2]
    dni_min                  float   Min. direct normal irradiance [W/m^2]
    dni_max                  float   Max. direct normal irradiance [W/m^2]
    dhi                      float   Mean diffuse horizontal irradiance [W/m^2]
    dhi_std                  float   Std. diffuse horizontal irradiance [W/m^2]
    dhi_min                  float   Min. diffuse horizontal irradiance [W/m^2]
    dhi_max                  float   Max. diffuse horizontal irradiance [W/m^2]
    lwd                      float   Mean. downward long-wave radiation [W/m^2]
    lwd_std                  float   Std. downward long-wave radiation [W/m^2]
    lwd_min                  float   Min. downward long-wave radiation [W/m^2]
    lwd_max                  float   Max. downward long-wave radiation [W/m^2]
    temp_air                 float   Air temperature [°C]
    relative_humidity        float   Relative humidity [%]
    pressure                 float   Atmospheric pressure [hPa]
    =======================  ======  ==========================================

    References
    ----------
    .. [1] `World Radiation Monitoring Center - Baseline Surface Radiation
        Network (BSRN)
        <https://bsrn.awi.de/>`_
    .. [2] `Update of the Technical Plan for BSRN Data Management, 2013,
       Global Climate Observing System (GCOS) GCOS-172.
       <https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/gcos-174.pdf>`_
    .. [3] `BSRN Data Retrieval via FTP
       <https://bsrn.awi.de/data/data-retrieval-via-ftp/>`_
    .. [4] `BSRN Data Release Guidelines
       <https://bsrn.awi.de/data/conditions-of-data-release/>`_
    """

    # Read file and store the starting line number for each logical record (LR)
    line_no_dict = {}

    fbuf.readline()  # first line should be *U0001, so read it and discard
    line_no_dict['0001'] = 0
    date_line = fbuf.readline()  # second line contains the year and month
    start_date = pd.Timestamp(year=int(date_line[7:11]),
                              month=int(date_line[3:6]), day=1,
                              tz='UTC')  # BSRN timestamps are UTC
    for num, line in enumerate(fbuf, start=2):
        if line.startswith('*'):  # Find start of all logical records
            line_no_dict[line[2:6]] = num  # key is 4 digit LR number

    fbuf.seek(0)  # reset buffer to start of data

    # Determine start and end line of logical record LR0100 to be parsed
    start_row = line_no_dict['0100'] + 1  # Start line number
    # If LR0100 is the last logical record, then read rest of file
    if start_row-1 == max(line_no_dict.values()):
        end_row = num  # then parse rest of the file
    else:  # otherwise parse until the beginning of the next logical record
        end_row = min([i for i in line_no_dict.values() if i > start_row]) - 1
    nrows = end_row-start_row+1

    # Read file as a fixed width file (fwf)
    data = pd.read_fwf(fbuf, skiprows=start_row, nrows=nrows, header=None,
                       colspecs=COL_SPECS, na_values=[-999.0, -99.9],
                       compression='infer')

    # Create multi-index and unstack, resulting in one column for each variable
    data = data.set_index([data.index // 2, data.index % 2])
    data = data.unstack(level=1).swaplevel(i=0, j=1, axis='columns')

    # Sort columns to match original order and assign column names
    data = data.reindex(sorted(data.columns), axis='columns')
    data.columns = BSRN_COLUMNS
    # Drop empty columns
    data = data.drop('empty', axis='columns')

    # Change day and minute type to integer
    data['day'] = data['day'].astype('Int64')
    data['minute'] = data['minute'].astype('Int64')

    # Set datetime index
    data.index = (start_date
                  + pd.to_timedelta(data['day']-1, unit='d')
                  + pd.to_timedelta(data['minute'], unit='T'))

    return data


def read_bsrn_from_nasa_larc(start, end):
    """Read a range of BRSN monthly data from the NASA LARC.

    Parameters
    ----------
    start: pandas.Timestamp
    end: pandas.Timestamp

    Returns
    -------
    bsrn_data: pd.DataFrame
    """
    # data not available until month is complete, so avoid requesting file
    # that does not exist. assumes file is available as soon as month is
    # complete.
    end_of_last_month = (
        pd.Timestamp.utcnow().normalize() - pd.offsets.MonthBegin()
        - pd.Timedelta('1s'))
    range_end = min(end, end_of_last_month)
    # use period_range to avoid this funky date_range behavior:
    # > pd.date_range(start='2020-01-01', end='2020-01-30 23:59:59', freq='M')
    # DatetimeIndex([], dtype='datetime64[ns]', freq='M')
    # > pd.date_range(start='2020-01-01', end='2020-01-31 00:00:00', freq='M')
    # DatetimeIndex(['2020-01-31'], dtype='datetime64[ns]', freq='M')
    months = pd.period_range(start=start, end=range_end, freq='M')
    # a better programmer would use asyncio
    month_data = []
    for month in months:
        try:
            d = read_bsrn_month_from_nasa_larc(month.year, month.month)
        except Exception as e:
            logger.warning('could not get bsrn data from nasa larc for '
                           f'{month.year}, {month.month}. {e}')
        else:
            month_data.append(d)
    # concat raises exception on empty list. maybe better to let that bubble up
    if len(month_data):
        bsrn_data = pd.concat(month_data)
        return bsrn_data[start:end]
    else:
        # not sure how we get here in practice
        return pd.DataFrame()  # pragma: no cover


def read_bsrn_month_from_nasa_larc(year, month):
    """Read one month of BSRN data from the NASA LARC.

    Parameters
    ----------
    year: int, str
        The year of the data.
    month: int, str
        The month of the data (1 - 12).

    Returns
    -------
    bsrn_data: pd.DataFrame

    Notes
    -----
    Data starts in December, 2014.
    """
    base_url = 'https://cove.larc.nasa.gov/BSRN/LRC49/'
    year = str(year)
    url = f'{base_url}{year}/lrc{int(month):02}{year[2:]}.dat'
    r = requests.get(url)
    r.raise_for_status()
    with StringIO(r.text) as buf:
        return parse_bsrn(buf)
