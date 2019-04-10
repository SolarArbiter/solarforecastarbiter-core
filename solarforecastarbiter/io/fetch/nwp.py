"""
Fetch NWP files from NCEP Nomads for select variables. Should primarily be used
as a CLI program.
"""
import asyncio
import argparse
from functools import partial
import logging
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile


import aiohttp
import pandas as pd
import xarray as xr


from solarforecastarbiter.io.fetch import (
    handle_exception, basic_logging_config, make_session,
    run_in_executor, start_cluster, abort_all_on_exception)


logger = logging.getLogger(__name__)


CHECK_URL = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/{}/prod'
BASE_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/'
DOMAIN = {'subregion': '',
          'leftlon': -126,
          'rightlon': -66,
          'toplat': 50,
          'bottomlat': 24}


def _gfs_valid_hr_gen(init_hr):
    i = 0  # should we start at 1 and avoid the analysis?
    while True:
        yield i
        if i < 120:
            i += 1
        elif i >= 120 and i < 240:
            i += 3
        elif i >= 240 and i < 384:
            i += 12
        else:
            break


GFS_0P25_1HR = {'endpoint': 'filter_gfs_0p25_1hr.pl',
                'file': 'gfs.t{init_hr:02d}z.pgrb2.0p25.f{valid_hr:03d}',
                'dir': '/gfs.{init_dt}',
                'lev_2_m_above_ground': 'on',
                'lev_10_m_above_ground': 'on',
                'lev_entire_atmosphere': 'on',
                'lev_surface': 'on',
                'var_DSWRF': 'on',
                'var_TCDC': 'on',
                'var_TMP': 'on',
                'var_UGRD': 'on',
                'var_VGRD': 'on',
                'update_freq': '6h',
                'valid_hr_gen': _gfs_valid_hr_gen,
                'time_between_fcst_hrs': 60,
                'delay_to_first_forecast': '200min',
                'avg_max_run_length': '100min'}


def _nam_valid_hr_gen(init_hr):
    i = 0
    while True:
        yield i
        if i < 36:
            i += 1
        elif i >= 36 and i < 84:
            i += 3
        else:
            break


NAM_CONUS = {'endpoint': 'filter_nam.pl',
             'file': 'nam.t{init_hr:02d}z.awphys{valid_hr:02d}.tm00.grib2',
             'dir': '/nam.{init_date}',
             'lev_2_m_above_ground': 'on',
             'lev_10_m_above_ground': 'on',
             'lev_entire_atmosphere_\\(considered_as_a_single_layer\\)': 'on',
             'lev_surface': 'on',
             'var_DSWRF': 'on',
             'var_TCDC': 'on',
             'var_TMP': 'on',
             'var_UGRD': 'on',
             'var_VGRD': 'on',
             'update_freq': '6h',
             'valid_hr_gen': _nam_valid_hr_gen,
             'time_between_fcst_hrs': 60,
             'delay_to_first_forecast': '90min',
             'avg_max_run_length': '80min'}


# should be able to use RANGE requests and get data directly from grib files
# like https://www.cpc.ncep.noaa.gov/products/wesley/fast_downloading_grib.html
# so we can get DSWRF for RAP
RAP = {'endpoint': 'filter_rap.pl',
       'file': 'rap.t{init_hr:02d}z.awp130pgrbf{valid_hr:02d}.grib2',
       'dir': '/rap.{init_date}',
       'lev_2_m_above_ground': 'on',
       'lev_10_m_above_ground': 'on',
       'lev_entire_atmosphere': 'on',
       'lev_surface': 'on',
       'var_TCDC': 'on',
       'var_TMP': 'on',
       'var_UGRD': 'on',
       'var_VGRD': 'on',
       'update_freq': '1h',
       'valid_hr_gen': (
           lambda x: range(40) if x in (3, 9, 15, 21) else range(22)),
       'time_between_fcst_hrs': 60,
       'delay_to_first_forecast': '50min',
       'avg_max_run_length': '30min'}


HRRR_HOURLY = {
    'endpoint': 'filter_hrrr_2d.pl',
    'file': 'hrrr.t{init_hr:02d}z.wrfsfcf{valid_hr:02d}.grib2',
    'dir': '/hrrr.{init_date}/conus',
    'lev_2_m_above_ground': 'on',
    'lev_10_m_above_ground': 'on',
    'lev_entire_atmosphere': 'on',
    'lev_surface': 'on',
    'var_DSWRF': 'on',
    'var_VBDSF': 'on',
    'var_VDDSF': 'on',
    'var_TCDC': 'on',
    'var_TMP': 'on',
    'var_UGRD': 'on',
    'var_VGRD': 'on',
    'update_freq': '1h',
    'valid_hr_gen': (
        lambda x: range(37) if x in (0, 6, 12, 18) else range(19)),
    'time_between_fcst_hrs': 120,
    'delay_to_first_forecast': '45min',
    'avg_max_run_length': '70min'}


HRRR_SUBHOURLY = {
    'endpoint': 'filter_hrrr_sub.pl',
    'file': 'hrrr.t{init_hr:02d}z.wrfsubhf{valid_hr:02d}.grib2',
    'dir': '/hrrr.{init_date}/conus',
    'lev_2_m_above_ground': 'on',
    'lev_10_m_above_ground': 'on',
    'lev_entire_atmosphere': 'on',
    'lev_surface': 'on',
    'var_DSWRF': 'on',
    'var_VBDSF': 'on',
    'var_VDDSF': 'on',
    'var_TMP': 'on',
    'var_UGRD': 'on',
    'var_VGRD': 'on',
    'update_freq': '1h',
    'valid_hr_gen': (lambda x: range(19)),
    'time_between_fcst_hrs': 120,
    'delay_to_first_forecast': '45min',
    'avg_max_run_length': '50min'}

EXTRA_KEYS = ['update_freq', 'valid_hr_gen', 'time_between_fcst_hrs',
              'delay_to_first_forecast', 'avg_max_run_length']

model_map = {'gfs_0p25': GFS_0P25_1HR, 'nam_12km': NAM_CONUS,
             'rap': RAP, 'hrrr_hourly': HRRR_HOURLY,
             'hrrr_subhourly': HRRR_SUBHOURLY}


GRIB_TO_NC4 = """
#/usr/bin/bash
FOLDER=$1
NCFILENAME=$2

pushd $FOLDER
pwd

# make the nc_table
cat <<EOF > nc.tbl
TMP:surface:ignore
TMP:2 m above ground:t2m
UGRD:10 m above ground:ignore
VGRD:10 m above ground:ignore
TCDC:entire atmosphere:tcdc
TCDC:entire atmosphere (considered as a single layer):tcdc
DSWRF:surface:dswrf
VBDSF:surface:vbdsf
VDDSF:surface:vddsf
WIND:10 m above ground:si10
EOF


# add wind speed to grib files
for file in $(ls -1 *.grib2); do
    wgrib2 $file -wind_speed - -match "(UGRD|VGRD)" | \
        wgrib2 - -append -grib_out $file;
    done;

# make netcdf
cat *.grib2 | wgrib2 - -nc_table nc.tbl {} -append -netcdf $NCFILENAME
rm nc.tbl
popd
"""

COMPRESSION = {'zlib': True, 'complevel': 1, 'shuffle': True,
               'fletcher32': True}
DEFAULT_ENCODING = {
    # stores the time steps, not an actual time
    'time': {'dtype': 'int16'},
    'latitude': {'dtype': 'float32', 'least_significant_digit': 3},
    'longitude': {'dtype': 'float32', 'least_significant_digit': 3}
}
LEAST_SIGNIFICANT_DIGITS = {
    't2m': 2,
    'tcdc': 1,
    'si10': 2,
    'dswrf': 1,
    'vbdsf': 1,
    'vddsf': 1
}


async def get_with_retries(get_func, *args, retries=5, **kwargs):
    """
    Call get_func and retry if the request fails

    Params
    ------
    get_func : function
        Function that performs an aiohttp call to be retried
    retries : int
        Number of retries before raising the error
    *args, **kwargs
        Passed to get_func

    Returns
    -------
    Result of get_func

    Raises
    ------
    aiohttp.ClientResponseError
        When get_func fails after retrying retries times
    """
    retried = 0
    while True:
        try:
            res = await get_func(*args, **kwargs)
        except aiohttp.ClientResponseError as e:
            logger.warning('Request to %s failed with code %s, retrying',
                           e.request_info.url, e.status)
            retried += 1
            if retried >= retries:
                raise
            await asyncio.sleep(60)
        else:
            return res


async def get_available_dirs(session, model):
    """Get the available date/date+init_hr directories"""
    simple_model = model['file'].split('.')[0]
    is_init_date = 'init_date' in model['dir']
    model_url = BASE_URL + model['endpoint']

    async def _get(model_url):
        async with session.get(model_url, raise_for_status=True) as r:
            return await r.text()

    page = await get_with_retries(_get, model_url)
    if is_init_date:
        list_avail_days = set(
            re.findall(simple_model + '\\.([0-9]{8})', page))
    else:
        list_avail_days = set(
            re.findall(simple_model + '\\.([0-9]{10})', page))
    return list_avail_days


def _process_params(model, init_time):
    """Generator to get the parameters for fetching forecasts for a given
    model at a given init_time"""
    params = model.copy()
    params.update(DOMAIN)
    valid_hr_gen = params['valid_hr_gen'](init_time.hour)
    for p in EXTRA_KEYS:
        del params[p]
    params['dir'] = params['dir'].format(
        init_date=init_time.strftime('%Y%m%d'),
        init_dt=init_time.strftime('%Y%m%d%H'))
    for i in valid_hr_gen:
        newp = params.copy()
        newp['file'] = newp['file'].format(
            init_hr=init_time.hour,
            valid_hr=i)
        yield newp


async def check_next_inittime(session, init_time, model):
    """Check if data from the next model initializtion time is available"""
    next_inittime = init_time + pd.Timedelta(model['update_freq'])
    simple_model = model['file'].split('.')[0]
    next_init_url = (CHECK_URL.format(simple_model)
                     + model['dir'].format(
                         init_date=next_inittime.strftime('%Y%m%d'),
                         init_dt=next_inittime.strftime('%Y%m%d%H'))
                     + '/' + model['file'].format(init_hr=next_inittime.hour,
                                                  valid_hr=0))

    try:
        async with session.head(next_init_url) as r:
            if r.status == 200:
                logger.warning(
                    'Skipping to next init time at %s for %s',
                    next_inittime, simple_model)
                return True
            else:
                return False
    except aiohttp.ClientOSError:
        return False


async def files_to_retrieve(session, model, init_time):
    """Generator to return the parameters of the available files for download
    """
    possible_params = _process_params(model, init_time)
    simple_model = model['file'].split('.')[0]
    first_file_modified_at = None
    for next_params in possible_params:
        next_model_url = (CHECK_URL.format(simple_model)
                          + next_params['dir'] + '/' + next_params['file'])
        while True:
            # is the next file ready?
            try:
                async with session.head(
                        next_model_url, raise_for_status=True) as r:
                    if first_file_modified_at is None:
                        first_file_modified_at = pd.Timestamp(
                            r.headers['Last-Modified'])
                        logger.debug('First file was available at %s',
                                     first_file_modified_at)
            except (aiohttp.ClientOSError, aiohttp.ClientResponseError):
                logger.debug('Next file not ready yet for %s at %s',
                             simple_model, init_time)
            else:
                logger.debug('%s/%s is ready for download',
                             next_params['dir'], next_params['file'])
                yield next_params
                break

            # if the current time is after 'avg_max_run_length' after the
            # first forecast was available, check if forecasts from the
            # next model run are available and if so, move on to that run
            if (
                    first_file_modified_at is not None and
                    pd.Timestamp.utcnow() > first_file_modified_at +
                    pd.Timedelta(model['avg_max_run_length'])
            ):
                nextrun_available = await check_next_inittime(
                    session, init_time, model)
                if nextrun_available:
                    return
            await asyncio.sleep(model['time_between_fcst_hrs'])


async def _get_file(session, url, params, tmpfile, chunksize):
    timeout = aiohttp.ClientTimeout(total=660, connect=60, sock_read=600)
    async with session.get(url, params=params, raise_for_status=True,
                           timeout=timeout) as r:
        with open(tmpfile, 'wb') as f:
            while True:
                chunk = await r.content.read(chunksize * 1024)
                if not chunk:
                    break
                f.write(chunk)


@abort_all_on_exception
async def fetch_grib_files(session, params, basepath, init_time, chunksize):
    """
    Fetch the grib file referenced by params and save to the appropriate
    folder under basepath. Retrieves the files in chunks.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The HTTP session to use to request the file
    params : dict
        Parameters to include in the GET query to params['endpoint']
    basepath : Path
        Path to the base directory where files will be saved. New directories
        under basepath of the form basepath / year / month / day / hour
        will be created as necessary.
    init_time : datetime
        Initialization time of the model we're trying to fetch
    chunksize : int
        Chunksize in KB to fetch and save at once

    Returns
    -------
    filename : Path
        Path of the successfully saved file

    Raises
    ------
    aiohttp.ClientResponseError
        When the HTTP request fails/returns a status code >= 400
    """
    endpoint = params.pop('endpoint')
    url = BASE_URL + endpoint
    filename = (
        basepath / init_time.strftime('%Y/%m/%d/%H') / params['file'])
    if not filename.suffix == '.grib2':
        filename = filename.with_suffix(filename.suffix + '.grib2')
    if filename.exists():
        return filename
    if not filename.parent.is_dir():
        filename.parent.mkdir(parents=True)
    logger.info('Getting file %s', filename)
    tmpfile = filename.with_name('.tmp_' + filename.name)
    await get_with_retries(_get_file, session, url, params, tmpfile, chunksize)
    tmpfile.rename(filename)
    logging.debug('Successfully saved %s', filename)
    return filename


def _process_grib(nctmp, folder, with_avg=False):
    if with_avg:
        # for hrrr subhourly, assume TMP and VDDSF have no average but others
        fmt = "-match 'ave|TMP|VDDSF'"
    else:
        fmt = ''
    with tempfile.NamedTemporaryFile(mode='w') as tmpfile:
        tmpfile.write(GRIB_TO_NC4.format(fmt))
        tmpfile.flush()
        try:
            subprocess.run(
                ['sh', tmpfile.name, str(folder), str(nctmp)],
                check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error('Error converting grib to NetCDF\n%s',
                         e.stderr)
            nctmp.unlink()
            raise


@abort_all_on_exception
async def process_grib_to_netcdf(folder):
    nctmp = Path(tempfile.mkstemp(dir=folder)[1])
    logger.info('Converting GRIB files to NetCDF with wgrib2')
    await run_in_executor(_process_grib, nctmp, folder,
                          'subhourly' in str(folder))
    return nctmp


def _optimize_netcdf(nctmpfile, out_path):
    """Optmizes the netcdf file for accessing by time slice."""
    ds = xr.open_dataset(nctmpfile, engine='netcdf4',
                         backend_kwargs={'mode': 'r'})
    # time is likely unlimited
    if 'unlimited_dims' in ds.encoding:
        ds.encoding = {}

    chunksizes = []
    for dim, size in ds.dims.items():
        if dim == 'time':
            chunksizes.append(size)
        else:
            chunksizes.append(50)

    encoding = DEFAULT_ENCODING.copy()
    encoding.update(
        {key: {'dtype': 'float32',
               'least_significant_digit': LEAST_SIGNIFICANT_DIGITS[key],
               'chunksizes': chunksizes,
               **COMPRESSION}
         for key in ds.keys()})
    ds.to_netcdf(out_path, format='NETCDF4',
                 mode='w', unlimited_dims=None,
                 encoding=encoding)


async def optimize_netcdf(nctmpfile, final_path):
    """Compress the netcdf file and adjust the chunking for fast time-series
    access"""
    logger.info('Optimizing NetCDF file')
    tmp_path = Path(tempfile.mkstemp(dir=final_path.parent)[1])
    # possible that this leaks memory, so run in separate process
    # that is restarted after a number of jobs
    try:
        await run_in_executor(_optimize_netcdf, nctmpfile, tmp_path)
    except Exception:
        tmp_path.unlink()
        raise
    else:
        tmp_path.rename(final_path)
    finally:
        nctmpfile.unlink()


async def sleep_until_inittime(inittime, model):
    # don't bother requesting a file until it might be ready
    now = pd.Timestamp.utcnow()
    likely_ready_time = inittime + pd.Timedelta(
        model['delay_to_first_forecast'])
    if likely_ready_time > now:
        seconds = (likely_ready_time - now).total_seconds()
        logger.info('Sleeping %0.1fs for next model run', seconds)
        await asyncio.sleep(seconds)


async def startup_find_next_runtime(model_path, session, model):
    """Find the next model run to get based on what is available
    on NOMADS and what .nc files are present locally"""
    dirs = await get_available_dirs(session, model)
    no_file = []
    for dir_ in dirs:
        if len(dir_) == 8:
            path = model_path / dir_[:4] / dir_[4:6] / dir_[6:8]
            for hr in range(0, 24, int(model['update_freq'].strip('h'))):
                hrpath = path / f'{hr:02d}'
                glob = list(hrpath.glob('*.nc'))
                if len(glob) == 0:
                    no_file.append(pd.Timestamp(f'{dir_[:8]}T{hr:02d}00Z'))
        else:
            hrpath = model_path / dir_[:4] / dir_[4:6] / dir_[6:8] / dir_[8:10]
            glob = list(hrpath.glob('*.nc'))
            if len(glob) == 0:
                no_file.append(pd.Timestamp(f'{dir_[:8]}T{dir_[8:10]}00Z'))
    if len(no_file) == 0:
        # all files for current dirs present, get next day
        inittime = max(
            [pd.Timestamp(f'{dir_[:8]}T0000Z')]) + pd.Timedelta('1d')
    else:
        inittime = min(no_file)
    await sleep_until_inittime(inittime, model)
    return inittime


async def next_run_time(inittime, modelpath, model):
    inittime += pd.Timedelta(model['update_freq'])
    # check if nc file exists for this inittime
    if len(list(
            (modelpath / inittime.strftime('%Y/%m/%d/%H')).glob('*.nc'))) > 0:
        return await next_run_time(inittime, modelpath, model)
    await sleep_until_inittime(inittime, model)
    return inittime


async def run(basepath, model_name, chunksize, once=False):
    session = make_session()
    modelpath = basepath / model_name
    model = model_map[model_name]
    inittime = await startup_find_next_runtime(modelpath, session, model)
    while True:
        fetch_tasks = set()
        async for params in files_to_retrieve(session, model, inittime):
            fetch_tasks.add(asyncio.create_task(
                fetch_grib_files(session, params, modelpath, inittime,
                                 chunksize)))
        files = await asyncio.gather(*fetch_tasks)
        if len(files) != 0:  # skip to next inittime
            path_to_files = files[0].parent
            nctmpfile = await process_grib_to_netcdf(path_to_files)
            try:
                await optimize_netcdf(
                    nctmpfile, path_to_files / f'{model_name}.nc')
            except Exception:
                raise
            else:
                # remove grib files
                for f in files:
                    f.unlink()
        if once:
            break
        else:
            logger.info('Moving on to next model run')
            inittime = await next_run_time(inittime, modelpath, model)
    await session.close()


async def optimize_only(path_to_files, model_name):
    nctmpfile = await process_grib_to_netcdf(path_to_files)
    try:
        await optimize_netcdf(
            nctmpfile, path_to_files / f'{model_name}.nc')
    except Exception:
        raise
    else:
        # remove grib files
        for f in path_to_files.glob('*.grib2'):
            f.unlink()


def check_wgrib2():
    if shutil.which('wgrib2') is None:
        logger.error('wgrib2 was not found in PATH and is required')
        sys.exit(1)


def main():
    sys.excepthook = handle_exception
    basic_logging_config()
    check_wgrib2()

    argparser = argparse.ArgumentParser(
        description=('Retrieve weather forecasts with variables relevant to '
                     'solar power from the NCEP NOMADS server. The utility '
                     'function  wgrib2 is required to convert these forecasts '
                     'into netCDF format.'))
    argparser.add_argument('-v', '--verbose', action='count')
    argparser.add_argument('--chunksize', default=128,
                           help='Size of a chunk (in KB) to save at one time')
    argparser.add_argument('--once', action='store_true',
                           help='Only get one forecast initialization time')
    argparser.add_argument(
        '--netcdf-only', action='store_true',
        help='Only convert files at save_directory to netcdf')
    argparser.add_argument('save_directory',
                           help='Directory to save data in')
    argparser.add_argument(
        'model', choices=['gfs_0p25', 'nam_12km', 'rap', 'hrrr_hourly',
                          'hrrr_subhourly'],
        help='The model to get data for')
    args = argparser.parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    start_cluster()
    basepath = Path(args.save_directory).resolve()
    if args.netcdf_only:
        path_to_files = basepath
        if (
                not path_to_files.is_dir() or
                len(list(path_to_files.glob('*.grib2'))) == 0
        ):
            logger.error('%s is not a valid directory with grib files',
                         path_to_files)
            sys.exit(1)
        fut = asyncio.ensure_future(optimize_only(path_to_files, args.model))
    else:
        fut = asyncio.ensure_future(run(basepath, args.model, args.chunksize,
                                        args.once))

    loop = asyncio.get_event_loop()

    def bail(ecode):
        fut.cancel()
        sys.exit(ecode)

    loop.add_signal_handler(signal.SIGUSR1, partial(bail, 1))
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, partial(bail, 0))

    loop.run_until_complete(fut)


if __name__ == '__main__':
    main()
