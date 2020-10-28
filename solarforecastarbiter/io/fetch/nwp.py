"""
Fetch NWP files from NCEP NOMADS for select variables. Should primarily be used
as a CLI program.


The script fetches grib2 files from NOMADS as they are available using g2sub,
uses wgrib2 to convert the grib files to netCDF (and adds wind speed),
and optimizes the netCDF file for our expected usage accessing a time-series
for a single location.

Each possible model has an associated dictonary with parameters that are
passed in the HTTP request. Other keys in the dictionaries are used
to set the timing of the HTTP requests:
- update_freq: the time between full model runs
- valid_hr_gen: A generator to produce the valid forecast hours that the model
  produces
- time_between_fcst_hrs: The time in seconds that it typically takes to produce
  the grib2 file for the next forecasted hour
- delay_to_first_forecast: The time after the nominal initialization time
  that the first forecast from the model is typically available
- avg_max_run_length: The average time it takes the model to produce the last
  possible forecast file from the first forecast file.

Many of these parameters are inferred from
https://www.nco.ncep.noaa.gov/pmb/nwprod/prodstat/


This script uses features of asyncio that are likely not available in Windows.
"""
import asyncio
from itertools import chain
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile


import aiohttp
import pandas as pd
import xarray as xr


from solarforecastarbiter.io.fetch import (
    make_session, run_in_executor, abort_all_on_exception)


logger = logging.getLogger(__name__)


CHECK_URL = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/{}/prod'
BASE_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/'
DOMAIN = {'subregion': '',
          'leftlon': -126,
          'rightlon': -66,
          'toplat': 50,
          'bottomlat': 24}


GFS_0P25_1HR = {'endpoint': 'filter_gfs_0p25_1hr.pl',
                'file': 'gfs.t{init_hr:02d}z.pgrb2.0p25.f{valid_hr:03d}',
                'dir': '/gfs.{init_date}/{init_hr}',
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
                'valid_hr_gen': lambda x: chain(range(120), range(120, 240, 3),
                                                range(240, 385, 12)),
                'time_between_fcst_hrs': 60,
                'delay_to_first_forecast': '200min',
                'avg_max_run_length': '100min',
                'filename': 'gfs_0p25.nc'}


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
             'valid_hr_gen': lambda x: chain(range(36), range(36, 85, 3)),
             'time_between_fcst_hrs': 60,
             'delay_to_first_forecast': '90min',
             'avg_max_run_length': '80min',
             'filename': 'nam_12km.nc'}


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
       'avg_max_run_length': '30min',
       'filename': 'rap.nc'}


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
    'avg_max_run_length': '70min',
    'filename': 'hrrr_hourly.nc'}


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
    'var_WIND': 'on',
    'update_freq': '1h',
    'valid_hr_gen': (lambda x: range(19)),
    'time_between_fcst_hrs': 120,
    'delay_to_first_forecast': '45min',
    'avg_max_run_length': '50min',
    'filename': 'hrrr_subhourly.nc'}


# each GEFS stat_or_member is treated separately
# really makes use of async capabilities
GEFS_0P50_RAW = {'endpoint': 'filter_gefs_atmos_0p50a.pl',
                 'file': 'ge{stat_or_member}.t{init_hr:02d}z.pgrb2a.0p50.f{valid_hr:03d}',  # NOQA
                 'dir': '/gefs.{init_date}/{init_hr}/atmos/pgrb2ap5',
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
                 'valid_hr_gen': lambda x: chain(range(0, 240, 3),
                                                 range(240, 385, 6)),
                 'time_between_fcst_hrs': 60,
                 'delay_to_first_forecast': '280min',
                 'avg_max_run_length': '60min',
                 'filename': 'gefs_{stat_or_member}.nc',
                 'members': (['avg', 'c00', 'spr'] +
                             [f'p{r:02d}' for r in range(1, 21)]),
                 'check_url_name': 'gens'}


EXTRA_KEYS = ['update_freq', 'valid_hr_gen', 'time_between_fcst_hrs',
              'delay_to_first_forecast', 'avg_max_run_length', 'filename',
              'check_url_name', 'member']

model_map = {'gfs_0p25': GFS_0P25_1HR, 'nam_12km': NAM_CONUS,
             'rap': RAP, 'hrrr_hourly': HRRR_HOURLY,
             'hrrr_subhourly': HRRR_SUBHOURLY,
             'gefs': GEFS_0P50_RAW}

NC_TBL = """
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

    Parameters
    ----------
    get_func : function
        Function that performs an aiohttp call to be retried
    retries : int
        Number of retries before raising the error
    *args
        Passed to get_func
    **kwargs
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
        except aiohttp.ClientError:
            logger.warning('Request failed in connection, retrying')
            retried += 1
            if retried >= retries:
                raise
        else:
            return res

        await asyncio.sleep(60)


def _simple_model(model):
    return model['dir'].split('.')[0][1:]


@abort_all_on_exception
async def get_available_dirs(session, model):
    """Get the available date/date+init_hr directories"""
    simple_model = _simple_model(model)
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
        if p in params:
            del params[p]
    params['dir'] = params['dir'].format(
        init_date=init_time.strftime('%Y%m%d'),
        init_dt=init_time.strftime('%Y%m%d%H'),
        init_hr=init_time.strftime('%H'))
    for i in valid_hr_gen:
        newp = params.copy()
        newp['file'] = newp['file'].format(
            init_hr=init_time.hour,
            valid_hr=i)
        yield newp


async def check_next_inittime(session, init_time, model):
    """Check if data from the next model initializtion time is available"""
    next_inittime = init_time + pd.Timedelta(model['update_freq'])
    simple_model = _simple_model(model)
    next_init_url = (CHECK_URL.format(model.get('check_url_name',
                                                simple_model))
                     + model['dir'].format(
                         init_date=next_inittime.strftime('%Y%m%d'),
                         init_dt=next_inittime.strftime('%Y%m%d%H'),
                         init_hr=next_inittime.strftime('%H'))
                     + '/' + model['file'].format(init_hr=next_inittime.hour,
                                                  valid_hr=0))

    try:
        async with session.head(next_init_url) as r:
            if r.status == 200:
                logger.warning(
                    'Skipping to next init time at %s for %s %s',
                    next_inittime, simple_model, model.get('member', ''))
                return True
            else:
                return False
    except aiohttp.ClientOSError:
        return False


def get_filename(basepath, init_time, params):
    filename = (
        basepath / init_time.strftime('%Y/%m/%d/%H') / params['file'])
    if not filename.suffix == '.grib2':
        filename = filename.with_suffix(filename.suffix + '.grib2')
    return filename


async def files_to_retrieve(session, model, modelpath, init_time):
    """Generator to return the parameters of the available files for download
    """
    possible_params = _process_params(model, init_time)
    simple_model = _simple_model(model)
    first_file_modified_at = None
    for next_params in possible_params:
        logger.debug('Checking if file is available for %s', next_params)
        filename = get_filename(modelpath, init_time, next_params)
        if filename.exists():
            yield next_params
            continue
        next_model_url = (CHECK_URL.format(model.get('check_url_name',
                                                     simple_model))
                          + next_params['dir'] + '/' + next_params['file'])
        while True:
            # is the next file ready?
            try:
                logger.debug('Calling HEAD %s', next_model_url)
                async with session.head(
                        next_model_url, raise_for_status=True) as r:
                    if first_file_modified_at is None:
                        first_file_modified_at = pd.Timestamp(
                            r.headers['Last-Modified'])
                        logger.debug('First file was available at %s %s',
                                     first_file_modified_at,
                                     model.get('member', ''))
                logger.debug('HEAD returned %s', next_model_url)
            except aiohttp.ClientResponseError as e:
                if e.status == 404:  # Not found
                    logger.debug(
                        'Next file not ready yet for %s at %s %s\n%s %s',
                        simple_model, init_time, model.get('member', ''),
                        e.status, e.message)
                else:
                    logger.error(
                        'Error checking if next file is ready %s\n'
                        '%s %s', model.get('member', ''), e.status, e.message)
            except aiohttp.ClientError as e:
                logger.warning('Error in checking for next file %s %s',
                               model.get('member', ''), str(e))
            except Exception:
                logger.exception(
                    'Uncaught exception when checking for next file %s',
                    model.get('member', ''))
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
            async for chunk in r.content.iter_chunked(chunksize * 1024):
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
    filename = get_filename(basepath, init_time, params)
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


@abort_all_on_exception
async def process_grib_to_netcdf(folder, model):
    logger.info('Converting GRIB files to NetCDF with wgrib2 %s',
                model.get('member', ''))
    _handle, nctmp = tempfile.mkstemp()
    os.close(_handle)
    nctmp = Path(nctmp)
    # possible that this holds up processing on file io
    # so run in separate process
    grib_prefix = model['file'].split('.')[0]
    wind_in_model = 'var_WIND' not in model
    try:
        await run_in_executor(_process_grib, folder, nctmp, grib_prefix,
                              wind_in_model)
    except Exception:
        nctmp.unlink()
        raise
    return nctmp


def _process_grib(folder, nctmp, grib_prefix, wind_in_model):
    if wind_in_model:
        # need to add wind to the grib files
        for grbfile in folder.glob(f'{grib_prefix}*.grib2'):
            path = str(grbfile.resolve())
            try:
                subprocess.run(
                    f'wgrib2 {path} -wind_speed - -match "(UGRD|VGRD)" | '
                    f'wgrib2 - -append -grib_out {path}',
                    shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error('Error converting wind in file %s\n%s',
                             grbfile, e.stderr)
                raise OSError

    if 'subhourly' in str(folder):
        # for hrrr subhourly, assume TMP and VDDSF have no average but others
        fmt = "-match 'ave|TMP|VDDSF'"
    else:
        fmt = ''

    with tempfile.NamedTemporaryFile(mode='w') as tmp_nc_tbl:
        tmp_nc_tbl.write(NC_TBL)
        tmp_nc_tbl.flush()

        try:
            subprocess.run(
                f'cat {str(folder)}/{grib_prefix}*.grib2 | '
                f'wgrib2 - -nc4 -nc_table {tmp_nc_tbl.name} {fmt} -append -netcdf {str(nctmp)}',  # NOQA
                shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error('Error converting grib files %s*.grib2 to netCDF\n%s',
                         grib_prefix, e.stderr)
            raise OSError
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
    ds.close()


async def optimize_netcdf(nctmpfile, final_path):
    """Compress the netcdf file and adjust the chunking for fast time-series
    access"""
    logger.info('Optimizing NetCDF file to save at %s', final_path)
    parent = Path(final_path.parent)
    if not parent.is_dir():
        parent.mkdir(parents=True)
    _handle, tmp_path = tempfile.mkstemp(dir=parent)
    os.close(_handle)
    tmp_path = Path(tmp_path)
    # possible that this leaks memory, so run in separate process
    # that is restarted after a number of jobs
    try:
        await run_in_executor(_optimize_netcdf, nctmpfile, tmp_path)
    except Exception:
        tmp_path.unlink()
        raise
    else:
        tmp_path.rename(final_path)
        final_path.chmod(stat.S_IRGRP | stat.S_IRUSR | stat.S_IROTH |
                         stat.S_IWUSR)
        logger.info('Done optimizing NetCDF at %s', final_path)
    finally:
        nctmpfile.unlink()


async def sleep_until_inittime(inittime, model):
    # don't bother requesting a file until it might be ready
    now = pd.Timestamp.utcnow()
    likely_ready_time = inittime + pd.Timedelta(
        model['delay_to_first_forecast'])
    if likely_ready_time > now:
        seconds = (likely_ready_time - now).total_seconds()
        logger.info('Sleeping %0.1fs for next model run %s', seconds,
                    model.get('member', ''))
        await asyncio.sleep(seconds)


async def startup_find_next_runtime(model_path, session, model):
    """Find the next model run to get based on what is available
    on NOMADS and what .nc files are present locally"""
    dirs = await get_available_dirs(session, model)
    no_file = []
    first = pd.Timestamp('20000101T0000Z')
    max_time = first
    for dir_ in dirs:
        if len(dir_) == 8:
            path = model_path / dir_[:4] / dir_[4:6] / dir_[6:8]
            for hr in range(0, 24, int(model['update_freq'].strip('h'))):
                hrpath = path / f'{hr:02d}'
                hrtime = pd.Timestamp(f'{dir_[:8]}T{hr:02d}00Z')
                if not (hrpath / model['filename']).exists():
                    no_file.append(hrtime)
                else:
                    max_time = max(max_time, hrtime)
        else:
            hrpath = model_path / dir_[:4] / dir_[4:6] / dir_[6:8] / dir_[8:10]
            hrtime = pd.Timestamp(f'{dir_[:8]}T{dir_[8:10]}00Z')
            if not (hrpath / model['filename']).exists():
                no_file.append(hrtime)
            else:
                max_time = max(max_time, hrtime)
    if len(no_file) == 0:
        if max_time > first:
            inittime = max_time + pd.Timedelta(model['update_freq'])
        else:  # No available dirs?
            raise ValueError(
                'Failed to find next available model from NOMADS %s' %
                model.get('member', ''))
    else:
        inittime = min(no_file)
    await sleep_until_inittime(inittime, model)
    return inittime


async def next_run_time(inittime, modelpath, model):
    inittime += pd.Timedelta(model['update_freq'])
    # check if nc file exists for this inittime
    if (
            modelpath / inittime.strftime('%Y/%m/%d/%H') /
            model['filename']).exists():
        return await next_run_time(inittime, modelpath, model)
    await sleep_until_inittime(inittime, model)
    return inittime


async def _run_loop(session, model, modelpath, chunksize, once, use_tmp):
    inittime = await startup_find_next_runtime(modelpath, session, model)
    while True:
        fetch_tasks = set()
        finalpath = (modelpath / inittime.strftime('%Y/%m/%d/%H') /
                     model['filename'])
        if use_tmp:
            _tmpdir = tempfile.TemporaryDirectory()
            gribdir = Path(_tmpdir.name)
        else:
            gribdir = modelpath
        async for params in files_to_retrieve(session, model, gribdir,
                                              inittime):
            logger.debug('Processing parameters %s', params)
            fetch_tasks.add(asyncio.create_task(
                fetch_grib_files(session, params, gribdir, inittime,
                                 chunksize)))
        files = await asyncio.gather(*fetch_tasks)
        if len(files) != 0:  # skip to next inittime
            path_to_files = files[0].parent
            try:
                nctmpfile = await process_grib_to_netcdf(path_to_files,
                                                         model)
                await optimize_netcdf(nctmpfile, finalpath)
            except Exception:
                raise
        if use_tmp:
            _tmpdir.cleanup()
        else:
            # remove grib files
            for f in files:
                f.unlink()
        if once:
            break
        else:
            logger.info('Moving on to next model run')
            inittime = await next_run_time(inittime, modelpath, model)


async def run(basepath, model_name, chunksize, once=False, use_tmp=False):
    session = make_session()
    modelpath = basepath / model_name
    if model_name != 'gefs':
        model = model_map[model_name]
        await _run_loop(session, model, modelpath, chunksize, once, use_tmp)
    else:
        base_model = model_map[model_name].copy()
        members = base_model.pop('members')
        member_loops = set()
        for member in members:
            model = base_model.copy()
            model['member'] = member
            model['file'] = model['file'].replace('{stat_or_member}', member)
            model['filename'] = model['filename'].format(stat_or_member=member)
            member_loops.add(asyncio.create_task(
                _run_loop(session, model, modelpath, chunksize, once,
                          use_tmp)))
        await asyncio.wait(member_loops)
    await session.close()


async def optimize_only(path_to_files, model_name):
    model = model_map[model_name]
    nctmpfile = await process_grib_to_netcdf(path_to_files, model)
    try:
        await optimize_netcdf(
            nctmpfile, path_to_files / f'{model_name}.nc')
    except Exception:
        raise
    else:
        # remove grib files
        for f in path_to_files.glob(f'{model["file"].split(".")[0]}*.grib2'):
            f.unlink()


def check_wgrib2():
    if shutil.which('wgrib2') is None:
        logger.error('wgrib2 was not found in PATH and is required')
        sys.exit(1)
