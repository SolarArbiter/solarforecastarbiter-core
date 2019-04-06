"""
Fetch NWP files from NCEP Nomads
"""
import asyncio
import argparse
import logging
from pathlib import Path
import re
import signal
import sys


import aiohttp
import pandas as pd


from solarforecastarbiter.io.fetch import (
    handle_exception, basic_logging_config, make_session)


logger = logging.getLogger(__name__)


CHECK_URL = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/{}/prod'
BASE_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/'
DOMAIN = {'subregion': None,
          'leftlon': -126,
          'rightlon': -66,
          'toplat': 50,
          'bottomlat': 24}


def _gfs_valid_hr_gen(init_hr):
    i = 0
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
                'valid_hr_gen': _gfs_valid_hr_gen}


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
             'valid_hr_gen': _nam_valid_hr_gen}


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
           lambda x: range(40) if x in (3, 9, 15, 21) else range(22))}


HRRR = {'endpoint': 'filter_hrrr_2d.pl',
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
            lambda x: range(37) if x in (0, 6, 12, 18) else range(19))}

model_map = {'gfs_0p25': GFS_0P25_1HR, 'nam_12km': NAM_CONUS,
             'rap': RAP, 'hrrr': HRRR}


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
            if retried >= retries:
                raise
            retried += 1
            await asyncio.sleep(60)
        else:
            return res


def _make_regex_from_filename(filename):
    out = filename
    format_fields = re.findall('\\{\\w*:\\w*\\}', filename)
    for field in format_fields:
        num_digits = str(int(re.findall('\\d+', field)[0]))
        out = out.replace(field, '(\\d{' + num_digits + '})')
    return out


async def get_available_dirs(session, model):
    """Get the available date/date+init_hr directories"""
    simple_model = model['file'].split('.')[0]
    is_init_date = 'init_date' in model['dir']
    model_url = BASE_URL + model['endpoint']

    async def _get(model_url):
        async with session.get(model_url) as r:
            return await r.text()

    page = await get_with_retries(_get, model_url)
    if is_init_date:
        list_avail_days = set(
            re.findall(simple_model + '\\.([0-9]{8})', page))
    else:
        list_avail_days = set(
            re.findall(simple_model + '\\.([0-9]{10})', page))
    return list_avail_days


async def get_available_runs_in_dir(session, model, dir_):
    """Get the available runs in a given directory"""
    model_url = BASE_URL + model['endpoint']
    fname_regex = _make_regex_from_filename(model['file'])
    async with session.get(model_url, params={'dir': dir_}) as r:
        if r.status != 200:
            pass
        files_text = await r.text()
    init_valid = set(re.findall(fname_regex, files_text))
    dateday = pd.Timestamp(dir_.split('.')[-1][:8])
    day_init_valid = [(dateday, int(iv[0]), int(iv[1]))
                      for iv in init_valid]
    df = pd.DataFrame.from_records(
        sorted(day_init_valid), columns=['date', 'init_hr', 'valid_hr'])
    return df


def _process_params(model, init_time):
    """Generator to get the parameters for fetching forecasts for a given
    model at a given init_time"""
    params = model.copy()
    del params['update_freq']
    valid_hr_gen = params['valid_hr_gen'](init_time.hour)
    del params['valid_hr_gen']
    params['dir'] = params['dir'].format(
        init_date=init_time.strftime('%Y%m%d'),
        init_dt=init_time.strftime('%Y%m%d%H'))
    for i in valid_hr_gen:
        newp = params.copy()
        newp['file'] = newp['file'].format(
            init_hr=init_time.hour,
            valid_hr=i)
        yield newp


async def files_to_retrieve(session, model, init_time):
    """Generator to return the parameters of the available files for download
    """
    possible_params = _process_params(model, init_time)
    next_inittime = init_time + pd.Timedelta(model['update_freq'])
    simple_model = model['file'].split('.')[0]
    next_init_url = (CHECK_URL.format(simple_model)
                     + model['dir'].format(
                         init_date=next_inittime.strftime('%Y%m%d'),
                         init_dt=next_inittime.strftime('%Y%m%d%H'))
                     + '/' + model['file'].format(init_hr=next_inittime.hour,
                                                  valid_hr=0))
    for next_params in possible_params:
        next_model_url = (CHECK_URL.format(simple_model)
                          + next_params['dir'] + '/' + next_params['file'])
        while True:
            # is the next file ready?
            async with session.head(next_model_url) as r:
                if r.status == 200:
                    logger.info('%s/%s is ready for download',
                                next_params['dir'], next_params['file'])
                    yield next_params
                    break
                else:
                    logger.debug('Next file not ready yet for %s at %s',
                                 simple_model, init_time)
            # was the older run cancelled?
            async with session.head(next_init_url) as r:
                if r.status == 200:
                    logger.warning(
                        'Skipping to next init time at %s for %s',
                        next_inittime, simple_model)
                    return
            await asyncio.sleep(300)


async def _get_file(session, url, params, tmpfile, chunksize):
    async with session.get(url, params=params, raise_for_status=True) as r:
        with open(tmpfile, 'wb') as f:
            while True:
                chunk = await r.content.read(chunksize * 1024)
                if not chunk:
                    break
                f.write(chunk)


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
        basepath / init_time.strftime('%Y/%m/%d/%H') / params['file']
        ).with_suffix('.grib2')
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


def process_grib_to_netcdf():
    # make x, y coords in proper projection for hrrr
    pass


def optimize_netcdf():
    pass


async def find_next_runtime(model_path, session, model):
    dirs = await get_available_dirs(session, model)
    no_file = []
    for dir_ in dirs:
        # path w/ year/month/day
        path = model_path / dir_[:4] / dir_[4:6] / dir_[6:8]
        for hr in range(0, 24, int(model['update_freq'].strip('h'))):
            hrpath = path / f'{hr:02d}'
            glob = list(hrpath.glob('*.nc'))
            if len(glob) == 0:
                no_file.append(pd.Timestamp(f'{dir_[:8]}T{hr:02d}00'))
    if len(no_file) == 0:
        # all files for current dirs present, get next day
        return max([pd.Timestamp(f'{dir_[:8]}T0000')]) + pd.Timedelta('1d')
    else:
        return min(no_file)


async def run_once(basepath, model_name, chunksize):
    session = make_session()
    modelpath = basepath / model_name
    model = model_map[model_name]
    inittime = await find_next_runtime(modelpath, session, model)
    breakpoint()
    tasks = set()
    async for params in files_to_retrieve(session, model, inittime):
        tasks.add(asyncio.create_task(
            fetch_grib_files(session, params, modelpath, inittime,
                             chunksize)))

    await asyncio.gather(*tasks)
    await session.close()


def main():
    sys.excepthook = handle_exception
    basic_logging_config()
    argparser = argparse.ArgumentParser(
        description='Retrieve forecasts from the fxapi and post them to PI')
    argparser.add_argument('-v', '--verbose', action='count')
    argparser.add_argument('--chunksize', default=128,
                           help='Size of a chunk (in KB) to save at one time')
    argparser.add_argument('save_directory',
                           help='Directory to save data in')
    argparser.add_argument(
        'model', choices=['gfs_0p25', 'nam_12km', 'rap', 'hrrr'],
        help='The model to get data for')
    args = argparser.parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose and args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    basepath = Path(args.save_directory).resolve()
    fut = asyncio.ensure_future(run_once(basepath, args.model, args.chunksize))

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, fut.cancel)

    loop.run_until_complete(fut)


if __name__ == '__main__':
    main()
