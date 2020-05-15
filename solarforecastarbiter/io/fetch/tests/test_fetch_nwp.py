from pathlib import Path
import shutil


import aiohttp
from asynctest import CoroutineMock, MagicMock
import pandas as pd
from pkg_resources import resource_filename, Requirement
import pytest


from solarforecastarbiter.io.fetch import nwp


def test_gfs_valid_hr_gen():
    expected = (list(range(120)) + list(range(120, 240, 3)) +
                list(range(240, 385, 12)))
    assert list(nwp.GFS_0P25_1HR['valid_hr_gen'](0)) == expected


def test_nam_valid_hr_gen():
    expected = list(range(36)) + list(range(36, 85, 3))
    assert list(nwp.NAM_CONUS['valid_hr_gen'](0)) == expected


@pytest.mark.asyncio
async def test_get_with_retries(mock_sleep):
    getfcn = CoroutineMock()
    getfcn.side_effect = [
        aiohttp.ClientResponseError(MagicMock(), MagicMock())] * 4 + ['hi']
    res = await nwp.get_with_retries(getfcn, retries=5)
    assert res == 'hi'
    assert getfcn.await_count == 5


@pytest.mark.asyncio
async def test_get_with_retries_fails(mock_sleep):
    getfcn = CoroutineMock()
    getfcn.side_effect = [
        aiohttp.ClientResponseError(MagicMock(), MagicMock())] * 4 + ['hi']
    with pytest.raises(aiohttp.ClientResponseError):
        await nwp.get_with_retries(getfcn, retries=4)


@pytest.mark.asyncio
@pytest.mark.parametrize('page,expected,model', [
    ("""
    nam.20190409 other things
    more before ... 201900 nam.20190410
    """, ['20190409', '20190410'], nwp.NAM_CONUS),
    ("""
    gfs.20190409 other things
    more before ... 201900 nam.20190410
    """, ['20190409'], nwp.GFS_0P25_1HR),
])
async def test_get_available_dirs(mocker, page, expected, model):
    get = mocker.patch('solarforecastarbiter.io.fetch.nwp.get_with_retries',
                       new=CoroutineMock())
    get.return_value = page
    assert await nwp.get_available_dirs(None, model) == set(expected)


def test_process_params():
    paramgen = nwp._process_params(nwp.GFS_0P25_1HR,
                                   pd.Timestamp('20190409T1200Z'))
    params = next(paramgen)
    assert 'subregion' in params
    assert 'leftlon' in params
    assert 'endpoint' in params
    assert 'file' in params
    assert 'valid_hr_gen' not in params
    assert 'lev_surface' in params
    assert params['file'] == 'gfs.t12z.pgrb2.0p25.f000'


@pytest.mark.asyncio
async def test_check_next_inittime(mocker):
    session = MagicMock()
    session.head.return_value.__aenter__.return_value.status = 200
    assert await nwp.check_next_inittime(
        session, pd.Timestamp('20190101T0000Z'), nwp.RAP)


@pytest.mark.asyncio
async def test_check_next_inittime_400(mocker):
    session = MagicMock()
    session.head.return_value.__aenter__.return_value.status = 400
    assert not await nwp.check_next_inittime(
        session, pd.Timestamp('20190101T0000Z'), nwp.RAP)


@pytest.mark.asyncio
async def test_check_next_inittime_raises(mocker):
    session = MagicMock()
    session.head.return_value.__aenter__.return_value.status = 200
    session.head.return_value.__aenter__.side_effect = aiohttp.ClientOSError
    assert not await nwp.check_next_inittime(
        session, pd.Timestamp('20190101T0000Z'), nwp.RAP)


@pytest.fixture()
def mock_sleep(mocker):
    sleep = mocker.patch('solarforecastarbiter.io.fetch.nwp.asyncio.sleep',
                         new=CoroutineMock())
    sleep.return_value = None


@pytest.mark.asyncio
async def test_files_to_retrieve(mocker, mock_sleep):
    session = MagicMock()
    session.head.return_value.__aenter__.return_value.headers.__getitem__.return_value = '20190101T0000Z'  # NOQA

    params = [p async for p in nwp.files_to_retrieve(
        session, nwp.RAP, Path('/'), pd.Timestamp('20190409T1200Z'))]
    assert len(params) == 22


@pytest.mark.asyncio
async def test_files_to_retrieve_ends_after_no_new(mocker, mock_sleep):
    def make_header_mocks(init):
        for i in range(100):
            # first time needs headers for lastmod
            if i == 0:
                hmock = MagicMock()
                hmock.headers.__getitem__.return_value = init
                yield hmock
            elif i in range(3, 13):
                yield aiohttp.ClientOSError
            else:
                yield i

    model = nwp.RAP.copy()
    model['avg_max_run_length'] = '30min'
    init = pd.Timestamp('20190409T0000Z')
    headers = list(make_header_mocks(init))
    session = MagicMock()
    session.head.return_value.__aenter__.side_effect = headers
    checknext = mocker.patch(
            'solarforecastarbiter.io.fetch.nwp.check_next_inittime',
            new=CoroutineMock())
    checknext.return_value = False

    params = [p async for p in nwp.files_to_retrieve(session, model, Path('/'),
                                                     init)]

    assert len(params) == 22
    assert checknext.await_count == 10


@pytest.mark.asyncio
async def test_files_to_retrieve_cut_short(mocker, mock_sleep):
    def make_header_mocks(init):
        for i in range(100):
            # first time needs headers for lastmod
            if i == 0:
                hmock = MagicMock()
                hmock.headers.__getitem__.return_value = init
                yield hmock
            elif i == 10:
                yield aiohttp.ClientOSError
            else:
                yield i

    model = nwp.RAP.copy()
    # so now is after avg max run length
    model['avg_max_run_length'] = '30min'
    init = pd.Timestamp('20190409T0000Z')
    headers = list(make_header_mocks(init))
    session = MagicMock()
    session.head.return_value.__aenter__.side_effect = headers
    checknext = mocker.patch(
            'solarforecastarbiter.io.fetch.nwp.check_next_inittime',
            new=CoroutineMock())
    checknext.return_value = True
    params = [p async for p in nwp.files_to_retrieve(session, model, Path('/'),
                                                     init)]
    assert len(params) == 10


@pytest.mark.asyncio
async def test_sleep_until_inittime(mocker):
    m = {'delay_to_first_forecast': '30s'}
    inittime = pd.Timestamp.utcnow()
    sleep = mocker.patch('solarforecastarbiter.io.fetch.nwp.asyncio.sleep',
                         new=CoroutineMock())

    await nwp.sleep_until_inittime(inittime, m)
    assert sleep.mock_calls[0]


@pytest.mark.asyncio
async def test_startup_find_next_runtime(mocker, tmp_path):
    mocker.patch('solarforecastarbiter.io.fetch.nwp.get_available_dirs',
                 new=CoroutineMock(return_value=['20190409',
                                                 '2019041000',
                                                 '2019041006',
                                                 '2019041012']))
    mocker.patch('solarforecastarbiter.io.fetch.nwp.sleep_until_inittime',
                 new=CoroutineMock())
    model = {'update_freq': '6h', 'filename': 'file.nc'}
    withnc = ['2019/04/09/00', '2019/04/09/06', '2019/04/09/12',
              '2019/04/09/18', '2019/04/10/00', '2019/04/10/12']
    for p in withnc:
        pp = tmp_path / p
        pp.mkdir(parents=True)
        (pp / 'file.nc').touch()

    res = await nwp.startup_find_next_runtime(tmp_path, None, model)
    assert res == pd.Timestamp('20190410T0600Z')


@pytest.mark.asyncio
async def test_startup_find_next_runtime_all_there(mocker, tmp_path):
    mocker.patch('solarforecastarbiter.io.fetch.nwp.get_available_dirs',
                 new=CoroutineMock(return_value=['20190409',
                                                 '2019041000',
                                                 '2019041006',
                                                 '2019041012']))
    mocker.patch('solarforecastarbiter.io.fetch.nwp.sleep_until_inittime',
                 new=CoroutineMock())
    model = {'update_freq': '6h', 'filename': 'file.nc'}
    withnc = ['2019/04/09/00', '2019/04/09/06', '2019/04/09/12',
              '2019/04/09/18', '2019/04/10/00', '2019/04/10/06',
              '2019/04/10/12']
    for p in withnc:
        pp = tmp_path / p
        pp.mkdir(parents=True)
        (pp / 'file.nc').touch()

    res = await nwp.startup_find_next_runtime(tmp_path, None, model)
    assert res == pd.Timestamp('20190410T1800Z')


@pytest.mark.asyncio
async def test_startup_find_next_runtime_all_there_out_of_order(
        mocker, tmp_path):
    mocker.patch('solarforecastarbiter.io.fetch.nwp.get_available_dirs',
                 new=CoroutineMock(return_value=['20190409',
                                                 '2019041006',
                                                 '2019041012',
                                                 '2019041000']))
    mocker.patch('solarforecastarbiter.io.fetch.nwp.sleep_until_inittime',
                 new=CoroutineMock())
    model = {'update_freq': '6h', 'filename': 'file.nc'}
    withnc = ['2019/04/09/00', '2019/04/09/06', '2019/04/09/12',
              '2019/04/10/00', '2019/04/10/06', '2019/04/09/18',
              '2019/04/10/12']
    for p in withnc:
        pp = tmp_path / p
        pp.mkdir(parents=True)
        (pp / 'file.nc').touch()

    res = await nwp.startup_find_next_runtime(tmp_path, None, model)
    assert res == pd.Timestamp('20190410T1800Z')


@pytest.mark.asyncio
async def test_next_run_time(mocker, tmp_path):
    model = {'update_freq': '12h', 'filename': 'file.nc'}
    mocker.patch('solarforecastarbiter.io.fetch.nwp.sleep_until_inittime',
                 new=CoroutineMock())
    init = pd.Timestamp('20190409T0000Z')
    withnc = ['2019/04/09/00', '2019/04/09/12',
              '2019/04/10/12']
    for p in withnc:
        pp = tmp_path / p
        pp.mkdir(parents=True)
        (pp / 'file.nc').touch()
    res = await nwp.next_run_time(init, tmp_path, model)
    assert res == pd.Timestamp('20190410T0000Z')


@pytest.mark.asyncio
async def test_optimize_only(mocker, tmp_path):
    if shutil.which('wgrib2') is None:
        pytest.skip('wgrib2 must be on the PATH to run this test')
    grib_dir = tmp_path / 'grib'
    shutil.copytree(
        resource_filename(
            Requirement.parse('solarforecastarbiter'),
            'solarforecastarbiter/io/fetch/tests/data/grib'),
        grib_dir)

    async def run(func, *args, **kwargs):
        return func(*args, **kwargs)

    mocker.patch('solarforecastarbiter.io.fetch.signal.pthread_kill')
    mocker.patch('solarforecastarbiter.io.fetch.nwp.run_in_executor',
                 new=run)
    await nwp.optimize_only(grib_dir, 'rap')
    assert [x.name for x in grib_dir.iterdir()] == ['rap.nc']
