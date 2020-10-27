import asyncio
from functools import partial, wraps
import logging
import os
import signal
import threading


import aiohttp
from loky import get_reusable_executor


WORKERS = 1


def ignore_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def update_num_workers(max_workers):
    global WORKERS
    WORKERS = max_workers


async def run_in_executor(func, *args, **kwargs):
    exc = get_reusable_executor(max_workers=WORKERS)
    # uses the asyncio default thread pool executor to then
    # apply the function on the pool of processes
    # inefficient, but ProcessPoolExecutor will not restart
    # processes in case of memory leak
    res = await asyncio.get_event_loop().run_in_executor(
        exc, partial(func, *args, **kwargs))
    return res


def make_session():
    """Make an aiohttp session"""
    conn = aiohttp.TCPConnector(limit_per_host=int(
        os.getenv('AIO_CONN_LIMIT', 10)))
    timeout = aiohttp.ClientTimeout(
        total=int(os.getenv('AIO_TOTAL_TIMEOUT', 80)),
        connect=int(os.getenv('AIO_CONN_TIMEOUT', 15)),
        sock_read=int(os.getenv('AOI_READ_TIMEOUT', 30)),
        sock_connect=int(os.getenv('AIO_CONN_TIMEOUT', 15)),
    )
    s = aiohttp.ClientSession(connector=conn, timeout=timeout)
    return s


def abort_all_on_exception(f):
    @wraps(f)
    async def wrapper(*args, **kwargs):
        try:
            ret = await f(*args, **kwargs)
        except Exception:
            logging.exception('Aborting on error')
            signal.pthread_kill(threading.get_ident(), signal.SIGUSR1)
        else:
            return ret
    return wrapper
