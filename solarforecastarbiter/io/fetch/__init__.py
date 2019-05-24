import asyncio
import atexit
from functools import partial, wraps
import logging
import multiprocessing as mp
import signal
import threading


import aiohttp


CLUSTER = None


def ignore_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def start_cluster(max_workers=4, maxtasksperchild=5):
    global CLUSTER
    mp.set_start_method("forkserver")
    CLUSTER = mp.Pool(max_workers, maxtasksperchild=maxtasksperchild,
                      initializer=ignore_interrupt)
    atexit.register(CLUSTER.terminate)
    return


async def run_in_executor(func, *args, **kwargs):
    exc = partial(CLUSTER.apply, func, args, kwargs)
    # uses the asyncio default thread pool executor to then
    # apply the function on the pool of processes
    # inefficient, but ProcessPoolExecutor will not restart
    # processes in case of memory leak
    res = await asyncio.get_event_loop().run_in_executor(None, exc)
    return res


def make_session():
    """Make an aiohttp session"""
    conn = aiohttp.TCPConnector(limit_per_host=20)
    timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
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
