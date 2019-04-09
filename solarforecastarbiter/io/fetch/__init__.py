import asyncio
import atexit
from functools import partial, wraps
import logging
import multiprocessing as mp
import signal
import sys
import threading


import aiohttp


cluster = None


def ignore_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def start_cluster(max_workers=4, maxtasksperchild=5):
    global cluster
    mp.set_start_method("forkserver")
    cluster = mp.Pool(max_workers, maxtasksperchild=maxtasksperchild,
                      initializer=ignore_interrupt)
    atexit.register(cluster.terminate)
    return


async def run_in_executor(func, *args, **kwargs):
    exc = partial(cluster.apply, func, args, kwargs)
    # uses the asyncio default thread pool executor to then
    # apply the function on the pool of processes
    # inefficient, but ProcessPoolExecutor will not restart
    # processes in case of memory leak
    res = await asyncio.get_event_loop().run_in_executor(None, exc)
    return res


def make_session():
    """Make an aiohttp session"""
    s = aiohttp.ClientSession(read_timeout=60, conn_timeout=60)
    return s


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception",
                  exc_info=(exc_type, exc_value, exc_traceback))


def basic_logging_config():
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')


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
