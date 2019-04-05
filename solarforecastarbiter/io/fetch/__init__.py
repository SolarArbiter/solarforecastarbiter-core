import asyncio
import atexit
from functools import partial
import logging
import multiprocessing as mp
import sys


import aiohttp


cluster = None


def start_cluster(max_workers=4):
    global cluster
    mp.set_start_method("forkserver")
    cluster = mp.Pool(max_workers, maxtasksperchild=100)
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
    s = aiohttp.ClientSession(read_timeout=60,
                              conn_timeout=600)
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
