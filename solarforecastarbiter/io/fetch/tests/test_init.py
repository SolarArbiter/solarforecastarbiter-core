import multiprocessing as mp
import os
import subprocess
import time


import pytest
from loky.process_executor import TerminatedWorkerError


from solarforecastarbiter.io import fetch


def badfun():
    raise ValueError


def bad_subprocess():
    subprocess.run(['cat', '/nowaythisworks'], check=True, capture_output=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('bad,err', [
    (badfun, ValueError),
    (bad_subprocess, subprocess.CalledProcessError)
])
async def test_cluster_error(bad, err):
    with pytest.raises(err):
        await fetch.run_in_executor(bad)


def getpid():  # pragma: no cover
    return mp.current_process().pid


def longrunning():  # pragma: no cover
    time.sleep(3)


@pytest.mark.asyncio
@pytest.mark.timeout(5, method='thread')
async def test_cluster_external_kill():
    pid = await fetch.run_in_executor(getpid)
    long = fetch.run_in_executor(longrunning)
    os.kill(pid, 9)
    with pytest.raises(TerminatedWorkerError):
        await long
