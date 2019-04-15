import subprocess


import pytest


from solarforecastarbiter.io import fetch


def badfun():
    raise ValueError


def bad_subprocess():
    subprocess.run(['cat', '/nowaythisworks'], check=True, capture_output=True)


@pytest.fixture(scope='session')
def startcluster():
    fetch.start_cluster(1)


@pytest.mark.asyncio
@pytest.mark.parametrize('bad,err', [
    (badfun, ValueError),
    (bad_subprocess, subprocess.CalledProcessError)
])
async def test_cluster_error(bad, err, startcluster):
    with pytest.raises(err):
        await fetch.run_in_executor(bad)
