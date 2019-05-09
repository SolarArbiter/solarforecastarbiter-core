import dramatiq
import pytest


from solarforecastarbiter import tasks


@pytest.fixture()
def stub_broker():
    broker = tasks.broker
    return broker


@pytest.fixture()
def stub_worker(stub_broker):
    worker = dramatiq.Worker(stub_broker, worker_timeout=100)
    worker.start()
    yield worker
    worker.stop()


def test_enqueue_function_stub(stub_broker, stub_worker, mocker):
    @dramatiq.actor()
    def noop(*args):
        return args
    assert ('arg1', 'arg2') == tasks.enqueue_function(noop, 'arg1', 'arg2')


def test_enqueue_function(mocker):
    @dramatiq.actor()
    def noop(*args):
        return args  # pragma: no cover
    mocker.patch('solarforecastarbiter.tasks.broker',
                 new=None)
    mocked = mocker.patch.object(noop, 'send')
    tasks.enqueue_function(noop, 'a', 'b')
    mocked.assert_called_with('a', 'b')


def test_immediate_observation_validation_task(stub_broker, stub_worker,
                                               mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.validation.tasks.immediate_observation_validation')  # NOQA
    task = tasks.immediate_observation_validation.send('TOKEN', 'OBSID',
                                                       'start', 'end')
    stub_broker.join(task.queue_name)
    stub_worker.join()
    mocked.assert_called_with('TOKEN', 'OBSID', 'start', 'end')
