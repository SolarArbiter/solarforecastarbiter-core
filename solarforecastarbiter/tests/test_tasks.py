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


def test_immediate_observation_validation_task(stub_broker, stub_worker,
                                               mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.validation.tasks.immediate_observation_validation')  # NOQA
    task = tasks.enqueue_function(
        tasks.immediate_observation_validation, 'TOKEN', 'OBSID', 'start',
        'end')
    stub_broker.join(task.queue_name)
    stub_worker.join()
    mocked.assert_called_with('TOKEN', 'OBSID', 'start', 'end')
