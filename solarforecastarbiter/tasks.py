"""
High-level tasks that imports other tasks and defines the task queue
"""
import os


import dramatiq
from dramatiq.brokers.stub import StubBroker


from solarforecastarbiter.validation import tasks as validation_tasks


if 'REDIS_URL' in os.environ:  # pragma: no cover
    from dramatiq.brokers.redis import RedisBroker
    broker = RedisBroker(url=os.environ['REDIS_URL'],
                         db=0,
                         namespace='sfa-queue')
else:
    broker = StubBroker()
dramatiq.set_broker(broker)


def enqueue_function(func, *args, **kwargs):
    """Convience function to queue function. Will allow for altering task
    queue without changing code that queues up the tasks"""
    return func.send(*args, **kwargs)


@dramatiq.actor(max_retries=3)
def immediate_observation_validation(*args, **kwargs):
    return validation_tasks.immediate_observation_validation(*args, **kwargs)
