"""
High-level tasks that imports other tasks and defines the task queue
"""
import os


import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.brokers.stub import StubBroker


from solarforecastarbiter.validation import tasks as validation_tasks


if os.getenv('UNIT_TEST') == '1':
    broker = StubBroker()
else:  # pragma: no cover
    broker = RedisBroker(host=os.getenv('REDIS_HOST', '127.0.0.1'),
                         port=int(os.getenv('REDIS_PORT', '6379')),
                         db=0,
                         password=os.getenv('REDIS_PASSWORD', ''),
                         namespace='sfa-queue')
dramatiq.set_broker(broker)


def enqueue_function(func, *args, **kwargs):
    """Convience function to queue function. Will allow for altering task
    queue without changing code that queues up the tasks"""
    return func.send(*args, **kwargs)


@dramatiq.actor(max_retries=3)
def immediate_observation_validation(*args, **kwargs):
    return validation_tasks.immediate_observation_validation(*args, **kwargs)
