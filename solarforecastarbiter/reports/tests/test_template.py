
from solarforecastarbiter.reports import template

from test_main import dummy_metrics


def test_templates():
    metadata = {'versions': 'None'}
    metrics = dummy_metrics
    out = template.main(metadata, metrics)
    assert len(out) > 0
    assert False
