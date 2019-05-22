
from solarforecastarbiter.reports import template

from solarforecastarbiter.reports.tests.test_main import dummy_metrics


def test_templates():
    return
    metadata = {'versions': 'None'}
    metrics = dummy_metrics
    out = template.prereport(report, metadata, metrics)
    assert len(out) > 0
    assert False
