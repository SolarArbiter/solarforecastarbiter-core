
from solarforecastarbiter.reports import template


def test_templates():
    out = template.main()
    assert len(out) > 0
