import json
from jinja2 import Environment, PackageLoader, select_autoescape
import pytest


from solarforecastarbiter.reports import template


@pytest.mark.parametrize('metadata,expected', [
    (json.dumps([{'name': 'obs', 'field': '{"field": {"inner": 1}}'}]),
     r'"[{\"name\": \"obs\", \"field\": \"{\\\"field\\\": {\\\"inner\\\": 1}}\"}]"'),  # noqa
    (json.dumps([{'name': 'obs', 'field': '{"field": "<div>html</div>"}'}]),
     r'"[{\"name\": \"obs\", \"field\": \"{\\\"field\\\": \\\"\u003cdiv\u003ehtml\u003c/div\u003e\\\"}\"}]"'),  # noqa

])
def test_load_metadata(metadata, expected):
    env = Environment(
        loader=PackageLoader(
            'solarforecastarbiter.reports', 'templates/html'),
        autoescape=select_autoescape(['html', 'xml']),
        lstrip_blocks=True,
        trim_blocks=True
    )
    env.filters['unique_flags_filter'] = template._unique_flags_filter
    html_template = env.get_template('load_metadata.html')
    rendered = html_template.render(metadata_json=metadata)
    ex = f"<script>var metadata_json = JSON.parse({expected});</script>"
    assert rendered == ex
