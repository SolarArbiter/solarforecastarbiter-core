import pytest
from jinja2 import Environment, PackageLoader, select_autoescape


from solarforecastarbiter import datamodel


@pytest.fixture
def macro_test_template():
    def fn(macro_name_and_args):
        macro_template = f"{{% import 'macros.j2' as macros with context%}}{{{{macros.{macro_name_and_args} | safe }}}}"  # noqa
        env = Environment(
            loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
            autoescape=select_autoescape(['html', 'xml']),
            lstrip_blocks=True,
            trim_blocks=True
        )
        html_template = env.from_string(macro_template)
        return html_template
    return fn


metric_format = """<table style="width:100%;">
  <thead>
    <tr class="header">
      <th style="text-align: left;">Forecast</th>
        <th style="text-align: left;">MAE</th>
        <th style="text-align: left;">RMSE</th>
        <th style="text-align: left;">MBE</th>
    </tr>
  </thead>
  <tbody>
      <tr>
        <td>{}</td>
                <td>2</td>
                <td>2</td>
                <td>2</td>
      </tr>
  </tbody>
</table>
"""


def test_metric_table(report_with_raw, macro_test_template):
    metric_table_template = macro_test_template('metric_table(report_metrics,category,metric_ordering)')  # noqa
    metrics = report_with_raw.raw_report.metrics
    category = 'total'
    for i in range(3):
        expected_metric = (metrics[i],)
        rendered_metric_table = metric_table_template.render(
            report_metrics=expected_metric,
            category=category,
            metric_ordering=report_with_raw.metrics,
            human_metrics=datamodel.ALLOWED_METRICS)
        assert rendered_metric_table == metric_format.format(
            expected_metric[0].name)


def test_download_csv_script(macro_test_template):
    download_template = macro_test_template('download_csv_script()')
    rendered = download_template.render()
    # should this test more?
    assert rendered[:8] == '<script>'
    assert rendered[-10:] == '</script>\n'
