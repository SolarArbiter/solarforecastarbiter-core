import pytest

pytest.importorskip("jinja2", reason="requires [all] packages")  # noqa:E402

from jinja2 import Environment, PackageLoader, select_autoescape


from solarforecastarbiter import datamodel
from solarforecastarbiter.reports import template


@pytest.fixture
def macro_test_template():
    def fn(macro_name_and_args):
        macro_template = f"{{% import 'macros.j2' as macros with context%}}{{{{macros.{macro_name_and_args} | safe }}}}"  # noqa
        env = Environment(
            loader=PackageLoader(
                'solarforecastarbiter.reports', 'templates/html'),
            autoescape=select_autoescape(['html', 'xml']),
            lstrip_blocks=True,
            trim_blocks=True
        )
        env.filters['unique_flags_filter'] = template._unique_flags_filter
        html_template = env.from_string(macro_template)
        return html_template
    return fn


metric_table_fx_vert_format = """<details>
  <summary>
    <h4>Table of {} metrics</h4>
  </summary>
  <div class="report-table-wrapper">
  <table class="table table-striped metric-table-fx-vert" style="width:100%;" id="metric_table_fx_vert">
    <thead>
      <tr class="header">
        <th class="sortable" style="text-align: left;" 
            onclick="sortTable('metric_table_fx_vert', 0, 0, false)">
          Forecast
        </th>
          <th class="sortable" style="text-align: left;" 
              onclick="sortTable('metric_table_fx_vert', 1)">
            MAE
          </th>
          <th class="sortable" style="text-align: left;" 
              onclick="sortTable('metric_table_fx_vert', 2)">
            RMSE
          </th>
          <th class="sortable" style="text-align: left;" 
              onclick="sortTable('metric_table_fx_vert', 3)">
            MBE
          </th>
          <th class="sortable" style="text-align: left;" 
              onclick="sortTable('metric_table_fx_vert', 4)">
            Skill
          </th>
          <th class="sortable" style="text-align: left;" 
              onclick="sortTable('metric_table_fx_vert', 5)">
            Cost
          </th>
      </tr>
    </thead>
    <tbody>
        <tr>
          <td>{}</td>
                  <td>2</td>
                  <td>2</td>
                  <td>2</td>
                  <td>2</td>
                  <td>2</td>
        </tr>
    </tbody>
  </table>
  </div>
</details>
"""


def test_metric_table_fx_vert(report_with_raw, macro_test_template):
    metric_table_template = macro_test_template('metric_table_fx_vert(report_metrics,category,metric_ordering)')  # noqa
    metrics = [m for m in report_with_raw.raw_report.metrics
               if not m.is_summary]
    category = 'total'
    for i in range(3):
        expected_metric = (metrics[i],)
        rendered_metric_table = metric_table_template.render(
            report_metrics=expected_metric,
            category=category,
            metric_ordering=report_with_raw.report_parameters.metrics,
            human_metrics=datamodel.ALLOWED_METRICS)
        assert rendered_metric_table == metric_table_fx_vert_format.format(
            category.lower(), expected_metric[0].name)


metric_table_fx_horz_format = """<table class="table table-striped" style="width:100%;">
  <caption style="caption-side:top; text-align:center">
    Table of {0} metrics
  </caption>
  <thead>
    <tr class="header">
      <th></th>
        <th colspan="{1}" style="text-align: center;">{2}</th>
    </tr>
    <tr class="header">
      <th style="text-align: left;">{0} Value</th>
          <th style="test-align: center;">MAE</th>
          <th style="test-align: center;">RMSE</th>
          <th style="test-align: center;">MBE</th>
          <th style="test-align: center;">Skill</th>
          <th style="test-align: center;">Cost</th>
    </tr>
  </thead>
  <tbody>
      <tr>
        <td>1</td>
                  <td>2</td>
                  <td>2</td>
                  <td>2</td>
                  <td>2</td>
                  <td>2</td>
      </tr>
  </tbody>
</table>
"""


def test_metric_table_fx_horz(report_with_raw, macro_test_template):
    metric_table_template = macro_test_template('metric_table_fx_horz(report_metrics,category,metric_ordering)')  # noqa
    metrics = [m for m in report_with_raw.raw_report.metrics
               if not m.is_summary]
    category = 'hour'
    for i in range(3):
        expected_metric = (metrics[i],)
        rendered_metric_table = metric_table_template.render(
            report_metrics=expected_metric,
            category=category,
            metric_ordering=report_with_raw.report_parameters.metrics,
            human_metrics=datamodel.ALLOWED_METRICS)
        assert rendered_metric_table == metric_table_fx_horz_format.format(
            category, 5, expected_metric[0].name)


validation_table_format = """<details>
  <summary>
    <h4>Table of data validation results {} resampling</h4>
  </summary>
  <div class="report-table-wrapper">
  <table class="table table-striped validation-table" style="width:100%;" id="data-validation-results-{}-table">
    <thead>
      <tr class="header">
        <th class="sortable" style="text-align: left;"
            onclick="sortTable('data-validation-results-before-table', 0, 1, false)">
          Aligned Pair
        </th>
        <th class="sortable" style="text-align: center; vertical-align: middle"
            onclick="sortTable('data-validation-results-{}-table', 1, 1)">
          {}
        </th>
        <th class="sortable" style="text-align: center; vertical-align: middle"
            onclick="sortTable('data-validation-results-{}-table', 2, 1)">
          {}
        </th>
      </tr>
      <tr class="header">
        <th class="sortable" style="text-align: left;"
            onclick="sortTable('data-validation-results-before-table', 0, 1, false)">
          Observation
        </th>
        <th class="sortable" style="text-align: center; vertical-align: middle"
            onclick="sortTable('data-validation-results-{}-table', 1, 1)">
          {}
        </th>
        <th class="sortable" style="text-align: center; vertical-align: middle"
            onclick="sortTable('data-validation-results-{}-table', 2, 1)">
          {}
        </th>
      </tr>
    </thead>
    <tbody>
        <tr>
          <td style="text-align: left">{}</td>
          <td style="text-align: center">0</td>
          <td style="text-align: center">0</td>
        </tr>
        <tr>
          <td style="text-align: left">{}</td>
          <td style="text-align: center">0</td>
          <td style="text-align: center">0</td>
        </tr>
        <tr>
          <td style="text-align: left">{}</td>
          <td style="text-align: center">0</td>
          <td style="text-align: center">0</td>
        </tr>
        <tr>
          <td style="text-align: left">{}</td>
          <td style="text-align: center">0</td>
          <td style="text-align: center">0</td>
        </tr>
        <tr>
          <td style="text-align: left">{}</td>
          <td style="text-align: center">0</td>
          <td style="text-align: center">0</td>
        </tr>
        <tr>
          <td style="text-align: left">{}</td>
          <td style="text-align: center">0</td>
          <td style="text-align: center">0</td>
        </tr>
    </tbody>
  </table>
  </div>
</details>
"""  # noqa: E501


def test_validation_results_table(report_with_raw, macro_test_template):
    validation_table_template = macro_test_template('validation_results_table(proc_fxobs_list, true)')  # noqa
    proc_fxobs_list = report_with_raw.raw_report.processed_forecasts_observations[0:2]  # noqa
    qfilters = list(
        f.quality_flags for f in report_with_raw.report_parameters.filters
        if isinstance(f, datamodel.QualityFlagFilter))[0]
    rendered_validation_table = validation_table_template.render(
        proc_fxobs_list=proc_fxobs_list, before_resample=True)
    expected = validation_table_format.format(
        'before', 'before', 'before',
        proc_fxobs_list[0].name, 'before',
        proc_fxobs_list[1].name, 'before',
        proc_fxobs_list[0].original.observation.name, 'before',
        proc_fxobs_list[1].original.observation.name,
        *qfilters)
    assert rendered_validation_table == expected


preprocessing_table_format = """<details open="">
  <summary>
    <h4>Table of data preprocessing results</h4>
  </summary>
  <div class="report-table-wrapper">
  <table class="anchor table table-striped preprocessing-table" style="width:100%;" id="data-preprocessing-results-table">
    <thead>
      <tr class="header">
        <th class="sortable" style="text-align: left;" onclick="sortTable('data-preprocessing-results-table', 0, 0, false)">
          Preprocessing Description
        </th>
        <th class="sortable" style="text-align: center;" onclick="sortTable('data-preprocessing-results-table', 1, 0, true)">
          {} <br>Number of Points
        </th>
        <th class="sortable" style="text-align: center;" onclick="sortTable('data-preprocessing-results-table', 2, 0, true)">
          {} <br>Number of Points
        </th>
      </tr>
    </thead>
    <tbody>
        <tr>
        <td style="test-align: left">{}</td>
              <td style="text-align: center">0</td>
              <td style="text-align: center">0</td>
        </tr>
        <tr>
        <td style="test-align: left">{}</td>
              <td style="text-align: center">0</td>
              <td style="text-align: center">0</td>
        </tr>
        <tr>
        <td style="test-align: left">{}</td>
              <td style="text-align: center">0</td>
              <td style="text-align: center">0</td>
        </tr>
    </tbody>
  </table>
  </div>
</details>
"""  # noqa: E501


def test_preprocessing_table(report_with_raw, macro_test_template,
                             preprocessing_result_types):
    preprocessing_table_template = macro_test_template('preprocessing_table(proc_fxobs_list)')  # noqa
    proc_fxobs_list = report_with_raw.raw_report.processed_forecasts_observations[0:2]  # noqa
    rendered_preprocessing_table = preprocessing_table_template.render(
        proc_fxobs_list=proc_fxobs_list)
    assert rendered_preprocessing_table == preprocessing_table_format.format(
        proc_fxobs_list[0].name,
        proc_fxobs_list[1].name,
        *preprocessing_result_types)


summary_stats_table_vert_format = """<details>
  <summary>
    <h4>Table of {stat} data summary statistics</h4>
  </summary>
  <div class="report-table-wrapper">
  <table class="table table-striped table-bordered summary-stats-table-vert" style="width:100%;"
      id="summary_stats_table_vert">
    <thead>
      <tr class="header">
        <th scope="col" class="sortable" style="text-align: left;"
          onclick="sortTable('summary_stats_table_vert', 0, 1, false)">
          Aligned Pair
        </th>
        <th scope="col" colspan="5" style="text-align: center;">Observation</th>
        <th scope="col" colspan="5" style="text-align: center;">Forecast</th>
        <th scope="col" colspan="5" style="text-align: center;">Reference Forecast</th>
      </tr>
      <tr>
        <th></th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              1, 1, true)">
          Mean
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              2, 1, true)">
          Min
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              3, 1, true)">
          Max
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              4, 1, true)">
          Median
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              5, 1, true)">
          Std.
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              6, 1, true)">
          Mean
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              7, 1, true)">
          Min
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              8, 1, true)">
          Max
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              9, 1, true)">
          Median
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              10, 1, true)">
          Std.
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              11, 1, true)">
          Mean
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              12, 1, true)">
          Min
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              13, 1, true)">
          Max
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              14, 1, true)">
          Median
        </th>
        <th class="sortable" style="text-align: center;"
            onclick="sortTable('summary_stats_table_vert',
              15, 1, true)">
          Std.
        </th>
      </tr>
    </thead>
    <tbody>
        <tr>
          <td>{name}</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>{ref}</td>
          <td>{ref}</td>
          <td>{ref}</td>
          <td>{ref}</td>
          <td>{ref}</td>
        </tr>
    </tbody>
  </table>
  </div>
</details>
"""  # NOQA


def test_summary_stats_table_vert(report_with_raw, macro_test_template):
    stats_table_template = macro_test_template('summary_stats_table_vert(report_metrics,category)')  # noqa
    metrics = [m for m in report_with_raw.raw_report.metrics
               if m.is_summary]
    category = 'total'
    human_categories = datamodel.ALLOWED_CATEGORIES
    for i in range(3):
        expected_metric = (metrics[i],)
        rendered_stats_table = stats_table_template.render(
            report_metrics=expected_metric,
            category=category,
            human_categories=human_categories,
            human_statistics=datamodel.ALLOWED_DETERMINISTIC_SUMMARY_STATISTICS
            )
        exp = summary_stats_table_vert_format.format(
            stat=human_categories[category].lower(),
            name=expected_metric[0].name,
            ref='1' if i == 1 else '')
        assert rendered_stats_table == exp
