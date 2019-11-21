"""
Inserts metadata and figures into the report template.
"""
import logging
import subprocess

from bokeh.embed import components
from bokeh.layouts import gridplot
from jinja2 import (Environment, DebugUndefined, PackageLoader,
                    select_autoescape, Template)


from solarforecastarbiter.reports import figures
from solarforecastarbiter.metrics import calculator


logger = logging.getLogger(__name__)


def template_report(report, metadata, metrics,
                    processed_forecasts_observations):
    """
    Render the markdown report template. Figures are left untemplated.

    Parameters
    ----------
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    metadata : solarforecastarbiter.datamodel.ReportMetadata
        Describes the pre-report
    metrics : tuple of dict
    processed_forecasts_observations : tuple of solarforecastarbiter.datamodel.ProcessedForecastObservation

    Returns
    -------
    markdown
    """  # noqa
    # By default, jinja removes undefined variables from the rendered string.
    # DebugUndefined leaves undefined variables in the string so that they
    # can be used in the full report template process.
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml']),
        undefined=DebugUndefined)

    template = env.get_template('template.md')

    script_metrics, data_table_div, figures_div = _metrics_script_divs(
        report, metrics)

    strftime = '%Y-%m-%d %H:%M:%S %z'

    rendered = template.render(
        name=metadata.name,
        start=metadata.start.strftime(strftime),
        end=metadata.end.strftime(strftime),
        now=metadata.now.strftime(strftime),
        proc_fx_obs=processed_forecasts_observations,
        validation_issues=metadata.validation_issues,
        versions=metadata.versions,
        script_metrics=script_metrics,
        tables=data_table_div,
        figures=figures_div,
        metrics_toc=calculator.AVAILABLE_CATEGORIES)

    return rendered


def _metrics_script_divs(report, metrics):
    cds = figures.construct_metrics_cds(metrics, 'Total', index='forecast',
                                        rename=figures.abbreviate)
    data_table = figures.metrics_table(cds)

    # Create initial bar figures
    figures_bar = []
    for num, metric in enumerate(report.metrics):
        fig = figures.bar(cds, metric)
        figures_bar.append(fig)

    # Components for 'Total' category
    script, (data_table_div, *figures_bar_divs) = components((data_table,
                                                              *figures_bar))

    script_metrics = script
    figures_dict = dict(Total=figures_bar_divs)

    # Components for other metrics
    for category in report.categories:
        if category == 'Total':
            continue

        script_cat, figures_bar_cat = _loop_over_metrics(report, metrics,
                                                         category)
        script_metrics += script_cat
        figures_dict[category] = figures_bar_cat

    return script_metrics, data_table_div, figures_dict


def _loop_over_metrics(report, metrics, kind):
    figs = []
    # series with MultiIndex of metric, forecast, day
    # JSON serialization issues if we don't drop or fill na.
    # fillna ensures 0 - 23 hrs on hourly plots.
    metrics_series = figures.construct_metrics_series(metrics, kind).fillna(0)
    for metric in report.metrics:
        cds = figures.construct_metrics_cds2(metrics_series, metric)
        # one figure with a subfig for each forecast
        fig = figures.bar_subdivisions(cds, kind, metric)
        figs.append(gridplot(fig, ncols=1))
    script, divs = components(figs)
    return script, divs


# not all args currently used, but expect they will eventually be used
def add_figures_to_report_template(fx_obs_cds, metadata, report_template,
                                   html=True):
    """
    Add figures to the report_template

    Parameters
    ----------
    fx_obs_cds : list
        List of (ProcessedForecastObservation, ColumnDataSource)
        tuples to pass to bokeh plotting objects.
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    report_template : str, markdown
        The templated report
    html : bool
        Indicates if the template will be rendered into html or pdf.

    Returns
    -------
    body : str, markdown
    """
    body_template = Template(report_template)

    ts_fig = figures.timeseries(fx_obs_cds, metadata.start, metadata.end,
                                timezone=metadata.timezone)
    scat_fig = figures.scatter(fx_obs_cds)
    try:
        script, div = components(gridplot((ts_fig, scat_fig), ncols=1))
    except Exception:
        logger.exception(
            'Failed to make Bokeh items for timeseries and scatterplot')
        script = ''
        div = """
::: warning
Failed to make timeseries and scatter figure from stored data. Try
generating report again.
:::
"""

    body = body_template.render(
        script_data=script,
        html=html,
        figures_timeseries_scatter=div)
    return body


def report_md_to_html(report_md):
    """
    Render markdown into simple html using pandoc.

    Parameters
    ----------
    report_md : str, markdown

    Returns
    -------
    str, html
    """
    try:
        out = subprocess.run(args=['pandoc', '--from', 'markdown+pipe_tables'],
                             input=report_md.encode(), capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise OSError('Error converting prereport to html using pandoc') from e
    return out.stdout.decode()


def full_html(body):
    """Create full html file.

    The Solar Forecast Arbiter dashboard will likely use its own
    templates for rendering the full html.

    Parameters
    ----------
    body : html

    Returns
    -------
        head : str, html
        Header for the full report.
    """
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml']))
    base_template = env.get_template('base.html')

    base = base_template.render(body=body)

    return base
