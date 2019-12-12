"""
Inserts metadata and figures into the report template.
"""
from collections import defaultdict
import logging
import subprocess
import tempfile


from bokeh.embed import components
from bokeh.io import export_svgs
from bokeh.layouts import gridplot
from jinja2 import (Environment, DebugUndefined, PackageLoader,
                    select_autoescape, Template)

from solarforecastarbiter import datamodel
from solarforecastarbiter.reports import figures


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

    script_metrics, figures_div = _metrics_script_divs(
        report, metrics)

    strftime = '%Y-%m-%d %H:%M:%S %z'

    def route_id(fx_ob):
        if isinstance(fx_ob.original, datamodel.ForecastObservation):
            return 'observations', fx_ob.original.observation.observation_id
        else:
            return 'aggregates', fx_ob.original.aggregate.aggregate_id

    proc_fx_obs = [
        (fx_ob, *route_id(fx_ob)) for fx_ob in processed_forecasts_observations
    ]
    rendered = template.render(
        name=metadata.name,
        start=metadata.start.strftime(strftime),
        end=metadata.end.strftime(strftime),
        now=metadata.now.strftime(strftime),
        proc_fx_obs=proc_fx_obs,
        validation_issues=metadata.validation_issues,
        versions=metadata.versions,
        script_metrics=script_metrics,
        figures=figures_div,
        metrics_toc=datamodel.ALLOWED_CATEGORIES)

    return rendered


def _metrics_script_divs(report, metrics):
    cds = figures.construct_metrics_cds(metrics,
                                        rename=figures.abbreviate)

    # Create initial bar figures
    figure_dict = {}
    # Components for other metrics
    for category in report.categories:
        for metric in report.metrics:
            if category == 'total':
                fig = figures.bar(cds, metric)
                figure_dict[f'total_{metric}'] = fig
            else:
                figs = figures.bar_subdivisions(cds, category, metric)
                for name, fig in figs.items():
                    figure_dict[f'{category}_{metric}_{name}'] = fig
    script, divs = components(figure_dict)
    out_divs = defaultdict(list)
    for k, v in divs.items():
        cat = k.split('_')[0]
        out_divs[cat].append(v)
    # make svg
    # return rawreportplots
    return script, out_divs


def raw_report_plots(report, metrics):
    cds = figures.construct_metrics_cds(metrics,
                                        rename=figures.abbreviate)

    # Create initial bar figures
    figure_dict = {}
    # Components for other metrics
    for category in report.categories:
        for metric in report.metrics:
            if category == 'total':
                fig = figures.bar(cds, metric)
                figure_dict[f'total_{metric}_all'] = fig
            else:
                figs = figures.bar_subdivisions(cds, category, metric)
                for name, fig in figs.items():
                    figure_dict[f'{category}_{metric}_{name}'] = fig
    script, divs = components(figure_dict)
    mplots = []
    for k, v in divs.items():
        cat, met, name = k.split('_')
        fig = figure_dict[k]
        fig.output_backend = 'svg'
        with tempfile.NamedTemporaryFile() as tmpfile:
            export_svgs(fig, filename=tmpfile.name)
            tmpfile.flush()
            tmpfile.seek(0)
            svg = tmpfile.read().decode()
        mplots.append(datamodel.MetricPlot(name, cat, met, v, svg))
    out = datamodel.RawReportPlots(script, tuple(mplots))
    return out


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
