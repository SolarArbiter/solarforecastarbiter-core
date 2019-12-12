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


def full_html(report, dash_url='https://dashboard.solarforecastarbiter.org'):
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
        autoescape=select_autoescape(['html', 'xml']),
        lstrip_blocks=True,
        trim_blocks=True
    )
    base_template = env.get_template('base.html')

    out = base_template.render(
        human_categories=datamodel.ALLOWED_CATEGORIES,
        report=report,
        dash_url=dash_url,
        bokeh_version=report.raw_report.plots.bokeh_version)
    return out


def render_body(report, dash_url=''):
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml']),
        lstrip_blocks=True,
        trim_blocks=True
    )
    body_template = env.get_template('body.html')

    body = body_template.render(
        human_categories=datamodel.ALLOWED_CATEGORIES,
        report=report,
        dash_url=dash_url)
    return body
