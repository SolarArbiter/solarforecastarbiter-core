"""
Inserts metadata and figures into the report template.
"""
import logging


from jinja2 import (Environment, PackageLoader,
                    select_autoescape)

from solarforecastarbiter import datamodel
from solarforecastarbiter.reports import figures


logger = logging.getLogger(__name__)


def _get_render_kwargs(report, dash_url, with_timeseries):
    """Creates a dictionary of key word template arguments for a jinja2
    report template.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`
    dash_url: str
        URL of the Solar Forecast arbiter dashboard to use when building links.
    with_timeseries: bool
        Whether or not to include timeseries plots. If an error occurs, sets
        `timeseries_script` to an empty string and `timeseries_div` will
        contain a <div> element for warning the user of the failure.

    Returns
    -------
    kwargs: dict
        Dictionary of template variables to unpack as key word arguments when
        rendering.
    """
    kwargs = dict(
        human_categories=datamodel.ALLOWED_CATEGORIES,
        human_metrics=datamodel.ALLOWED_METRICS,
        report=report,
        category_blurbs=datamodel.CATEGORY_BLURBS,
        dash_url=dash_url,
        bokeh_version=report.raw_report.plots.bokeh_version
    )
    if with_timeseries:
        try:
            script, div = figures.timeseries_plots(report)
        except Exception:
            logger.exception(
                'Failed to make Bokeh items for timeseries and scatterplot')
            script = ''
            div = """<div class="alert alert-warning">
  <strong>Warning</strong> Failed to make timeseries and scatter plots
  from stored data. Try generating report again.
</div>"""
        kwargs['timeseries_script'] = script
        kwargs['timeseries_div'] = div
    return kwargs


def get_template_and_kwargs(report, dash_url, with_timeseries, body_only):
    """Returns the jinja2 Template object and a dict of template variables for
    the report.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`
    dash_url: str
        URL of the Solar Forecast arbiter dashboard to use when building links.
    with_timeseries: bool
        Whether or not to include timeseries plots.
    body_only: bool
        When True, returns a div for injecting into another template,
        otherwise returns a standalone report template with the required
        <html> and <head> tags.

    Returns
    -------
    template: jinja2.environment.Template
    kwargs: dict
        Dictionary of template variables to use as keyword arguments to
        template.render().
    """
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml']),
        lstrip_blocks=True,
        trim_blocks=True
    )
    if body_only:
        template = env.get_template('body.html')
    else:
        template = env.get_template('base.html')
    kwargs = _get_render_kwargs(report, dash_url, with_timeseries)
    return template, kwargs


def render_html(report, dash_url='https://dashboard.solarforecastarbiter.org',
                with_timeseries=True, body_only=False):
    """Create full html file.

    The Solar Forecast Arbiter dashboard will likely use its own
    templates for rendering the full html.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`
    dash_url: str
        URL of the Solar Forecast arbiter dashboard to use when building links.
    with_timeseries: bool
        Whether or not to include timeseries plots.
    body_only: bool
        When True, returns a div for injecting into another template,
        otherwise returns a standalone report template with the required
        <html> and <head> tags.

    Returns
    -------
    str
        The rendered html report
    """
    template, kwargs = get_template_and_kwargs(
        report, dash_url, with_timeseries, body_only)
    out = template.render(**kwargs)
    return out
