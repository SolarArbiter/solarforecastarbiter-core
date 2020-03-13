"""
Inserts metadata and figures into the report template.
"""
import logging


from bokeh import __version__ as bokeh_version
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly import __version__ as plotly_version
from solarforecastarbiter import datamodel
from solarforecastarbiter.reports.figures import bokeh_figures


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
    )
    report_plots = getattr(report.raw_report, 'plots', None)

    # get plotting library versions used when plots were generated.
    # if plot generation failed, fallback to the curent version
    plot_bokeh = getattr(report_plots, 'bokeh_version', None)
    kwargs['bokeh_version'] = plot_bokeh if plot_bokeh else bokeh_version

    plot_plotly = getattr(report_plots, 'plotly_version', None)
    kwargs['plotly_version'] = plot_plotly if plot_plotly else plotly_version
    if with_timeseries:
        try:
            script, div = bokeh_figures.timeseries_plots(report)
        except Exception:
            logger.exception(
                'Failed to make Bokeh items for timeseries and scatterplot')
            script = ''
            div = """<div class="alert alert-warning">
  <strong>Warning</strong> Failed to make timeseries and scatter plots
  from stored data.
</div>"""
        kwargs['timeseries_script'] = script
        kwargs['timeseries_div'] = div
    return kwargs


def get_template_and_kwargs(report, dash_url, with_timeseries, body_only):
    """Returns the jinja2 Template object and a dict of template variables for
    the report. If the report failed to compute, the template and kwargs will
    be for an error page.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`
    dash_url: str
        URL of the Solar Forecast arbiter dashboard to use when building links.
    with_timeseries: bool
        Whether or not to include timeseries plots.
    body_only: bool
        When True, returns a div for injecting into another template,
        otherwise returns a full html document with the required
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
    kwargs = _get_render_kwargs(report, dash_url, with_timeseries)
    if report.status == 'complete':
        template = env.get_template('body.html')
    elif report.status == 'failed':
        template = env.get_template('failure.html')
    elif report.status == 'pending':
        template = env.get_template('pending.html')
    else:
        raise ValueError(f'Unknown status for report {report.status}')

    if body_only:
        kwargs['base_template'] = env.get_template('empty_base.html')
    else:
        kwargs['base_template'] = env.get_template('base.html')
    return template, kwargs


def render_html(report, dash_url=datamodel.DASH_URL,
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
        otherwise returns a full html document with the required
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
