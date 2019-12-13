"""
Inserts metadata and figures into the report template.
"""
import logging


from bokeh.embed import components
from bokeh.layouts import gridplot
from jinja2 import (Environment, PackageLoader,
                    select_autoescape)

from solarforecastarbiter import datamodel
from solarforecastarbiter.reports import figures


logger = logging.getLogger(__name__)


def _get_render_kwargs(report, dash_url, with_timeseries):
    kwargs = dict(
        human_categories=datamodel.ALLOWED_CATEGORIES,
        report=report,
        dash_url=dash_url,
        bokeh_version=report.raw_report.plots.bokeh_version
    )
    if with_timeseries:
        fx_obs_cds = [
            (pfxobs, figures.construct_fx_obs_cds(
                pfxobs.forecast_values, pfxobs.observation_values))
            for pfxobs in report.raw_report.processed_forecasts_observations]
        ts_fig = figures.timeseries(
            fx_obs_cds, report.raw_report.metadata.start,
            report.raw_report.metadata.end,
            timezone=report.raw_report.metadata.timezone)
        scat_fig = figures.scatter(fx_obs_cds)
        try:
            script, div = components(gridplot((ts_fig, scat_fig), ncols=1))
        except Exception:
            logger.exception(
                'Failed to make Bokeh items for timeseries and scatterplot')
            script = ''
            div = """<div class="alert alert-warning">
  <strong>Warning</strong> Failed to make timeseries and scatter figure
  from stored data. Try generating report again.
</div>
            """
        kwargs['timeseries_script'] = script
        kwargs['timeseries_div'] = div
    return kwargs


def get_template_and_kwargs(report, dash_url, with_timeseries, body_only):
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
    body : html

    Returns
    -------
        head : str, html
        Header for the full report.
    """
    template, kwargs = get_template_and_kwargs(
        report, dash_url, with_timeseries, body_only)
    out = template.render(**kwargs)
    return out
