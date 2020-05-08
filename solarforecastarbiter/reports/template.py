"""
Inserts metadata and figures into the report template.
"""
import base64
import logging
from pathlib import Path
import re
import subprocess
import tempfile


from bokeh import __version__ as bokeh_version
from jinja2 import Environment, PackageLoader, select_autoescape, ChoiceLoader
from plotly import __version__ as plotly_version


from solarforecastarbiter import datamodel
from solarforecastarbiter.reports.figures import plotly_figures


logger = logging.getLogger(__name__)


def build_metrics_json(report):
    """Creates a dict from the metrics results in the report.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    str
        The json representing the report metrics. The string will be a string
        representing an empty json array if the report does not have a
        computed raw_report.
    """
    if getattr(report, 'raw_report') is not None:
        df = plotly_figures.construct_metrics_dataframe(
            report.raw_report.metrics,
            rename=plotly_figures.abbreviate)
        return df.to_json(orient="records")
    else:
        return "[]"


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
        metrics_json=build_metrics_json(report),
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
            timeseries_specs = plotly_figures.timeseries_plots(report)
        except Exception:
            logger.exception(
                'Failed to make Plotly items for timeseries and scatterplot')
            timeseries_specs = ('{}', '{}')
        kwargs['timeseries_spec'] = timeseries_specs[0]
        kwargs['scatter_spec'] = timeseries_specs[1]

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
        loader=ChoiceLoader([
            PackageLoader('solarforecastarbiter.reports', 'templates/html'),
            PackageLoader('solarforecastarbiter.reports', 'templates'),
        ]),
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


def html_to_tex(value):
    value = (value
             .replace('<p>', '')
             .replace('</p>', '\n')
             .replace('<em>', '\\emph{')
             .replace('</em>', '}')
             .replace('<code>', '\\verb|')
             .replace('</code>', '|')
             .replace('<b>', '\\textbf{')
             .replace('</b>', '}')
    )
    return value


def render_pdf(report, dash_url, with_timeseries=True, max_runs=5):
    env = Environment(
        loader=ChoiceLoader([
            PackageLoader('solarforecastarbiter.reports', 'templates/pdf'),
            PackageLoader('solarforecastarbiter.reports', 'templates'),
        ]),
        autoescape=False,
        lstrip_blocks=True,
        trim_blocks=True,
        block_start_string='\\BLOCK{',
        block_end_string='}',
        variable_start_string='\\VAR{',
        variable_end_string='}',
        comment_start_string='\\#{',
        comment_end_string='}',
        line_statement_prefix='%-',
        line_comment_prefix='%#'
    )
    env.filters['html_to_tex'] = html_to_tex
    kwargs = _get_render_kwargs(report, dash_url, with_timeseries)
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        template = env.get_template('base.tex')
        tex = template.render(**kwargs)
        texfile = tmpdir / 'out.tex'
        texfile.write_text(tex)
        auxfile = tmpdir / 'out.aux'
        logfile = tmpdir / 'out.log'

        # save figures
        figdir = tmpdir / 'figs'
        figdir.mkdir()
        for fig in report.raw_report.plots.figures:
            name = (
                fig.category + '_' + fig.metric + '_' +
                fig.name.replace('^', '-').replace(' ', '_') +
                '.pdf'
            )
            figpath = figdir / name
            figpath.write_bytes(base64.a85decode(fig.pdf))

        args = (
            'pdflatex',
            '-interaction=batchmode',
            '-halt-on-error',
            '-no-shell-escape',
            '-file-line-error',
            'out.tex'
        )
        runs_left = max_runs
        prev_aux = ''
        while runs_left > 0:
            try:
                subprocess.run(args, check=True, cwd=str(tmpdir.absolute()))
            except subprocess.CalledProcessError:
                logger.exception(logfile.read_text())
                raise

            aux = auxfile.read_text()
            if aux == prev_aux:
                break
            else:
                prev_aux = aux
                runs_left -= 1
        else:
            raise RuntimeError(
                f'PDF generation unstable after {max_runs} runs')

        return (tmpdir / 'out.pdf').read_bytes()
