"""
Inserts metadata and figures into the report template.
"""
import base64
import json
import logging
from pathlib import Path
import re
import subprocess
import tempfile


from bokeh import __version__ as bokeh_version
from jinja2 import Environment, PackageLoader, select_autoescape, ChoiceLoader
from jinja2.runtime import Undefined
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
            list(filter(lambda x: not getattr(x, 'is_summary', False),
                 report.raw_report.metrics)),
            rename=plotly_figures.abbreviate)
        return df.to_json(orient="records")
    else:
        return "[]"


def build_summary_stats_json(report):
    """Creates a dict from the summary statistics in the report.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    str
        The json representing the summary statistics. Will be a string
        representing an empty json array if the report does not have a
        computed raw_report.

    Raises
    ------
    ValueError
        If report.raw_report is populated but no
        report.raw_report.metrics have `is_summary == True`
        indicating that the report was made without
        summary statistics.
    """
    if getattr(report, 'raw_report') is not None:
        df = plotly_figures.construct_metrics_dataframe(
            list(filter(lambda x: getattr(x, 'is_summary', False),
                 report.raw_report.metrics)),
            rename=plotly_figures.abbreviate)
        if df.empty:
            raise ValueError('No summary statistics in report.')
        return df.to_json(orient="records")
    else:
        return "[]"


def build_metadata_json(report):
    """Creates a JSON array of ProcessedForecastObservations parameters
    in the report.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    str
        The JSON representing the report forecast-observation metadata.
    """
    if getattr(report, 'raw_report') is None:
        return "[]"

    drop_keys = {
        '__blurb__', 'site', 'aggregate',
    }

    def _process_forecast(fx):
        if fx is None:
            return None
        out = {k: v for k, v in fx.to_dict().items()
               if k not in drop_keys}
        if isinstance(fx, datamodel.ProbabilisticForecast):
            out['constant_values'] = [
                cdf.constant_value for cdf in fx.constant_values]
        return out

    out = []
    for pfxobs in report.raw_report.processed_forecasts_observations:
        minp = pfxobs.replace(original=None)
        thisout = {k: v for k, v in minp.to_dict().items()
                   if k in (
                           'name', 'interval_value_type', 'interval_length',
                           'interval_label', 'normalization_factor',
                           'uncertainty', 'cost')}

        thisout['forecast'] = _process_forecast(pfxobs.original.forecast)
        thisout['reference_forecast'] = _process_forecast(
            pfxobs.original.reference_forecast)
        thisout['observation'] = None
        thisout['aggregate'] = None
        if hasattr(pfxobs.original, 'observation'):
            thisout['observation'] = {
                k: v for k, v in pfxobs.original.observation.to_dict().items()
                if k not in drop_keys
            }
        elif hasattr(pfxobs.original, 'aggregate'):
            thisout['aggregate'] = {
                k: v for k, v in pfxobs.original.aggregate.to_dict().items()
                if k not in drop_keys or k == 'observations'
            }
            obs = []
            for aggobs in pfxobs.original.aggregate.observations:
                obsd = aggobs.to_dict()
                obsd['observation_id'] = obsd.pop('observation')[
                    'observation_id']
                obs.append(obsd)
            thisout['aggregate']['observations'] = obs
        out.append(thisout)
    return json.dumps(out).replace('NaN', 'null')


def _get_render_kwargs(report, dash_url, with_timeseries):
    """Creates a dictionary of key word template arguments for a jinja2
    report template.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`
    dash_url: str
        URL of the Solar Forecast arbiter dashboard to use when building links.
    with_timeseries: bool
        Whether or not to include timeseries plots. If an error occurs when
        trying to generate timeseries plots, the `timeseries_spec`,
        `scatter_spec`, and `timeseries_prob_spec` arguments will not be
        defined.

    Returns
    -------
    kwargs: dict
        Dictionary of template variables to unpack as key word arguments when
        rendering.
    """
    kwargs = dict(
        human_categories=datamodel.ALLOWED_CATEGORIES,
        human_metrics=datamodel.ALLOWED_METRICS,
        human_statistics=datamodel.ALLOWED_SUMMARY_STATISTICS,
        report=report,
        category_blurbs=datamodel.CATEGORY_BLURBS,
        dash_url=dash_url,
        metrics_json=build_metrics_json(report),
        metadata_json=build_metadata_json(report),
        templating_messages=[]
    )
    report_plots = getattr(report.raw_report, 'plots', None)
    # get plotting library versions used when plots were generated.
    # if plot generation failed, fallback to the curent version
    plot_bokeh = getattr(report_plots, 'bokeh_version', None)
    kwargs['bokeh_version'] = plot_bokeh if plot_bokeh else bokeh_version

    plot_plotly = getattr(report_plots, 'plotly_version', None)
    kwargs['plotly_version'] = plot_plotly if plot_plotly else plotly_version

    try:
        kwargs['summary_stats'] = build_summary_stats_json(report)
    except ValueError:
        kwargs['templating_messages'].append(
            'No data summary statistics were calculated with this report.')
        kwargs['summary_stats'] = '[]'

    if with_timeseries:
        try:
            timeseries_specs = plotly_figures.timeseries_plots(report)
        except Exception:
            logger.exception(
                'Failed to make Plotly items for timeseries and scatterplot')
        else:
            if timeseries_specs[0] is not None:
                kwargs['timeseries_spec'] = timeseries_specs[0]

            if timeseries_specs[1] is not None:
                kwargs['scatter_spec'] = timeseries_specs[1]

            if timeseries_specs[2] is not None:
                kwargs['timeseries_prob_spec'] = timeseries_specs[2]

            kwargs['includes_distribution'] = timeseries_specs[3]

    return kwargs


def _pretty_json(value):
    if isinstance(value, Undefined):  # pragma: no cover
        return value
    return json.dumps(value, indent=4, separators=(',', ':'))


def _figure_name_filter(value):
    """replace characters that may cause problems for html/javascript ids"""
    if isinstance(value, Undefined):
        return value
    out = (value
           .replace('^', '-')
           .replace(' ', '-')
           .replace('.', 'dot')
           .replace('%', 'percent')
           .replace('<', 'lt')
           .replace('>', 'gt')
           .replace('=', 'eq')
           .replace('(', 'lp')
           .replace(')', 'rp')
           .replace('/', 'fsl')
           .replace('\\', 'bsl')
           )
    out = re.sub('[^\\w-]', 'special', out)
    return out


def _unique_flags_filter(proc_fxobs_list, before_resample):
    # use a dict to preserve order and guarantee uniqueness of keys
    names = {}
    for proc_fxobs in proc_fxobs_list:
        for val_result in proc_fxobs.validation_results:
            if val_result.before_resample == before_resample:
                names[val_result.flag] = None
    unique_names = list(names.keys())
    return unique_names


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
    env.filters['pretty_json'] = _pretty_json
    env.filters['figure_name_filter'] = _figure_name_filter
    env.filters['unique_flags_filter'] = _unique_flags_filter
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


def _link_filter(value):
    """convert html href markup to tex href markup"""
    if isinstance(value, Undefined):  # pragma: no cover
        return value
    match = re.search(
        """<a\\s+(?:[^>]*?\\s+)?href=(["'])(.*?)(["'])>(.*?)<\\/a>""",
        value, re.DOTALL)
    if match:
        new = "\\href{" + match.group(2) + "}{" + match.group(4) + "}"
        out = value[:match.start()] + new + value[match.end():]
        return out
    else:
        return value


def _html_to_tex(value):
    if isinstance(value, Undefined):
        return value
    value = (value
             .replace('<p>', '')
             .replace('</p>', '\n')
             .replace('<em>', '\\emph{')
             .replace('</em>', '}')
             .replace('<code>', '\\verb|')
             .replace('</code>', '|')
             .replace('<b>', '\\textbf{')
             .replace('</b>', '}')
             .replace('<ol>', '\\begin{enumerate}')
             .replace('</ol>', '\\end{enumerate}')
             .replace('<li>', '\\item ')
             .replace('</li>', '\n')
             .replace('</a>', '')
             .replace('<=', '$\\leq$')
             .replace("%", "\\%")
             .replace('W/m^2', '$W/m^2$')
             )
    value = re.sub('\\<a.*\\>', '', value)
    return value


def render_pdf(report, dash_url, max_runs=5):
    """
    Create a PDF report using LaTeX.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`
    dash_url: str
        URL of the Solar Forecast Arbiter dashboard to use when building links.
    max_runs: int, default 5
        Maximum number of times to run pdflatex

    Returns
    -------
    bytes
        The rendered PDF report

    Notes
    -----
    This code was inspired by the latex package available at
    https://github.com/mbr/latex/ under the following license:

    Copyright (c) 2015, Marc Brinkmann
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of latex nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """  # NOQA
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
    env.filters['html_to_tex'] = _html_to_tex
    env.filters['link_filter'] = _link_filter
    env.filters['pretty_json'] = _pretty_json
    env.filters['unique_flags_filter'] = _unique_flags_filter
    kwargs = _get_render_kwargs(report, dash_url, False)
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        logfile, auxfile = _prepare_latex_support_files(tmpdir, env, kwargs)
        _save_figures_to_pdf(tmpdir, report)
        _compile_files_into_pdf(tmpdir, logfile, auxfile, max_runs)
        return (tmpdir / 'out.pdf').read_bytes()


def _prepare_latex_support_files(tmpdir, env, kwargs):
    template = env.get_template('base.tex')
    tex = template.render(**kwargs)
    texfile = tmpdir / 'out.tex'
    texfile.write_text(tex)
    auxfile = tmpdir / 'out.aux'
    logfile = tmpdir / 'out.log'
    return logfile, auxfile


def _save_figures_to_pdf(tmpdir, report):
    figdir = tmpdir / 'figs'
    figdir.mkdir()
    for fig in report.raw_report.plots.figures:
        name = (
            fig.category + '+' + fig.metric + '+' +
            fig.name
        ).replace('^', '-').replace(' ', '+').replace('_', '+').replace(
            '<=', 'lte').replace('%', 'pct').replace('.', '').replace('/', '-')
        name += '.pdf'
        # handle characters that will cause problems for tex
        figpath = figdir / name
        figpath.write_bytes(base64.a85decode(fig.pdf))


def _compile_files_into_pdf(tmpdir, logfile, auxfile, max_runs):
    args = (
        'pdflatex',
        '-interaction=batchmode',
        '-halt-on-error',
        '-no-shell-escape',
        '-file-line-error',
        'out.tex'
    )
    runs_left = max_runs
    prev_aux = 'nothing to see here'
    # run pdflatex until it settles
    while runs_left > 0:
        try:
            subprocess.run(args, check=True, cwd=str(tmpdir.absolute()))
        except subprocess.CalledProcessError:
            try:
                logger.exception(logfile.read_text())
            except Exception:
                logger.exception('Pdflatex failed and so did reading log')
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
