"""
Inserts metadata and figures into the report template.
"""
import subprocess

from bokeh.embed import components

from jinja2 import (Environment, DebugUndefined, PackageLoader,
                    select_autoescape, Template)

from solarforecastarbiter.reports import figures


def prereport(report, metadata, metrics):
    """
    Render the markdown prereport. Figures are left untemplated.

    Parameters
    ----------
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    metadata : str, dict
        Describes the pre-report
    metrics : tuple of dict

    Returns
    -------
    prereport : markdown
    """
    # By default, jinja removes undefined variables from the rendered string.
    # DebugUndefined leaves undefined variables in the string so that they
    # can be used in the full report template process.
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml']),
        undefined=DebugUndefined)

    template = env.get_template('template.md')

    cds = figures.construct_metrics_cds(metrics, 'total')
    data_table = figures.metrics_table(cds)

    cds = figures.construct_metrics_cds(metrics, 'total', index='forecast')
    figures_bar = {}
    for num, metric in enumerate(report.metrics):
        fig = figures.bar(cds, metric)
        figures_bar[f'figures_bar_{num}'] = fig

    script, divs = components(dict(tables=data_table, **figures_bar))

    print_versions = metadata['versions']
    rendered = template.render(
        name=metadata['name'],
        start=metadata['start'],
        end=metadata['end'],
        now=metadata['now'],
        print_versions=print_versions,
        script_metrics=script,
        **divs)
    return rendered


# not all args currently used, but expect they will eventually be used
def add_figures_to_prereport(fx_obs_cds, report, metadata, prereport):
    """
    Add figures to the prereport, convert to html

    Parameters
    ----------
    fx_obs_cds : list
        List of (forecast, observation, ColumnDataSource) tuples to
        pass to bokeh plotting objects.
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    metadata : str, dict
        Describes the pre-report
    prereport : str, markdown or html
        The templated pre-report.

    Returns
    -------
    body : str, markdown or html
        The body of the full report in the same format
        (markdown or html) as the prereport.
    """
    body_template = Template(prereport)

    ts_fig = figures.timeseries(fx_obs_cds, report.start, report.end)
    scat_fig = figures.scatter(fx_obs_cds)

    script, divs = components(
        {'figures_timeseries': ts_fig, 'figures_scatter': scat_fig})

    body = body_template.render(script_data=script, **divs)
    return body


def prereport_to_html(prereport):
    """
    Render markdown into simple html using pandoc.

    Parameters
    ----------
    prereport : str, markdown

    Returns
    -------
    prereport : str, html
    """
    try:
        out = subprocess.run(args=['pandoc'], input=prereport.encode(),
                             capture_output=True)
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
