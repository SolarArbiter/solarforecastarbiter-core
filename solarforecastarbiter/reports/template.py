"""
Inserts metadata and figures into the report template.
"""

from bokeh.embed import components

from jinja2 import (Environment, DebugUndefined, PackageLoader, BaseLoader,
                    select_autoescape, Template)
import markdown

from solarforecastarbiter.reports import figures


TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <link href="http://netdna.bootstrapcdn.com/twitter-bootstrap/2.3.0/css/bootstrap-combined.min.css" rel="stylesheet">
    <style>
        body {
            font-family: sans-serif;
        }
        code, pre {
            font-family: monospace;
        }
        h1 code,
        h2 code,
        h3 code,
        h4 code,
        h5 code,
        h6 code {
            font-size: inherit;
        }
    </style>
</head>
<body>
<div class="container">
{{content}}
</div>
</body>
</html>
"""


def prereport(metadata, metrics):
    """
    Render the markdown prereport. Figures are left untemplated.

    Parameters
    ----------
    metadata : dict
    metrics : dict

    Returns
    -------
    prereport : markdown
    """
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml']),
        undefined=DebugUndefined)

    template = env.get_template('template.md')

    print_versions = metadata['versions']
    rendered = template.render(print_versions=print_versions)
    return rendered


# not all args currently used, but expect they will eventually be used
def add_figures_to_prereport_md(fx_obs_cds, report, metadata, prereport):
    """
    Add figures to the prereport, convert to html

    Parameters
    ----------
    fx_obs_cds : list
        List of (forecast, observation, ColumnDataSource) tuples to
        pass to bokeh plotting objects.
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    metadata : str, json
        Describes the pre-report
    prereport : str, markdown
        The templated pre-report.

    Returns
    -------
    report : str, html
        The full report.
    """
    extensions = ['extra', 'smarty']
    html = markdown.markdown(prereport, extensions=extensions, output_format='html5')
    doc = Template(TEMPLATE).render(content=html)

    ts_fig = figures.timeseries(fx_obs_cds, report.start, report.end)
    scat_fig = figures.scatter(fx_obs_cds)

    script, divs = components(
        {'figures_timeseries': ts_fig, 'figures_scatter': scat_fig})

    rendered = template.render(**divs)
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
    metadata : str, json
        Describes the pre-report
    prereport : str, markdown
        The templated pre-report.

    Returns
    -------
    head : str, html
        Header for the full report.
    body : str, markdown
        The full report in markdown format with embedded html
    """
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml']))
    head_template = env.get_template('head.html')

    body_template = Template(prereport)

    ts_fig = figures.timeseries(fx_obs_cds, report.start, report.end)
    scat_fig = figures.scatter(fx_obs_cds)

    script, divs = components(
        {'figures_timeseries': ts_fig, 'figures_scatter': scat_fig})

    head = head_template.render(script=script)
    body = body_template.render(**divs)
    return head, body
