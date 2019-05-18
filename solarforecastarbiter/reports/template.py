# maybe put this in main instead

from jinja2 import (Environment, DebugUndefined, PackageLoader,
                    select_autoescape)


def main(metadata, metrics):
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml']),
        undefined=DebugUndefined)

    template = env.get_template('template.md')

    print_versions = metadata['versions']
    rendered = template.render(print_versions=print_versions)
    return rendered
