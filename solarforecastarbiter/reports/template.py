
import pandas as pd
import numpy as np

from solarforecastarbiter import __version__

from jinja2 import Environment, PackageLoader, select_autoescape


def main():
    env = Environment(
        loader=PackageLoader('solarforecastarbiter.reports', 'templates'),
        autoescape=select_autoescape(['html', 'xml'])
    )

    template = env.get_template('template.md')

    print_versions = f"""
solarforecastarbiter: {__version__}
pandas: {pd.__version__}
numpy: {np.__version__}
    """
    rendered = template.render(print_versions=print_versions)
    return rendered


if __name__ == '__main__':
    rendered = main()
    print(rendered)
