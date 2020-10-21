from setuptools import setup, find_packages
from os import path


import versioneer


with open(path.join(path.abspath(path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()


EXTRAS_REQUIRE = {
    'test': ['pytest', 'pytest-cov', 'pytest-mock', 'pytest-asyncio',
             'asynctest', 'requests-mock'],
    'fetch': ['aiohttp', 'loky', 'psutil'],
    'log': ['sentry-sdk'],
    'cli': ['click'],
    'plotting': ['bokeh', 'selenium']
}
EXTRAS_REQUIRE['all'] = [
    vv for v in EXTRAS_REQUIRE.values() for vv in v]


setup(
    name='solarforecastarbiter',
    description='Core framework for Solar Forecast Arbiter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/solararbiter/solarforecastarbiter-core',
    author='Solar Forecast Arbiter Team',
    author_email='info@solarforecastarbiter.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'netCDF4',
        'numpy',
        'pandas>=0.25.1',
        'requests',
        'xarray',
        'tables',
        'pvlib>=0.7.0',
        'scipy',
        'bokeh',
        'statsmodels',
        'jsonschema',
        'plotly',
        'psutil',
        'matplotlib',
    ],
    extras_require=EXTRAS_REQUIRE,
    project_urls={
        'Bug Reports': 'https://github.com/solararbiter/solarforecastarbiter-core/issues',  # NOQA,
        'Source': 'https://github.com/solararbiter/solarforecastarbiter-core'
    },
    entry_points={
        'console_scripts': [
            'solararbiter=solarforecastarbiter.cli:cli'
        ]
    }

)
