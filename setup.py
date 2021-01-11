from setuptools import setup, find_packages
from os import path


import versioneer


with open(path.join(path.abspath(path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()


EXTRAS_REQUIRE = {
    'test': ['pytest', 'pytest-cov', 'pytest-mock', 'pytest-asyncio',
             'asynctest', 'requests-mock', 'pytest-timeout'],
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
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7, <4',
    install_requires=[
        'netCDF4',
        'numpy>=1.15',
        'pandas>=1.0.3',
        'requests',
        'xarray',
        'tables',
        'pvlib==0.8.0',
        'scipy',
        'bokeh>=1.4.0, <2',
        'statsmodels',
        'jsonschema',
        'plotly>=4.5.0, <5',
        'psutil',
        'matplotlib',
    ],
    extras_require=EXTRAS_REQUIRE,
    project_urls={
        'Bug Reports': 'https://github.com/solararbiter/solarforecastarbiter-core/issues',  # NOQA,
        'Source': 'https://github.com/solararbiter/solarforecastarbiter-core',
        'Documentation': 'https://solarforecastarbiter-core.readthedocs.io'
    },
    entry_points={
        'console_scripts': [
            'solararbiter=solarforecastarbiter.cli:cli'
        ]
    }

)
