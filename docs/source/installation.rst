.. _installation:

Installation
============

The ``solarforecastarbiter`` package is available on PyPI and conda-forge.

On PyPI::

    pip install solarforecastarbiter

Optional dependencies are specified with the following options:

    * ``fetch``: packages required for fetching NWP model grids.
    * ``plotting``: packages required for making plots and thus reports.
    * ``log``: packages requried for logging using `Sentry.io <https://sentry.io>`_.
    * ``test``: packages required for testing.
    * ``all``: all optional dependencies.

For example::

    pip install solarforecastarbiter[all]

The conda-forge package includes all optional dependencies::

    conda install -c conda-forge solarforecastarbiter

The package installation also includes the ``solararbiter`` :ref:`cli`.

The recommended way to install the package for development is as follows.

First, fork and clone the repository to your machine. From within the
root level of the repository, execute the follow shell commands::

    conda create -n sfacore python=3.7
    conda activate sfacore
    pip install -r requirements.txt -r requirements-test.txt
    pip install -e .

If everything worked, you should be able to run::

    pytest solarforecastarbiter
    flake8 solarforecastarbiter

If you want to build the docs, also run::

    pip install -r docs/requirements.txt

If you want to install all requirements, use::

    pip install -e .[all]

A docker image is also available at
`quay.io <https://quay.io/repository/solararbiter/solarforecastarbiter-core>`_::

    docker pull quay.io/solararbiter/solarforecastarbiter-core