.. _installation:

Installation
============

The package is not yet available on PyPI or conda-forge.

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
