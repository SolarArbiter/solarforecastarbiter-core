.. _contributing:

Contributing
============

We welcome and encourage your contributions to the Solar Forecast
Arbiter. This guide aims to provide potential contributors with a better
idea of how to prepare their code for the solarforecastarbiter-core and
what to expect from the project maintainers.

This guide focuses on code contributions, but we equally value
contributions such as issue reports, code review, and documentation
suggestions.

Nomenclature
------------

Contributors should review the `glossary of terms <https://github.com/SolarArbiter/solarforecastarbiter-core/wiki/Glossary>`_
and the guidance for
`variable names <https://github.com/SolarArbiter/solarforecastarbiter-core/wiki/Variable-names>`_.

Code style
----------

solarforecastarbiter-core follows the `PEP 8 -- Style Guide for Python Code
<https://www.python.org/dev/peps/pep-0008/>`_. Maximum line length for code
is 79 characters.

Code must be compatible with Python 3.7 and above.

Set your editor to strip extra whitespace from line endings. This
prevents the git commit history from becoming cluttered with whitespace
changes.

The majority of the solarforecastarbiter-core code consists of
functions that accept primitives or pandas.Series. Some higher-level
functions accept :py:mod:`~solarforecastarbiter.datamodel` objects.
Avoid writing new classes and methods.

Exceptions
----------

Code should raise an exception if a problem occurs and a solution is not
certain to be correct in all situations.

Logging
-------

Logging is encouraged in higher-level functions.

DataFrames and Series
---------------------

The solarforecastarbiter-core repository focuses on analyses of time
series data. The code makes extensive use of
pandas objects with DatetimeIndexes. In most cases, code should expect
and return one or more pandas.Series rather than a
pandas.DataFrame. A function that requires multiple Series is
self-documenting by virtue of the names of the parameters. Providing or
expecting the wrong data from such a function raises an exception at the
API level, where it belongs. A function that expects a single DataFrame
must carefully document the required keys, providing the wrong data will
result in a problem deep within the function, and expecting the wrong
data will result in a problem far away from the function.

The exceptions to the "use Series instead of DataFrames" rule are:

  * observation data may be packaged into a DataFrame with 'value' and
    'quality_flag' columns.
  * probabilistic forecasts.

Documentation
-------------

Documentation must be written in
`numpydoc format <https://numpydoc.readthedocs.io/>`_ format which is rendered
using the `Sphinx Napoleon extension
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.

To build the docs locally, install the dependencies specified in the
`docs/requirements.txt <https://github.com/SolarArbiter/solarforecastarbiter-core/blob/master/docs/requirements.txt>`_
file. After installing the dependencies, run ``make -C docs html`` from
the root of the repository.

Testing
-------

The vast majority of new or modified code requires contributors to write
new tests.

A pull request will automatically run the tests for you. However, it is
typically more efficient to run and debug the tests in your own local
environment. See :ref:`installation` for instructions to set up and use
an environment.

To run the tests locally, install the ``test`` dependencies specified in the
`setup.py <https://github.com/SolarArbiter/solarforecastarbiter-core/blob/master/setup.py>`_
file. The unit tests may be run using::

  pytest solarforecastarbiter

Pull request review
-------------------

The solarforecastarbiter-core maintainers and community members will
review pull requests for all of the topics discussed above. GitHub will
provide a checklist when creating a pull request. Feel free to submit a
pull request before the work is complete.

Please keep each contribution to as few lines of code as possible. We'd
much rather review many short pull requests than one long pull request.

See GitHub's
`About pull requests <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_
guide for more on the mechanics of the process.
