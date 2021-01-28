.. currentmodule:: solarforecastarbiter.cli

.. _cli:

######################
Command Line Interface
######################

The Solar Forecast Arbiter command line interface (CLI) is used to automate
a handful of common tasks. After installing ``solarforecastarbiter``,
the command line intereface may be invoked using the ``solararbiter``
command. Run ``solararbiter -h`` for help. The CLI may also be accessed
using ``python solarforecastarbiter.cli``.

CLI tasks that use the API will look for credentials stored in the
environment variables ``SFA_API_USER`` and ``SFA_API_PASSWORD``.

The supported tasks include:

.. autosummary::
   :toctree: generated

   validate
   referencedata
   fetchnwp
   referencenwp
   report
