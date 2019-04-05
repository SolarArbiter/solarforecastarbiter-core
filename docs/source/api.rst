.. currentmodule:: solarforecastarbiter

#############
API reference
#############

datamodel
=========

The datamodel.

.. autosummary::
   :toctree: generated/

   datamodel

There are two kinds of sites:

.. autosummary::
   :toctree: generated/

   datamodel.Site
   datamodel.SolarPowerPlant

The several model parameters are associated with the solar power plant site:

.. autosummary::
   :toctree: generated/

   datamodel.PVModelingParameters
   datamodel.FixedTiltModelingParameters
   datamodel.SingleAxisModelingParameters

The Observation and Forecast:

.. autosummary::
   :toctree: generated/

   datamodel.Observation
   datamodel.Forecast


pvmodel
=======

The :py:mod:`~solarforecastarbiter.pvmodel` module contains functions
closely associated with PV modeling.

.. autosummary::
   :toctree: generated/

   pvmodel

Several utility functions wrap pvlib functions:

.. autosummary::
   :toctree: generated/

   pvmodel.calculate_solar_position
   pvmodel.complete_irradiance_components
   pvmodel.calculate_clearsky


Three functions are useful for determining AOI, surface tilt, and
surface azimuth. :py:func:`~pvmodel.aoi_func_factory` is helpful for
standardizing the calculations for tracking and fixed systems.
See :py:func:`~pvmodel.calculate_poa_effective`, for example.

.. autosummary::
   :toctree: generated/

   pvmodel.aoi_func_factory
   pvmodel.aoi_fixed
   pvmodel.aoi_tracking


.. autosummary::
   :toctree: generated/

   pvmodel.calculate_poa_effective_explicit
   pvmodel.calculate_poa_effective
   pvmodel.calculate_power
   pvmodel.irradiance_to_power


subpackages
===========

The subpackages:

.. toctree::
   :maxdepth: 2

   reference_forecasts


.. autosummary::
   :toctree: generated/

   io
   metrics
   reports
   validation
