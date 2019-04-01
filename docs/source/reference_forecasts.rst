.. currentmodule:: solarforecastarbiter.reference_forecasts

###################
Reference Forecasts
###################

Structure
=========

The Solar Forecast Arbiter supports reference forecasts based on
data from numerical weather prediction models (NWP) and from site
observations.

The :py:mod:`~solarforecastarbiter.reference_forecasts.main`
module orchestrates reference forecast generation
within the Solar Forecast Arbiter.

.. autosummary::
   :toctree: generated/

   main.run


NWP
===

Forecasts based on numerical weather prediction model data are used
for intraday and longer forecasts. The Solar Forecast Arbiter contains
a series of functions with reference implementations of data processing
steps for NWP-based forecasts. These functions are found in the
:py:mod:`~solarforecastarbiter.reference_forecasts.models` module.
Each function is specific to a particular NWP
model and forecast time resolution.

.. autosummary::
   :toctree: generated/

   models.hrrr_subhourly_to_subhourly_instantaneous
   models.hrrr_subhourly_to_hourly_mean
   models.rap_to_instantaneous
   models.rap_irrad_to_hourly_mean
   models.rap_cloud_cover_to_hourly_mean
   models.gfs_3hour_to_hourly_mean
   models.gfs_hourly_to_hourly_mean
   models.nam_to_hourly_instantaneous
   models.nam_cloud_cover_to_hourly_mean


Many of the functions in
:py:mod:`~solarforecastarbiter.reference_forecasts.models` rely on common functions
related to forecast processing. These functions are found in
:py:mod:`~solarforecastarbiter.reference_forecasts.forecast`.

.. autosummary::
   :toctree: generated/

   forecast.cloud_cover_to_ghi_linear
   forecast.cloud_cover_to_irradiance_clearsky_scaling
   forecast.cloud_cover_to_irradiance_clearsky_scaling_solpos
   forecast.resample
   forecast.interpolate
   forecast.unmix_intervals


Persistence
===========

TBD.
