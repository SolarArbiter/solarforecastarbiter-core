.. currentmodule:: solarforecastarbiter.reference_forecasts

###################
Reference Forecasts
###################

Structure
=========

The Solar Forecast Arbiter supports reference forecasts based on
data from numerical weather prediction (NWP) models and from site
observations.

The :py:mod:`~solarforecastarbiter.reference_forecasts.main` module
orchestrates reference forecast generation within the Solar Forecast
Arbiter. It uses data types defined in
:py:mod:`solarforecastarbiter.datamodel`.

.. autosummary::
   :toctree: generated/

   main.run


NWP
===

Forecasts based on NWP model data are used for intraday and longer
forecasts. The Solar Forecast Arbiter contains a set of functions
to process data from NWP forecasts. These functions are found in the
:py:mod:`~solarforecastarbiter.reference_forecasts.models` module. Each
function is specific to:

  1. A particular NWP model data set (e.g. NAM or subhourly HRRR), and
  2. The post processing steps required to obtain a particular type of
     irradiance or power forecast data (e.g. hourly mean or
     instantaneous).

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

All of the above functions return weather forecast data, a *resampler*
function, and a solar position calculation function. The weather
forecast data may be supplied to a PV model and then resampled using the
*resampler* function. This workflow allows for seperation of weather
data processing and PV modeling while preserving the ability to use more
accurate, shorter time interval inputs to the PV model. Finally, these
functions return a solar position calcuation function (rather than the
actual solar position) to simplify the API while maintaining reasonable
performance. (Solar position is only sometimes needed within the model
processing functions and is only needed externally if power is to be
calculated.) See
:py:mod:`~solarforecastarbiter.reference_forecasts.models` module for
additional documentation.

Many of the functions in
:py:mod:`~solarforecastarbiter.reference_forecasts.models` rely on
common functions related to forecast processing. These functions are
found in :py:mod:`~solarforecastarbiter.reference_forecasts.forecast`.

.. autosummary::
   :toctree: generated/

   forecast.cloud_cover_to_ghi_linear
   forecast.cloud_cover_to_irradiance_ghi_clear
   forecast.cloud_cover_to_irradiance
   forecast.resample
   forecast.interpolate
   forecast.unmix_intervals


Persistence
===========

TBD.
