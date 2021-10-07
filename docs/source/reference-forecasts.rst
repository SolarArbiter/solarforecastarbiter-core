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

   main
   main.run_nwp
   main.run_persistence


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

   models
   models.hrrr_subhourly_to_subhourly_instantaneous
   models.hrrr_subhourly_to_hourly_mean
   models.rap_ghi_to_instantaneous
   models.rap_cloud_cover_to_hourly_mean
   models.gfs_quarter_deg_3hour_to_hourly_mean
   models.gfs_quarter_deg_hourly_to_hourly_mean
   models.gfs_quarter_deg_to_hourly_mean
   models.nam_12km_hourly_to_hourly_instantaneous
   models.nam_12km_cloud_cover_to_hourly_mean
   models.gefs_half_deg_to_hourly_mean

All of the above functions return weather forecast data, a *resampler*
function, and a solar position calculation function. The weather
forecast data may be supplied to a PV model and then resampled using the
*resampler* function. This workflow allows for seperation of weather
data processing and PV modeling while preserving the ability to use more
accurate, shorter time interval inputs to the PV model. In the case of
probabilistic forecasts, the *resampler* function also may define how
an ensemble of deterministic forecasts should be translated to a
probabilistic forecast. Finally, these
functions return a solar position calculation function (rather than the
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

   forecast
   forecast.cloud_cover_to_ghi_linear
   forecast.cloud_cover_to_irradiance_ghi_clear
   forecast.cloud_cover_to_irradiance
   forecast.resample
   forecast.reindex_fill_slice
   forecast.unmix_intervals
   forecast.sort_gefs_frame


Persistence
===========

The solarforecastarbiter supports several varieties of deterministic and
probabilistic persistence forecasts.

.. autosummary::
   :toctree: generated/

   persistence
   persistence.persistence_scalar
   persistence.persistence_interval
   persistence.persistence_scalar_index
   persistence.persistence_probabilistic
   persistence.persistence_probabilistic_timeofday


Automated Generation
====================

The Solar Forecast Arbiter runs workers that automatically create reference
forecasts. This feature is currently limited to reference forecasts defined
by the Arbiter itself. In the future we hope to extend this to user-defined
reference forecasts.

Automated generation of reference NWP forecasts is achieved by adding
a set of parameters to a Forecast's extra_parameters (formatted as JSON).
These parameters are:

  * *is_reference_forecast* - *true* or *'true'* for automated generation
  * *model* - string of one of the functions found in
    :py:mod:`~solarforecastarbiter.reference_forecasts.models`
  * *piggyback_on* - optional, the ID of another Forecast object to group
    together when making forecasts. For example, if ForecastA has variable
    *ac_power* and ForecastB has variable for *ghi* for the same site,
    ForecastB *piggyback_on* can be set to the forecast_id of ForecastA.
    Then these forecasts would be grouped together and the values of ForecastB
    would be the same GHI values that were used in the generation of ForecastA.


An example of a valid *extra_parameters* JSON for automated generation is:

.. code-block:: json

    {
        "is_reference_forecast": true,
        "model": "gfs_quarter_deg_hourly_to_hourly_mean",
        "piggyback_on": "da2bc386-8712-11e9-a1c7-0a580a8200ae"
    }


The function :py:func:`~solarforecastarbiter.reference_forecasts.main.make_latest_nwp_forecasts`
is responsible for listing all forecasts available to a user and generating
the appropriate reference NWP forecasts. In practice, the CLI script
:py:func:`~solarforecastarbiter.cli.referencenwp` is called as a cronjob
using an appropriate reference user account to continuously update pre-defined
reference forecasts.

.. autosummary::
   :toctree: generated/

   main.make_latest_nwp_forecasts
