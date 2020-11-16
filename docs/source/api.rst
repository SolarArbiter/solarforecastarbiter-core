.. currentmodule:: solarforecastarbiter

.. _apiref:

API reference
#############

Data model
==========

The data model.

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

Probabilistic forecasts:

.. autosummary::
   :toctree: generated/

   datamodel.ProbabilisticForecast
   datamodel.ProbabilisticForecastConstantValue

Event forecasts:

.. autosummary::
   :toctree: generated/

   datamodel.EventForecast

Aggregates:

.. autosummary::
   :toctree: generated/

   datamodel.AggregateObservation
   datamodel.Aggregate

Data validation toolkit filters for use with reports:

.. autosummary::
   :toctree: generated/

   datamodel.BaseFilter
   datamodel.QualityFlagFilter
   datamodel.TimeOfDayFilter
   datamodel.ValueFilter

Containers to associate forecasts and observations for use with reports:

.. autosummary::
   :toctree: generated/

   datamodel.ForecastObservation
   datamodel.ForecastAggregate
   datamodel.ProcessedForecastObservation


Report metrics and validation:

.. autosummary::
   :toctree: generated/

   datamodel.MetricResult
   datamodel.MetricValue
   datamodel.ValidationResult
   datamodel.PreprocessingResult
   datamodel.ReportMessage

Report plots:

.. autosummary::
   :toctree: generated/

   datamodel.RawReportPlots
   datamodel.ReportFigure


Reports:

.. autosummary::
   :toctree: generated/

   datamodel.ReportParameters
   datamodel.RawReport
   datamodel.Report


Cost:

.. autosummary::
   :toctree: generated/

   datamodel.ConstantCost
   datamodel.TimeOfDayCost
   datamodel.DatetimeCost
   datamodel.CostBand
   datamodel.ErrorBandCost
   datamodel.Cost


All :py:mod:`~solarforecastarbiter.datamodel` objects have ``from_dict`` and
``to_dict`` methods:

.. autosummary::
   :toctree: generated/

   datamodel.BaseModel.from_dict
   datamodel.BaseModel.to_dict


PV modeling
===========

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


Reference forecasts
===================

Entry points
------------

High-level functions for NWP and persistence forecasts.

.. autosummary::
   :toctree: generated/

   reference_forecasts.main.run_nwp
   reference_forecasts.main.run_persistence
   reference_forecasts.main.find_reference_nwp_forecasts
   reference_forecasts.main.process_nwp_forecast_groups
   reference_forecasts.main.make_latest_nwp_forecasts
   reference_forecasts.main.fill_nwp_forecast_gaps
   reference_forecasts.main.make_latest_persistence_forecasts
   reference_forecasts.main.make_latest_probabilistic_persistence_forecasts
   reference_forecasts.main.fill_persistence_forecasts_gaps
   reference_forecasts.main.fill_probabilistic_persistence_forecasts_gaps

NWP models
----------

.. autosummary::
   :toctree: generated/

   reference_forecasts.models.hrrr_subhourly_to_subhourly_instantaneous
   reference_forecasts.models.hrrr_subhourly_to_hourly_mean
   reference_forecasts.models.rap_ghi_to_instantaneous
   reference_forecasts.models.rap_cloud_cover_to_hourly_mean
   reference_forecasts.models.gfs_quarter_deg_3hour_to_hourly_mean
   reference_forecasts.models.gfs_quarter_deg_hourly_to_hourly_mean
   reference_forecasts.models.gfs_quarter_deg_to_hourly_mean
   reference_forecasts.models.nam_12km_hourly_to_hourly_instantaneous
   reference_forecasts.models.nam_12km_cloud_cover_to_hourly_mean

Probabilistic NWP models
------------------------

.. autosummary::
   :toctree: generated/

   reference_forecasts.models.gefs_half_deg_to_hourly_mean

Forecast processing
-------------------

Functions that process forecast data.

.. autosummary::
   :toctree: generated/

   reference_forecasts.forecast.cloud_cover_to_ghi_linear
   reference_forecasts.forecast.cloud_cover_to_irradiance_ghi_clear
   reference_forecasts.forecast.cloud_cover_to_irradiance
   reference_forecasts.forecast.resample
   reference_forecasts.forecast.reindex_fill_slice
   reference_forecasts.forecast.unmix_intervals
   reference_forecasts.forecast.sort_gefs_frame

Persistence
-----------

.. autosummary::
   :toctree: generated/

   reference_forecasts.persistence.persistence_scalar
   reference_forecasts.persistence.persistence_interval
   reference_forecasts.persistence.persistence_scalar_index

Probabilistic persistence
-------------------------

.. autosummary::
   :toctree: generated/

   reference_forecasts.persistence.persistence_probabilistic
   reference_forecasts.persistence.persistence_probabilistic_timeofday


Fetching external data
======================

ARM
---

.. autosummary::
   :toctree: generated/

   io.fetch.arm.format_date
   io.fetch.arm.request_arm_file_list
   io.fetch.arm.list_arm_filenames
   io.fetch.arm.request_arm_file
   io.fetch.arm.retrieve_arm_dataset
   io.fetch.arm.extract_arm_variables
   io.fetch.arm.fetch_arm

NWP
---

.. autosummary::
   :toctree: generated/

   io.fetch.nwp.get_with_retries
   io.fetch.nwp.get_available_dirs
   io.fetch.nwp.check_next_inittime
   io.fetch.nwp.get_filename
   io.fetch.nwp.files_to_retrieve
   io.fetch.nwp.process_grib_to_netcdf
   io.fetch.nwp.optimize_netcdf
   io.fetch.nwp.sleep_until_inittime
   io.fetch.nwp.startup_find_next_runtime
   io.fetch.nwp.next_run_time
   io.fetch.nwp.run
   io.fetch.nwp.optimize_only
   io.fetch.nwp.check_wgrib2

DOE RTC
-------

.. autosummary::
   :toctree: generated/

   io.fetch.rtc.request_doe_rtc_data
   io.fetch.rtc.fetch_doe_rtc

NREL PVDAQ
----------

.. autosummary::
   :toctree: generated/

   io.fetch.pvdaq.get_pvdaq_metadata
   io.fetch.pvdaq.get_pvdaq_data

EIA
---

.. autosummary::
   :toctree: generated/

   io.fetch.eia.get_eia_data


Reference observations
======================

The following modules contain code for initializing the reference
database, wrappers for fetching data, functions for processing (e.g.
renaming and resampling) data, and wrapper functions for posting data.
The pure fetch functions are found in ``pvlib.iotools`` and in
``solarforecastarbiter.io.fetch``. See the source code for additional
files with site and observation metadata.

.. autosummary::
   :toctree: generated/

   io.reference_observations.common
   io.reference_observations.crn
   io.reference_observations.midc_config
   io.reference_observations.midc
   io.reference_observations.reference_data
   io.reference_observations.rtc
   io.reference_observations.solrad
   io.reference_observations.srml
   io.reference_observations.surfrad
   io.reference_observations.arm
   io.reference_observations.pvdaq
   io.reference_observations.eia
   io.reference_observations.bsrn


Reference aggregates
====================

The following modules contain code for initializing the reference
aggregates using Reference Observations that have already been created.
Examples include average GHI and DNI at SURFRAD sites, and the total
PV power in the Portland, OR area of UO SRML sites.

.. autosummary::
   :toctree: generated/

   io.reference_aggregates.generate_aggregate
   io.reference_aggregates.make_reference_aggregates


SFA API
=======

To pass API calls through a proxy server, set either the HTTP_PROXY or
HTTPS_PROXY environment variable. If necessary, set a SSL certificate using the
REQUESTS_CA_BUNDLE environment variable.

Token
-----

Get an API token.

.. autosummary::
   :toctree: generated/

   io.api.request_cli_access_token

API Session
-----------

Class for communicating with the Solar Forecast Arbiter API.

.. autosummary::
   :toctree: generated/

   io.api.APISession
   io.api.APISession.request
   io.api.APISession.get_user_info

Sites

.. autosummary::
   :toctree: generated/

   io.api.APISession.get_site
   io.api.APISession.list_sites
   io.api.APISession.list_sites_in_zone
   io.api.APISession.create_site

Observations

.. autosummary::
   :toctree: generated/

   io.api.APISession.get_observation
   io.api.APISession.list_observations
   io.api.APISession.create_observation
   io.api.APISession.get_observation_values
   io.api.APISession.post_observation_values
   io.api.APISession.get_observation_time_range
   io.api.APISession.get_observation_values_not_flagged
   io.api.APISession.get_observation_value_gaps

Forecasts

.. autosummary::
   :toctree: generated/

   io.api.APISession.get_forecast
   io.api.APISession.list_forecasts
   io.api.APISession.create_forecast
   io.api.APISession.get_forecast_values
   io.api.APISession.post_forecast_values
   io.api.APISession.get_forecast_time_range
   io.api.APISession.get_forecast_value_gaps

Probabilistic Forecasts

.. autosummary::
   :toctree: generated/

   io.api.APISession.get_probabilistic_forecast
   io.api.APISession.list_probabilistic_forecasts
   io.api.APISession.create_probabilistic_forecast
   io.api.APISession.get_probabilistic_forecast_values
   io.api.APISession.get_probabilistic_forecast_value_gaps
   io.api.APISession.get_probabilistic_forecast_constant_value
   io.api.APISession.get_probabilistic_forecast_constant_value_values
   io.api.APISession.post_probabilistic_forecast_constant_value_values
   io.api.APISession.get_probabilistic_forecast_constant_value_time_range
   io.api.APISession.get_probabilistic_forecast_constant_value_value_gaps

Aggregates

.. autosummary::
   :toctree: generated/

   io.api.APISession.get_aggregate
   io.api.APISession.list_aggregates
   io.api.APISession.create_aggregate
   io.api.APISession.get_aggregate_values

Reports

.. autosummary::
   :toctree: generated/

   io.api.APISession.process_report_dict
   io.api.APISession.get_report
   io.api.APISession.list_reports
   io.api.APISession.create_report
   io.api.APISession.post_raw_report_processed_data
   io.api.APISession.get_raw_report_processed_data
   io.api.APISession.post_raw_report
   io.api.APISession.update_report_status

Climate Zones

.. autosummary::
   :toctree: generated/

   io.api.APISession.list_sites_in_zone
   io.api.APISession.search_climatezones

Convenience method for unifying API for getting time series values
for observations, forecasts, aggregates, and probabilistic forecasts:

.. autosummary::
   :toctree: generated/

   io.api.APISession.get_values
   io.api.APISession.chunk_value_request
   io.api.APISession.get_value_gaps

Utils
-----

Utility functions for data IO.

.. autosummary::
   :toctree: generated/

   io.utils.observation_df_to_json_payload
   io.utils.forecast_object_to_json
   io.utils.json_payload_to_observation_df
   io.utils.json_payload_to_forecast_series
   io.utils.adjust_start_end_for_interval_label
   io.utils.adjust_timeseries_for_interval_label
   io.utils.ensure_timestamps
   io.utils.serialize_timeseries
   io.utils.deserialize_timeseries
   io.utils.load_report_values
   io.utils.mock_raw_report_endpoints


Metrics
=======

Entry points for calculating metrics for
:py:class:`~solarforecastarbiter.datamodel.Forecast` and
:py:class:`~solarforecastarbiter.datamodel.Observation`:

.. autosummary::
   :toctree: generated/

   metrics.calculator.calculate_metrics
   metrics.calculator.calculate_deterministic_metrics
   metrics.calculator.calculate_probabilistic_metrics
   metrics.calculator.calculate_event_metrics
   metrics.calculator.calculate_all_summary_statistics
   metrics.calculator.calculate_summary_statistics

Preprocessing
-------------

Functions for preparing the timeseries data before calculating metrics:

.. autosummary::
   :toctree: generated/

   metrics.preprocessing.check_reference_forecast_consistency
   metrics.preprocessing.apply_fill
   metrics.preprocessing.filter_resample
   metrics.preprocessing.align
   metrics.preprocessing.process_forecast_observations

Deterministic
-------------

Functions to compute forecast deterministic performance metrics:

.. autosummary::
   :toctree: generated/

   metrics.deterministic.mean_absolute
   metrics.deterministic.mean_bias
   metrics.deterministic.root_mean_square
   metrics.deterministic.normalized_mean_absolute
   metrics.deterministic.normalized_mean_bias
   metrics.deterministic.normalized_root_mean_square
   metrics.deterministic.centered_root_mean_square
   metrics.deterministic.mean_absolute_percentage
   metrics.deterministic.forecast_skill
   metrics.deterministic.pearson_correlation_coeff
   metrics.deterministic.coeff_determination
   metrics.deterministic.relative_euclidean_distance
   metrics.deterministic.kolmogorov_smirnov_integral
   metrics.deterministic.over
   metrics.deterministic.combined_performance_index

Functions to compute costs:

.. autosummary::
   :toctree: generated/

   metrics.deterministic.constant_cost
   metrics.deterministic.time_of_day_cost
   metrics.deterministic.datetime_cost
   metrics.deterministic.error_band_cost
   metrics.deterministic.cost

Functions to compute errors and deadbands:

.. autosummary::
   :toctree: generated/

   metrics.deterministic.deadband_mask
   metrics.deterministic.error
   metrics.deterministic.error_deadband

Probabilistic
-------------

Functions to compute forecast probabilistic performance metrics:

.. autosummary::
    :toctree: generated/

    metrics.probabilistic.brier_score
    metrics.probabilistic.brier_skill_score
    metrics.probabilistic.quantile_score
    metrics.probabilistic.quantile_skill_score
    metrics.probabilistic.brier_decomposition
    metrics.probabilistic.reliability
    metrics.probabilistic.resolution
    metrics.probabilistic.uncertainty
    metrics.probabilistic.sharpness
    metrics.probabilistic.continuous_ranked_probability_score
    metrics.probabilistic.crps_skill_score

Event
-----

Functions to compute deterministic event forecast performance metrics:

.. autosummary::
    :toctree: generated/

    metrics.event.probability_of_detection
    metrics.event.false_alarm_ratio
    metrics.event.probability_of_false_detection
    metrics.event.critical_success_index
    metrics.event.event_bias
    metrics.event.event_accuracy


Reports
=======

Main
----

Functions to compute the report.

.. autosummary::
   :toctree: generated/

   reports.main.compute_report
   reports.main.get_data_for_report
   reports.main.create_raw_report_from_data

Figures
-------

Functions for generating Plotly report metric figures.

.. autosummary::
   :toctree: generated/

   reports.figures.plotly_figures.construct_metrics_dataframe
   reports.figures.plotly_figures.construct_timeseries_dataframe
   reports.figures.plotly_figures.bar
   reports.figures.plotly_figures.bar_subdivisions
   reports.figures.plotly_figures.output_svg
   reports.figures.plotly_figures.raw_report_plots
   reports.figures.plotly_figures.timeseries_plots
   reports.figures.plotly_figures.timeseries
   reports.figures.plotly_figures.scatter


Functions for generating Bokeh plots.

.. autosummary::
   :toctree: generated/

   reports.figures.bokeh_figures.construct_timeseries_cds
   reports.figures.bokeh_figures.construct_metrics_cds
   reports.figures.bokeh_figures.timeseries
   reports.figures.bokeh_figures.scatter
   reports.figures.bokeh_figures.bar
   reports.figures.bokeh_figures.bar_subdivisions
   reports.figures.bokeh_figures.output_svg
   reports.figures.bokeh_figures.raw_report_plots
   reports.figures.bokeh_figures.timeseries_plots

Template
--------

Functions to generate output (HTML, PDF) for reports

.. autosummary::
   :toctree: generated/

   reports.template.render_html
   reports.template.get_template_and_kwargs
   reports.template.render_pdf

Validation
==========

Validator
---------

Functions to perform validation.

.. autosummary::
   :toctree: generated/

   validation.validator.check_ghi_limits_QCRad
   validation.validator.check_dhi_limits_QCRad
   validation.validator.check_dni_limits_QCRad
   validation.validator.check_irradiance_limits_QCRad
   validation.validator.check_irradiance_consistency_QCRad
   validation.validator.check_temperature_limits
   validation.validator.check_wind_limits
   validation.validator.check_rh_limits
   validation.validator.check_ac_power_limits
   validation.validator.check_dc_power_limits
   validation.validator.check_ghi_clearsky
   validation.validator.check_poa_clearsky
   validation.validator.check_day_night
   validation.validator.check_day_night_interval
   validation.validator.check_timestamp_spacing
   validation.validator.detect_stale_values
   validation.validator.detect_interpolation
   validation.validator.detect_levels
   validation.validator.detect_clipping
   validation.validator.detect_clearsky_ghi
   validation.validator.stale_interpolated_window


Tasks
-----

Perform a sequence of validation steps. Used by the API to initiate validation.

.. autosummary::
   :toctree: generated/

   validation.tasks.validate_ghi
   validation.tasks.validate_dni
   validation.tasks.validate_dhi
   validation.tasks.validate_poa_global
   validation.tasks.validate_dc_power
   validation.tasks.validate_ac_power
   validation.tasks.validate_defaults
   validation.tasks.validate_air_temperature
   validation.tasks.validate_wind_speed
   validation.tasks.validate_relative_humidity
   validation.tasks.validate_daily_ghi
   validation.tasks.validate_daily_dc_power
   validation.tasks.validate_daily_ac_power
   validation.tasks.validate_daily_defaults
   validation.tasks.apply_immediate_validation
   validation.tasks.apply_daily_validation
   validation.tasks.apply_validation


Quality flag mapping
--------------------

Functions to handle the translation of validation results and database storage.

.. autosummary::
   :toctree: generated/

   validation.quality_mapping.convert_bool_flags_to_flag_mask
   validation.quality_mapping.mask_flags
   validation.quality_mapping.has_data_been_validated
   validation.quality_mapping.get_version
   validation.quality_mapping.check_if_single_value_flagged
   validation.quality_mapping.which_data_is_ok
   validation.quality_mapping.check_for_all_descriptions
   validation.quality_mapping.convert_mask_into_dataframe
   validation.quality_mapping.convert_flag_frame_to_strings
   validation.quality_mapping.check_if_series_flagged



Plotting
========

Timeseries
----------

Time series plotting.

.. autosummary::
   :toctree: generated/

   plotting.timeseries.build_figure_title
   plotting.timeseries.make_quality_bars
   plotting.timeseries.add_hover_tool
   plotting.timeseries.make_basic_timeseries
   plotting.timeseries.generate_forecast_figure
   plotting.timeseries.generate_observation_figure
   plotting.timeseries.generate_probabilistic_forecast_figure

Utils
-----

Utility functions for plotting.

.. autosummary::
   :toctree: generated/

   plotting.utils.format_variable_name
   plotting.utils.align_index
   plotting.utils.line_or_step


Generic Utilities
=================

Generic utility functions.

.. autosummary::
   :toctree: generated/

   utils.compute_aggregate
   utils.sha256_pandas_object_hash
   utils.generate_continuous_chunks
   utils.merge_ranges
