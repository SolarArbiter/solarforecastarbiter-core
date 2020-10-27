.. curentmodule: solarforecastarbiter.io.reference_observations

######################
Reference Observations
######################

Overview
========

The Solar Forecast Arbiter imports reference observation data from multiple
measurement networks. All of the logic for creating the appropriate Solar
Forecast Arbiter sites and observations and updating observations with new data
from the network can be found in the
:py:mod:`solarforecastarbiter.io.reference_observations` subpackage. Code for
retrieving data from the network's APIs are spread between the
:py:mod:`solarforecastarbiter.io.fetch` subpackage, and the
`pvlib python <https://pvlib-python.readthedocs.io/en/stable/index.html>`_ *iotools*
module.

A list of these networks and their Solar Forecast Arbiter modules can be found
in the `Available Network Handlers`_ section. A map of all of the sites
available in the reference dataset can be found on the
`Solar Forecast Arbiter project website <https://solarforecastarbiter.org/referencedata/>`_.


Structure
=========
The :py:mod:`solarforecastarbiter.io.reference_observations` subpackage
contains python modules and data files in JSON and CSV format.

Data Files
----------
* `sfa_reference_sites.csv`
   The master list of reference sites. See the comment at the top of this file
   for descriptions of its fields. The file contains extra fields that are not
   found in the Solar Forecast Arbiter API schema for Sites. These fields are
   for use with the source network's API and are stored in the
   `extra_parameters` field when the site is created for use in subsequent
   updates.

* `<network>_reference_sites.json`
   Network-specific files containing site and observation metadata in the Solar
   Forecast Arbiter API's JSON format. These are used when the master CSV does
   not contain all of the columns needed to accurately define a site or
   observation.

Modules
-------
* :py:mod:`solarforecastarbiter.io.reference_observations.reference_data`
   This module coordinates the initialization and update process. It also
   contains the `NETWORKHANDLER_MAP` dictionary, which maps network names to
   the correct `Network Handlers`_. The functions in the module are utilized by
   the CLI `referencedata` command.

* :py:mod:`solarforecastarbiter.io.reference_observations.common`
   The `common` module contains utility functions for use throughout the
   `reference_data` subpackage. It has useful functions for converting
   external data into Solar Forecast Arbiter Datamodel objects and
   network-agnostic utilities for preparing and posting data to the Solar
   Forecast Arbiter API. Most `Network Handlers`_ rely heavily on these
   functions.

Network Handlers
****************
Network Handlers are network specific modules that implement a handful of
functions with a common interface. See
:py:mod:`solarforecastarbiter.io.reference_observations.surfrad` for an
example.

The required network handler functions are:

* `initialize_site_observations(api, site)`
   Create an observation at the site for each variable available from the
   network.

      * api: :py:class:`solarforecastarbiter.io.api.APISession`
      * site: :py:class:`solarforecastarbiter.datamodel.Site`


* `initialize_site_forecasts(api, site)`
   Create a forecast for each observation at the site.

      * api: :py:class:`solarforecastarbiter.io.api.APISession`
      * site: :py:class:`solarforecastarbiter.datamodel.Site`


* `update_observation_data(api, sites, observations, start, end)`
   Retrieve data from the network then format and post it to each observation
   at the site.

      * api: :py:class:`solarforecastarbiter.io.api.APISession`
      * sites: list of :py:class:`solarforecastarbiter.datamodel.Site`
      * observations: list of :py:class:`solarforecastarbiter.datamodel.Observation`
      * start: datetime
      * end: datetime


* (optional) `adjust_site_parameters(site)`
   In instances where the master site CSV does not contain enough metadata about
   the site, (e.g. when a PV plant requires `modeling_parameters`) this function
   may be used to update the site metadata before it is posted to the API.

      * site: dict


Available Network Handlers
^^^^^^^^^^^^^^^^^^^^^^^^^^
* SURFRAD: NOAA Surface Radiation Budget Network
   https://www.esrl.noaa.gov/gmd/grad/surfrad/

   :py:mod:`solarforecastarbiter.io.reference_observations.surfrad`

* SOLRAD: NOAA SOLRAD Network
   https://www.esrl.noaa.gov/gmd/grad/solrad/index.html

  :py:mod:`solarforecastarbiter.io.reference_observations.solrad`

* CRN: NOAA U.S. Climate Reference Network
   https://www.ncdc.noaa.gov/crn/

   :py:mod:`solarforecastarbiter.io.reference_observations.crn`

* NREL MIDC: National Renewable Energy Laboratory Measurement and Instrumentation Data Center
   https://midcdmz.nrel.gov/

   :py:mod:`solarforecastarbiter.io.reference_observations.midc`

* UO SRML: University of Oregon Solar Radiation Monitoring Laboratory
   http://solardat.uoregon.edu/

   :py:mod:`solarforecastarbiter.io.reference_observations.srml`

* DOE RTC: DOE Regional Test Centers for Solar Technologies\*
   https://pv-dashboard.sandia.gov/

   :py:mod:`solarforecastarbiter.io.reference_observations.rtc`

* DOE ARM: DOE Atmospheric Radiation Measurement\*
   https://www.arm.gov/

   :py:mod:`solarforecastarbiter.io.reference_observations.arm`

* NREL PVDAQ: National Renewable Energy Laboratory PV Data Acquisition\*
   https://developer.nrel.gov/docs/solar/pvdaq-v3/

   :py:mod:`solarforecastarbiter.io.reference_observations.pvdaq`

* EIA: U.S. Energy Information Adminstration Open Data\*
   https://www.eia.gov/opendata/

   :py:mod:`solarforecastarbiter.io.reference_observations.eia`

* WRMC BSRN: World Radiation Monitoring Center - Baseline Surface Radiation Network\*
   https://bsrn.awi.de

   :py:mod:`solarforecastarbiter.io.reference_observations.bsrn`


\* Requesting data from these networks requires a valid api key or other
credentials for the associated API.
