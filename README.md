[![Build Status](https://dev.azure.com/solararbiter/solarforecastarbiter/_apis/build/status/SolarArbiter.solarforecastarbiter-core?branchName=master)](https://dev.azure.com/solararbiter/solarforecastarbiter/_build/latest?definitionId=1&branchName=master)
[![Coverage](https://img.shields.io/azure-devops/coverage/solararbiter/solarforecastarbiter/1/master.svg)](https://dev.azure.com/solararbiter/solarforecastarbiter/_build/latest?definitionId=1&branchName=master)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/SolarArbiter/solarforecastarbiter-core.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SolarArbiter/solarforecastarbiter-core/alerts/)
[![codecov](https://codecov.io/gh/solararbiter/solarforecastarbiter-core/branch/master/graph/badge.svg)](https://codecov.io/gh/solararbiter/solarforecastarbiter-core)
[![Documentation Status](https://readthedocs.org/projects/solarforecastarbiter-core/badge/?version=latest)](https://solarforecastarbiter-core.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3473590.svg)](https://doi.org/10.5281/zenodo.3473590)

# solarforecastarbiter-core
Core Solar Forecast Arbiter data gathering, validation, processing, and
reporting package.

# Installation

See the [installation](https://solarforecastarbiter-core.readthedocs.io/en/latest/installation.html) instructions in the documentation.

# Documentation

The documentation is hosted at [solarforecastarbiter-core.readthedocs.io](https://solarforecastarbiter-core.readthedocs.io/en/latest/)

# Contributing

We welcome your contributions. Please see our [contributing guide](https://solarforecastarbiter-core.readthedocs.io/en/latest/contributing.html).

# Architecture

The diagram below depicts data flow between components of the Solar
Forecast Arbiter framework. Users of the framework typically interact
with the ``solarforecastarbiter-core`` code through a queue/worker
system maintained by the API. Users may access the API directly or
through the Dashboard. The API queues analyses to be processed by
workers using core code. The workers then send their results to the API
for storage in the database.

Alternatively, users may choose to install the core package on their own
systems and perform their analyses independently of the Dashboard, API,
or database.

![system sketch](system_sketch.png)
