.. currentmodule: solarforecastarbiter.datamodel

#################
Calculating Costs
#################

Overview
========

The Solar Forecast Arbiter includes functionality to calculate the cost of
forecast errors. This page explains the motivation for and structure of the
cost calculation functionality.

Basic costs can be specified as a :py:class:`constant <.ConstantCost>`
cost per unit error, a cost per unit error that varies by
:py:class:`time of day <.TimeOfDayCost>`, or a cost per per unit error
that varies by :py:class:`date-time <.DatetimeCost>`. Additionally, an
:py:class:`error band <.ErrorBandCost>` cost that specifies one of the
aforementioned basic costs depending on the size of the error is
implemented.  This banded cost allows one to specify a cost similar to
charges from transmission generator imbalance service as described in
`FERC Order 890-B
<https://www.ferc.gov/whats-new/comm-meet/2008/061908/E-1.pdf>`_
as described below.


Basic Cost Models
=================

Constant
--------

The constant cost model parameters are defined by
:py:class:`.ConstantCost` and implemented for deterministic forecasts
by :py:func:`solarforecastarbiter.metrics.deterministic.constant_cost`.
This model expects a `cost` parameter with units of $ per unit error.
Thus, if one were comparing an AC power forecasts to observations, this
cost would be assumed to have units of $/MW. One could scale this cost
based on the forecast interval length to mimic a cost per MWh.

:py:class:`.ConstantCost` also expects `aggregation` and `net`
parameters. `aggregation` defines how costs are aggregated for a given
analysis time period (as specified in the report). Options include
`sum` and `mean`. The `net` parameter is boolean that specifies
whether or not the the sum/mean should be taken without (True) or with
(False) the absolute value of the error.

An example of a cost that would take the mean of error values after
taking the absolute value and applying a cost of $2.5/unit error is

.. code-block:: python

    from solarforecastarbiter import datamodel

    cost_model = solarforecastarbiter.datamodel.ConstantCost(
        cost=2.5,
        aggregation='mean',
        net=False
    )


Time of Day
-----------

The time-of-day cost model parameters are defined using
:py:class:`.TimeOfDayCost` and implemented for deterministic forecasts
by
:py:func:`solarforecastarbiter.metrics.deterministic.time_of_day_cost`.
Similar to the constant cost, the datamodel expects `aggregation` and
`net` parameters. In this case, `cost` is an iterable of cost values
that are paired with each time given by `times`. The `fill` parameter
specifies how the costs should be extended to times that are not
included in `times`. Options for `fill` include 'forward' and
'backward'. This filling parameter also controls how values "wrap
around" midnight. For example, for a cost describing different costs
depending on a evening peak,

.. code-block:: python

    import datetime
    from solarforecastarbiter import datamodel

    cost_model = datamodel.TimeOfDayCost(
        cost=[3.3, 1.2],
        times=[datetime.time(hour=15), datetime.time(hour=20)],
        net=True,
        aggregation='sum',
        fill='forward',
    )

the value of $3.3 / unit error applies from 15:00 to just before
20:00, and the value of $1.2 / unit error applies for all other times
in the day *except* 15:00 to 20:00. The `timezone` parameter defines
the timezone the `times` are referenced in. If `timezone` is None,
`times` is assumed to be in same timezone as the errors.


Date-time Cost
--------------

The date-time cost model is defined using :py:class:`.DatetimeCost`
and implemented for deterministic forecasts by
:py:func:`solarforecastarbiter.metrics.deterministic.datetime_cost`. Similar
to the time of day cost, the datamodel expects `aggregation`, `net`,
and `fill` parameters. In this case `cost` values are associated with
each date-time specified in `datetimes`. The `timezone` parameter
defines the timezone if `datetimes` are not localized, and if
`timezone` is None, the timezone of the errors is used.

The minimum/maximum bounds of `datetimes` should cover the range of
date-times that one wants to evaluate. For example, when evaluating
the cost defined by


.. code-block:: python

    import datetime
    from solarforecastarbiter import datamodel

    cost_model = datamodel.DatetimeCost(
        cost=[1.3, 1.9, 0.9, 2.0],
        times=[datetime.datetime(2020, 5, 1, 12, 0),
               datetime.datetime(2020, 5, 2, 12, 0),
               datetime.datetime(2020, 5, 3, 12, 0),
               datetime.datetime(2020, 5, 4, 12, 0)],
        net=True,
        aggregation='sum',
        fill='forward',
        timezone='UTC'
    )

errors in the timeseries before 2020-05-01T12:00 are not included in
the final calculation.


Error Band Cost
===============
