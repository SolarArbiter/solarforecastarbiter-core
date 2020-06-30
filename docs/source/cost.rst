.. currentmodule: solarforecastarbiter.datamodel

#################
Calculating Costs
#################

Overview
========

The Solar Forecast Arbiter includes functionality to calculate the
cost of forecast errors. Error in this context refers to the deviation
of forecasted values from observed values (possible including a
deadband) and not a specific error metric e.g. MBE, RMSE. This page
explains the motivation for and structure of the cost calculation
functionality.

Basic costs can be specified as a :py:class:`constant <.ConstantCost>`
cost per unit error, a cost per unit error that varies by
:py:class:`time of day <.TimeOfDayCost>`, or a cost per per unit error
that varies by :py:class:`date-time <.DatetimeCost>`. Additionally, an
:py:class:`error band <.ErrorBandCost>` cost that specifies one of the
aforementioned basic costs depending on the size of the error is
implemented.  This banded cost allows one to specify a cost similar to
charges from transmission generator imbalance service as described in
`FERC Order 890-B
<https://www.ferc.gov/whats-new/comm-meet/2008/061908/E-1.pdf>`_.
Examples are provided below.

Most cost models allow the specification of an `aggregation` and `net`
parameters. The `aggregation` parameter controls how the cost for each
error value in the timeseries are aggregated (e.g. summed or averaged) into a single cost
number. The `net` parameter is a boolean that indicates if the
aggregation should keep the sign of the error, or take the absolute
value of the error before aggregating. Note that when :code:`net ==
True` and the cost per unit error is positive, it is possible to
calculate a final cost that is negative.


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
`sum` and `mean`. The `net` parameter is a boolean that specifies
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

The error band cost model is defined using :py:class:`.ErrorBandCost`
and implemented for deterministic forecasts by
:py:func:`solarforecastarbiter.metrics.deterministic.error_band_cost`.
Each of `bands` is a :py:class:`.CostBand` that describes the range of
errors the band applies to and one of the cost models
above. For example,

.. code-block:: python

    import datetime
    from solarforecastarbiter import datamodel

    cost_model = datamodel.ErrorBandCost(
        bands=[
            datamodel.CostBand(
                error_range=(-5.0, 20.5),
                cost_function='constant',
                cost_function_parameters=datamodel.ConstantCost(
                    cost=33.0,
                    net=True,
                    aggregation='sum'
                )
            ),
            datamodel.CostBand(
                error_range=(20.5, float('inf')),
                cost_function='timeofday'
                cost_function_parameters=datamodel.TimeOfDayCost(
                    cost=[3.3, 1.2],
                    times=[datetime.time(hour=15), datetime.time(hour=20)],
                    net=True,
                    aggregation='sum',
                    fill='forward'
                )
            )
        ]
    )

defines a cost that will apply a constant cost of $33.0 / unit error
for all errors in the range [-5.0, 20.5]. For errors > 20.5, the time
of day cost applies. The errors within each band are aggregated
according to the `aggregation` and `net` parameter of the band
parameters, but the total cost is the sum of all error bands.

Band error ranges are evaluated in the order specified and any errors
outside the list of ranges *are not evaluated*. Thus, for the model
described by

.. code-block:: python

    from solarforecastarbiter import datamodel

    cost_model = datamodel.ErrorBandCost(
        bands=[
            datamodel.CostBand(
                error_range=(-5.0, 5.0),
                cost_function='constant',
                cost_function_parameters=datamodel.ConstantCost(
                    cost=2.0,
                    net=True,
                    aggregation='mean'
                )
            ),
            datamodel.CostBand(
                error_range=(-10.0, 10.0),
                cost_function='constant',
                cost_function_parameters=datamodel.ConstantCost(
                    cost=4.0,
                    net=True,
                    aggregation='sum'
                )
            )
        ]
    )

errors in the range [-5, 5] have cost of $2.0 / unit error. Errors
that are outside [-5, 5] but within [-10, 10], that is errors in the
range [-10, 5) or (5, 10] have a cost of $4.0 / unit error. Errors
outside the range of [-10, 10] are not evaluated at all and have an
effective cost of $0 / unit error. Therefore, most use cases should
specify -Inf and Inf in the error ranges to ensure all errors have
some cost assigned to them.

The above model is equivalent to

.. code-block:: python

    from solarforecastarbiter import datamodel

    cost_model = datamodel.ErrorBandCost(
        bands=[
            datamodel.CostBand(
                error_range=(-5.0, 5.0),
                cost_function='constant',
                cost_function_parameters=datamodel.ConstantCost(
                    cost=2.0,
                    net=True,
                    aggregation='mean'
                )
            ),
            datamodel.CostBand(
                error_range=(-10.0, 5.0),
                cost_function='constant',
                cost_function_parameters=datamodel.ConstantCost(
                    cost=4.0,
                    net=True,
                    aggregation='sum'
                )
            ),
            datamodel.CostBand(
                error_range=(5.0, 10.0),
                cost_function='constant',
                cost_function_parameters=datamodel.ConstantCost(
                    cost=4.0,
                    net=True,
                    aggregation='sum'
                )
            )
        ]
    )


It is especially important to consider the sign of the `cost`
parameter and the value of `net` when using the error band cost. For
example,

.. code-block:: python

    from solarforecastarbiter import datamodel

    cost_model = datamodel.ErrorBandCost(
        bands=[
            datamodel.CostBand(
                error_range=(float('-inf'), 0),
                cost_function='constant',
                cost_function_parameters=datamodel.ConstantCost(
                    cost=2.0,
                    net=True,
                    aggregation='sum'
                )
            ),
            datamodel.CostBand(
                error_range=(0, float(inf)),
                cost_function='constant'
                cost_function_parameters=datamodel.ConstantCost(
                    cost=0,
                    net=True,
                    aggregation='sum'
                )
            )
        ]
    )

will always result in a negative (or 0) cost because the `net`
parameter of the first error band is True (so no absolute value is
taken) and the cost factor 2.0 will therefore multiply negative values
that are summed. This model is consistent with a contract where a
generator is paid some additional amount if it overproduces and is not
penalized for underproducing. A negative cost value in the first error
band in this case would penalize the producer for overproducing
compared to the forecast.

Finally, to implement a cost similar to
charges from transmission generator imbalance service as described in
`FERC Order 890-B
<https://www.ferc.gov/whats-new/comm-meet/2008/061908/E-1.pdf>`_, one might
define a cost model like


.. code-block:: python

    import datetime
    from solarforecastarbiter import datamodel

    cost_model = datamodel.ErrorBandCost(
        bands=[
            datamodel.CostBand(
                error_range=(-2, 2),
                cost_function='constant',
                cost_function_parameters=datamodel.ConstantCost(
                    cost=1.0,
                    net=True,
                    aggregation='sum'
                )
            ),
            datamodel.CostBand(
                error_range=(float('-inf'), -2),
                cost_function='timeofday'
                cost_function_parameters=datamodel.TimeOfDayCost(
                    cost=[5.1, 0.3],  # decremental cost
                    times=[datetime.time(16, 0), datetime.time(19, 0)],
                    net=False,
                    aggregation='sum',
                    fill='forward'
                )
            ),
            datamodel.CostBand(
                error_range=(2, float('inf')),
                cost_function='timeofday'
                cost_function_parameters=datamodel.TimeOfDayCost(
                    cost=[7.1, 1.4],  # incremental cost
                    times=[datetime.time(16, 0), datetime.time(19, 0)],
                    net=False,
                    aggregation='sum',
                    fill='forward'
                )
            )
        ]
    )

If this cost model is used to evaluate an hourly, mean AC power
forecast, errors between :math:`\pm 2` MW are netted over the
evaluation time period and assigned a value of $1 / MWh error. For
overproduction errors over 2 MW, a decremental cost is
charged/refunded based on a time of day cost. Underproduction errors
over 2 MW are charged an incremental cost depending on the time of the
infraction. Therefore, the total cost over the evaluation time period
is the net cost of errors within :math:`\pm 2` MW plus the cost of
each error over :math:`\pm 2` MW charged at the time the error occured
and summed over the evaluation time period.
