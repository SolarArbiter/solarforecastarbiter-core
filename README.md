# solarforecastarbiter-core
Core data gathering, validation, processing, and reporting package.

Sketch of how some SFA components interact, including the core:

![system sketch](system_sketch.jpg)

All core code will be executed by *workers*. Workers are dispatched by
a *queue*. The workers will use the *API* to get and receive data from the
*database*.

The core's dataio subpackage will provide python wrappers for the API.
The API wrappers will typically return dicts of metadata and Series or
DataFrames of time series data. Most functions in the core subpackages
for validation, metrics, benchmark forecasts, and reports should assume
pandas Series or DataFrame inputs with a valid DatetimeIndex.
