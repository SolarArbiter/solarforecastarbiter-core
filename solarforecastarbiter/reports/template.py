
import pandas as pd
import numpy as np

from solarforecastarbiter import __version__

template = f"""
This report was automatically generated using the Solar Forecast Arbiter.

Important metadata...

warn if data checksum does not equal saved checksum

Data
----
INSERT figures.timeseries AND figures.scatter PLOTS

Metrics
-------
USE FUNCTIONS TO RENDER TABLES AND figures.bar PLOTS FROM JSON DATA

Versions
--------
solarforecastarbiter: {__version__}
pandas: {pd.__version__}
numpy: {np.__version__}

Hash
----
Report hash
"""
