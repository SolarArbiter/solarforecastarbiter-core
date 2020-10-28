"""Summary statistics (observations and forecasts)."""

import numpy as np


# Add new metrics to this map to map shorthand to function
_MAP = {
    'mean': (np.nanmean, 'Mean'),
    'min': (np.nanmin, 'Min'),
    'max': (np.nanmax, 'Max'),
    'std': (np.nanstd, 'Std'),
    'median': (np.nanmedian, 'Median'),
    'var': (np.nanvar, 'Variance'),
}

__all__ = [m[0].__name__ for m in _MAP.values()]
