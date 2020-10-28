"""Summary statistics (observations and forecasts)."""

import numpy as np


# Add new metrics to this map to map shorthand to function
_MAP = {
    'mean': (np.nanmean, 'Mean'),
    'min': (np.nanmin, 'Min'),
    'max': (np.nanmax, 'Max'),
    'median': (np.nanmedian, 'Median'),
    'std': (np.nanstd, 'Std.'),
}

__all__ = [m[0].__name__ for m in _MAP.values()]
