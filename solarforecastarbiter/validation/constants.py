"""
Define constant mappings between bit-mask values and understandable quality
flags
"""
from functools import wraps


# DO NOT REMOVE OR CHANGE VALUES TO MAINTAN BACKWARD COMPATIBILITY
# The quality_flag field  in MySQL is currently limited to 1 << 15
_BITMASK_DESCRIPTIONS = (
    ('OK', 1 << 0),
    ('USER FLAGGED', 1 << 1),
    ('NIGHTTIME', 1 << 2),
    ('CLOUDY', 1 << 3),
    ('SHADED', 1 << 4),
    # 1 << 5 reserved for future informational flags
    # 1 << 6 reserved for future informational flags
    ('UNEVEN FREQUENCY', 1 << 7),
    ('LIMITS EXCEEDED', 1 << 8),
    ('CLEARSKY EXCEEDED', 1 << 9),
    ('STALE VALUES', 1 << 10),
    ('INTERPOLATED VALUES', 1 << 11),
    ('CLIPPED VALUES', 1 << 12),
    ('INCONSISTENT IRRADIANCE COMPONENTS', 1 << 13),
    ('DIFFUSE RATIO OUT OF BOUNDS', 1 << 14),
)


DESCRIPTION_MASK_MAPPING = {v[0]: v[1] for v in _BITMASK_DESCRIPTIONS}
MASK_DESCRIPTION_MAPPING = {v[1]: v[0] for v in _BITMASK_DESCRIPTIONS}


def mask_flags(f, flag_description, invert=True):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return_bool = kwargs.pop('return_bool', True)
        flags = f(*args, **kwargs)
        if not return_bool:
            if invert:
                return ~flags * DESCRIPTION_MASK_MAPPING[flag_description]
            else:
                return flags * DESCRIPTION_MASK_MAPPING[flag_description]
        else:
            return flags
    return wrapper
