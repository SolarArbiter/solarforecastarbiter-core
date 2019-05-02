"""
Define constant mappings between bit-mask values and understandable quality
flags
"""
from functools import wraps


# The quality_flag field in MySQL is currently limited to 1 << 15;
# fields beyond 1 << 15 will require a change in the MySQL datatype
# for the quality_flag column.  versioned description-bitmask mapping
# dict (key is version) DO NOT MODIFY THE VALUES OF THE DICT instead,
# add a increment the key and add a new value tuple. The VERSION
# IDENTIFIER 0 - 2 must remain in their current positions. Versions 7
# and up will require another identifier bit to be determined at that
# time.  The version identifier also serves to mark data in the
# database as validated.
_BITMASK_DESCRIPTION_DICT = {1: (
    # start with 1 to distinguish validated vs not in DB
    ('OK', 0),
    ('USER FLAGGED', 1 << 0),
    ('VERSION IDENTIFIER 0', 1 << 1),
    ('VERSION IDENTIFIER 1', 1 << 2),
    ('VERSION IDENTIFIER 2', 1 << 3),
    ('NIGHTTIME', 1 << 4),
    ('CLOUDY', 1 << 5),
    ('SHADED', 1 << 6),
    ('UNEVEN FREQUENCY', 1 << 7),
    ('LIMITS EXCEEDED', 1 << 8),
    ('CLEARSKY EXCEEDED', 1 << 9),
    ('STALE VALUES', 1 << 10),
    ('INTERPOLATED VALUES', 1 << 11),
    ('CLIPPED VALUES', 1 << 12),
    ('INCONSISTENT IRRADIANCE COMPONENTS', 1 << 13),
    ('RESERVED 0', 1 << 14),  # available for new flag
    ('RESERVED 1', 1 << 15),  # available for new flag
    )
}


LATEST_VERSION = max(_BITMASK_DESCRIPTION_DICT.keys())
LATEST_BITMASK_DESCRIPTION = _BITMASK_DESCRIPTION_DICT[LATEST_VERSION]
LATEST_VERSION_FLAG = (
    LATEST_VERSION * LATEST_BITMASK_DESCRIPTION['VERSION IDENTIFIER 0'])
DESCRIPTION_MASK_MAPPING = {v[0]: v[1] for v in LATEST_BITMASK_DESCRIPTION}
MASK_DESCRIPTION_MAPPING = {v[1]: v[0] for v in LATEST_BITMASK_DESCRIPTION}


def convert_bool_flags_to_flag_mask(flags, flag_description, invert):
    if flags is None:
        return None
    if invert:
        bool_flags = ~(flags.astype(bool))
    else:
        bool_flags = flags.astype(bool)
    return ((bool_flags * DESCRIPTION_MASK_MAPPING[flag_description])
            | LATEST_VERSION_FLAG)


def mask_flags(f, flag_description, invert=True):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return_bool = kwargs.pop('return_bool', True)
        flags = f(*args, **kwargs)
        if not return_bool:
            if isinstance(flags, tuple):
                return (convert_bool_flags_to_flag_mask(
                    f, flag_description, invert) for f in flags)
            else:
                return convert_bool_flags_to_flag_mask(flags, flag_description,
                                                       invert)
        else:
            return flags
    return wrapper


# function to check if int is flagged for given test first, find
# version identifiers of the latest if 0 - 2 >= 7 not implemented for
# now
