"""
Define constant mappings between bit-mask values and understandable quality
flags
"""
from functools import wraps


# The quality_flag field in MySQL is currently limited to 1 << 15;
# fields beyond 1 << 15 will require a change in the MySQL datatype
# for the quality_flag column. The mapping from description to bitmask
# is versioned so that future addtions or removals are backwards compatible
# without rerunning the validation on all data.
# DO NOT MODIFY THE VALUES OF THE _BITMASK_DESCRIPTION_DICT instead,
# add a increment the key and add a new value tuple. The VERSION
# IDENTIFIER 0 - 2 must remain in their current positions. Versions 7
# and up will require another identifier bit to be determined at that
# time.  The version identifier also serves to mark data in the
# database as validated. The tuples are (description, bit mask, flag class)
_BITMASK_DESCRIPTION_DICT = {1: (
    # start with 1 to distinguish validated vs not in DB
    ('OK', 0, ''),
    ('USER FLAGGED', 1 << 0, 'bad'),
    ('VERSION IDENTIFIER 0', 1 << 1, ''),
    ('VERSION IDENTIFIER 1', 1 << 2, ''),
    ('VERSION IDENTIFIER 2', 1 << 3, ''),
    ('NIGHTTIME', 1 << 4, 'informational'),
    ('CLOUDY', 1 << 5, 'informational'),
    ('SHADED', 1 << 6, 'informational'),
    ('UNEVEN FREQUENCY', 1 << 7, 'bad'),
    ('LIMITS EXCEEDED', 1 << 8, 'bad'),
    ('CLEARSKY EXCEEDED', 1 << 9, 'bad'),
    ('STALE VALUES', 1 << 10, 'bad'),
    ('INTERPOLATED VALUES', 1 << 11, 'bad'),
    ('CLIPPED VALUES', 1 << 12, 'bad'),
    ('INCONSISTENT IRRADIANCE COMPONENTS', 1 << 13, 'bad'),
    ('RESERVED 0', 1 << 14, ''),  # available for new flag
    ('RESERVED 1', 1 << 15, ''),  # available for new flag
    )
}


# should never change unless another VERSION IDENTIFIER is required
VERSION_MASK = 0b1110
LATEST_VERSION = max(_BITMASK_DESCRIPTION_DICT.keys())
LATEST_BITMASK_DESCRIPTION = _BITMASK_DESCRIPTION_DICT[LATEST_VERSION]
DESCRIPTION_MASK_MAPPING = {v[0]: v[1] for v in LATEST_BITMASK_DESCRIPTION}
MASK_DESCRIPTION_MAPPING = {v[1]: v[0] for v in LATEST_BITMASK_DESCRIPTION}
LATEST_VERSION_FLAG = (
    LATEST_VERSION * DESCRIPTION_MASK_MAPPING['VERSION IDENTIFIER 0'])


def convert_bool_flags_to_flag_mask(flags, flag_description, invert):
    if flags is None:
        return None
    if invert:
        bool_flags = ~(flags.astype(bool))
    else:
        bool_flags = flags.astype(bool)
    return ((bool_flags * DESCRIPTION_MASK_MAPPING[flag_description])
            | LATEST_VERSION_FLAG)


def mask_flags(flag_description, invert=True):
    """
    Decorator to convert boolean flags to the integer of flag_description.
    Will only apply if the 'return_bool' keyword is pass to the decorated
    function is is False.

    Parameters
    ----------
    flag_description : str
        Description of the flag to convert from a boolean to integer. Must be
        a key of the DESCRIPTION_MASK_MAPPING dict.
    invert : boolean
        Whether to invert the boolean object before conversion e.g. if
        flag_description = 'LIMITS EXCEEDED' and a True value indicates
        that a parameter is within the limits, invert=True is required
        for the proper mapping.

    Returns
    -------
    flags : pandas Object
        Returns the output of the decorated function (which must be a pandas
        Object) as the original output or an object of type int with value
        determined by the truthiness of the orignal output and flag_description
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return_bool = kwargs.pop('return_bool', True)
            flags = f(*args, **kwargs)
            if not return_bool:
                if isinstance(flags, tuple):
                    return tuple(convert_bool_flags_to_flag_mask(
                        f, flag_description, invert) for f in flags)
                else:
                    return convert_bool_flags_to_flag_mask(
                        flags, flag_description, invert)
            else:
                return flags
        return wrapper
    return decorator


def has_data_been_validated(flags):
    return flags > 1


def get_version(flag):
    # will be more complicated if another version identifier must be added
    return (flag & VERSION_MASK) >> 1


def _get_mask_dict(flag):
    version = get_version(flag)
    return {v[0]: v[1] for v in _BITMASK_DESCRIPTION_DICT[version]}


def check_if_single_value_flagged(flag, flag_description):
    """Check if the single integer flag has been flagged for flag_description"""
    if not has_data_been_validated(flag):
        raise ValueError('Data has not been validated')
    mask_dict = _get_mask_dict(flag)
    mask = mask_dict[flag_description]
    if mask == 0:
        return which_data_is_ok(flag)
    else:
        return bool(flag & mask)


def which_data_is_ok(flags):
    """Return True for flags that have been validated and are OK"""
    return has_data_been_validated(flags) & ~VERSION_MASK == 0
