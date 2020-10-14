"""
Define constant mappings between bit-mask values and understandable quality
flags
"""
from functools import wraps


import pandas as pd
import numpy as np


# The quality_flag field in MySQL is currently limited to 1 << 15;
# fields beyond 1 << 15 will require a change in the MySQL datatype
# for the quality_flag column. The mapping from description to bitmask
# is versioned so that future addtions or removals are backwards compatible
# without rerunning the validation on all data.
# DO NOT MODIFY THE VALUES OF THE _BITMASK_DESCRIPTION_DICT instead,
# add a increment the key and add a new value tuple, i.e. add version 2 like
# 2: {'OK': 0, 'USER FLAGGED: 1 << 0, ...} . The VERSION
# IDENTIFIER 0 - 2 must remain in their current positions. Versions 7
# and up will require another identifier bit to be determined at that
# time.  The version identifier also serves to mark data in the
# database as validated. The tuples are (description, bit mask)
BITMASK_DESCRIPTION_DICT = {1: {
    # start with 1 to distinguish validated vs not in DB
    'OK': 0,
    'USER FLAGGED': 1 << 0,
    'VERSION IDENTIFIER 0': 1 << 1,
    'VERSION IDENTIFIER 1': 1 << 2,
    'VERSION IDENTIFIER 2': 1 << 3,
    'NIGHTTIME': 1 << 4,
    'CLEARSKY': 1 << 5,
    'SHADED': 1 << 6,
    'UNEVEN FREQUENCY': 1 << 7,
    'LIMITS EXCEEDED': 1 << 8,
    'CLEARSKY EXCEEDED': 1 << 9,
    'STALE VALUES': 1 << 10,
    'INTERPOLATED VALUES': 1 << 11,
    'CLIPPED VALUES': 1 << 12,
    'INCONSISTENT IRRADIANCE COMPONENTS': 1 << 13,
    'DAILY VALIDATION APPLIED': 1 << 14,
    'RESERVED 1': 1 << 15  # available for new flag
    }
}

# logical combinations of the masks defined above.
# add a version layer for compatibility if needed in the future.
# derived masks may reference masks defined in an earlier key.
DERIVED_MASKS = {
    'DAYTIME': (np.logical_not, 'NIGHTTIME'),
    'DAYTIME STALE VALUES': (np.logical_and, 'DAYTIME', 'STALE VALUES'),
    'DAYTIME INTERPOLATED VALUES': (
        np.logical_and, 'DAYTIME', 'INTERPOLATED VALUES'),
}

# flags that should typically be discarded before resampling because they
# represent truly bad data
DISCARD_BEFORE_RESAMPLE = [
    'USER FLAGGED', 'LIMITS EXCEEDED', 'INCONSISTENT IRRADIANCE COMPONENTS'
]

# should never change unless another VERSION IDENTIFIER is required
VERSION_MASK = 0b1110
LATEST_VERSION = max(BITMASK_DESCRIPTION_DICT.keys())
DESCRIPTION_MASK_MAPPING = BITMASK_DESCRIPTION_DICT[LATEST_VERSION]
LATEST_VERSION_FLAG = (
    LATEST_VERSION * DESCRIPTION_MASK_MAPPING['VERSION IDENTIFIER 0'])
DAILY_VALIDATION_FLAG = DESCRIPTION_MASK_MAPPING['DAILY VALIDATION APPLIED']


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
    Decorator that will convert a boolean pandas object into an integer,
    bitmasked object when `_return_mask=True`. This decorator adds the
    `_return_mask` kwarg to the decorated function. Using this decorator
    to mask values ensures the description and decorated function are
    clearly linked.

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
            return_mask = kwargs.pop('_return_mask', False)
            flags = f(*args, **kwargs)
            if return_mask:
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
    """Return True (or a boolean series) if flags has been validated"""
    return flags > 1


def get_version(flag):
    """Extract the version from flag"""
    # will be more complicated if another version identifier must be added
    return np.right_shift(flag & VERSION_MASK, 1)


def _flag_description_checks(flag_description):
    if isinstance(flag_description, str):
        return
    else:
        if len(flag_description) == 0:
            raise TypeError('flag_description must have len > 0')
        for k in iter(flag_description):
            if not isinstance(k, str):
                raise TypeError(
                    'Elements of flag_description must have type str')


def check_if_single_value_flagged(flag, flag_description,
                                  _perform_checks=True):
    """Check if the single integer flag has been flagged for flag_description

    Parameters
    ----------
    flag : integer
        Integer flag
    flag_description : string or iterable of strings
        Checks to compare againsts flag


    Returns
    -------
    Boolean
        Whether any of `flag_description` checks are represented by `flag`

    Raises
    ------
    ValueError
        If flag has not been validated
    TypeError
        If flag_description is not a string or iterable of strings
    KeyError
        If flag_description is not a possible check for the flag version
    """
    if _perform_checks:
        if not has_data_been_validated(flag):
            raise ValueError('Data has not been validated')
        _flag_description_checks(flag_description)
    mask_dict = BITMASK_DESCRIPTION_DICT[get_version(flag)]
    if isinstance(flag_description, str):
        mask = mask_dict[flag_description]
        ok_mask = mask == 0
    else:
        mask = 0
        ok_mask = False
        for k in flag_description:
            m = mask_dict[k]
            if m == 0:
                ok_mask = True
            mask |= m

    out = bool(flag & mask)
    if ok_mask:
        out |= which_data_is_ok(flag)
    return out


def which_data_is_ok(flags):
    """Return True for flags that have been validated and are OK"""
    return (flags & ~VERSION_MASK == 0) & has_data_been_validated(flags)


def _make_mask_series(version):
    descriptions = [k for k in BITMASK_DESCRIPTION_DICT[version].keys()
                    if not (k.startswith('VERSION') or
                            k.startswith('RESERVED') or k == 'OK')]
    masks = [BITMASK_DESCRIPTION_DICT[version][desc]
             for desc in descriptions]
    return pd.Series(masks, index=descriptions)


def check_for_all_descriptions(flag, _check_if_validated=True):
    """
    Return a boolean Series indicating the checks a flag represents
    """
    if _check_if_validated and not has_data_been_validated(flag):
        raise ValueError('Data has not been validated')
    version = get_version(flag)
    mask_series = _make_mask_series(version)
    out = (mask_series & flag).astype(bool)
    return out


def _convert_version_mask(ser):
    version = ser.name
    if version == 0:
        return pd.DataFrame({'NOT VALIDATED': [True] * len(ser.index)},
                            index=ser.index)
    mask_series = _make_mask_series(version)
    out = pd.DataFrame(
        np.bitwise_and(mask_series.values[None, :],
                       ser.values[:, None]),
        columns=mask_series.index,
        index=ser.index,
        dtype=bool)
    out['NOT VALIDATED'] = False
    return out


def _add_derived_masks(masks):
    """Copies input DataFrame and then adds new masks derived from
    input masks"""
    unvalidated = masks['NOT VALIDATED']
    if unvalidated.all():
        return masks
    out = masks.copy()[~unvalidated]
    for flag, operations in DERIVED_MASKS.items():
        func = operations[0]
        cols = operations[1:]
        args = [out[col] for col in cols]
        out[flag] = func(*args)
    return pd.concat([out, masks[unvalidated]], sort=False).fillna(False)


def convert_mask_into_dataframe(flag_series):
    """
    Convert `flag_series` into a boolean DataFrame indicating which checks
    the flags represent.

    Parameters
    ----------
    flag_series : pandas.Series
        Integer series of validated quality flags

    Returns
    -------
    pandas.DataFrame
       Columns are keys of BITMASK_DESCRIPTION_DICT and values are booleans
       indicating if the input flag corresponds to the given check. An
       additional column, NOT VALIDATED, indicates if the data has not
       been validated. Additional columns defined by DERIVED_MASKS are
       computed based on the results of the fundamental flags.
       Columns may vary depending the version of the quality
       flags in the series.
    """
    vers = get_version(flag_series)
    fundamental_masks = flag_series.groupby(vers, sort=False).apply(
        _convert_version_mask).fillna(False)
    out = _add_derived_masks(fundamental_masks)
    return out


def convert_flag_frame_to_strings(flag_frame, sep=', ', empty='OK'):
    """
    Convert the `flag_frame` output of :py:func:`~convert_mask_into_dataframe`
    into a pandas.Series of strings which are the active flag names separated
    by `sep`. Any row where all columns are false will have a value of `empty`.

    Parameters
    ----------
    flag_frame : pandas.DataFrame
        Boolean DataFrame with descriptive column names
    sep : str
        String to separate column names by
    empty : str
        String to replace rows where no columns are True

    Returns
    -------
    pandas.Series
        Of joined column names from `flag_frame` separated by `sep` if True.
        Has the same index as `flag_frame`.
    """
    return np.logical_and(flag_frame, flag_frame.columns + sep).replace(
        False, '').sum(axis=1).str.rstrip(sep).replace('', empty)


def check_if_series_flagged(flag_series, flag_description):
    """
    Check if `flag_series` has been flagged for the checks given by
    flag_description

    Parameters
    ----------
    flag_series : pandas.Series
        Series of integer quality flags
    flag_description : string or iterable of strings
        Checks to compare `flag_series` to. If this is an iterable, the result
        will be a boolean indicating if the flag represents *ANY* of the
        checks.

    Returns
    -------
    pandas.Series
        Boolean Series indicating if *ANY* of `flag_description` checks are
        represented by each flag

    Raises
    ------
    ValueError
        If any of `flag_series` has not been validated.
    TypeError
        If flag_description is not a string or iterable of strings
    KeyError
        If flag_description is not a possible check for the flag version
    """
    if not has_data_been_validated(flag_series).all():
        raise ValueError('Data has not been validated')
    _flag_description_checks(flag_description)
    return flag_series.apply(check_if_single_value_flagged,
                             flag_description=flag_description,
                             _perform_checks=False)
