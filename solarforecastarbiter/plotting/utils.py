from solarforecastarbiter.datamodel import ALLOWED_VARIABLES, COMMON_NAMES


def format_variable_name(variable):
    """Make a human readable name for the variable"""
    return f'{COMMON_NAMES[variable]} ({ALLOWED_VARIABLES[variable]})'
