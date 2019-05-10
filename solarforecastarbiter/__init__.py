from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

try:
    import sentry_sdk  # NOQA
except ImportError:
    pass
else:
    # Must set SENTRY_DSN for this to do anything
    sentry_sdk.init(send_default_pii=False,
                    release=f'solarforecastarbiter-core@{__version__}')
