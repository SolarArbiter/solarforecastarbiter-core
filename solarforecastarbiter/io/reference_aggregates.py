"""
Define some reference aggregates based on *exact* site names
"""
import logging


import pandas as pd


from solarforecastarbiter import datamodel
from solarforecastarbiter.io import api


logger = logging.getLogger('reference_aggregates')


REF_AGGREGATES = [
    {
        'name': 'NOAA SURFRAD Average GHI',
        'description': 'Average GHI across all SURFRAD sites',
        'variable': 'ghi',
        'aggregate_type': 'mean',
        'interval_length': pd.Timedelta('1h'),
        'interval_label': 'ending',
        'timezone': 'Etc/GMT+6',
        'observations': [
            {
                'site': 'NOAA SURFRAD Bondville IL',
                'observation': 'Bondville IL ghi',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Table Mountain Boulder CO',
                'observation': 'Table Mountain Boulder CO ghi',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Desert Rock NV',
                'observation': 'Desert Rock NV ghi',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Fort Peck MT',
                'observation': 'Fort Peck MT ghi',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Goodwin Creek MS',
                'observation': 'Goodwin Creek MS ghi',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Penn State Univ PA',
                'observation': 'Penn State Univ PA ghi',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Sioux Falls SD',
                'observation': 'Sioux Falls SD ghi',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
        ]
    }, {
        'name': 'NOAA SURFRAD Average DNI',
        'description': 'Average DNI across all SURFRAD sites',
        'variable': 'dni',
        'aggregate_type': 'mean',
        'interval_length': pd.Timedelta('1h'),
        'interval_label': 'ending',
        'timezone': 'Etc/GMT+6',
        'observations': [
            {
                'site': 'NOAA SURFRAD Bondville IL',
                'observation': 'Bondville IL dni',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Table Mountain Boulder CO',
                'observation': 'Table Mountain Boulder CO dni',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Desert Rock NV',
                'observation': 'Desert Rock NV dni',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Fort Peck MT',
                'observation': 'Fort Peck MT dni',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Goodwin Creek MS',
                'observation': 'Goodwin Creek MS dni',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Penn State Univ PA',
                'observation': 'Penn State Univ PA dni',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
            {
                'site': 'NOAA SURFRAD Sioux Falls SD',
                'observation': 'Sioux Falls SD dni',
                'from': '2017-01-01T00:00Z',
                'until': None
            },
        ]
    }, {
        'name': 'UO SRML Portland PV',
        'description': 'Sum of a number of small PV systems in Portland, OR',
        'variable': 'ac_power',
        'aggregate_type': 'sum',
        'interval_length': pd.Timedelta('1h'),
        'interval_label': 'ending',
        'timezone': 'Etc/GMT+8',
        'observations': [
            {
                'site': 'UO SRML Portland OR PV 15 deg tilt',
                'observation': 'Portland OR PV 15 deg tilt ac_power Unisolar',
                'from': '2019-01-01T08:00Z',
                'until': None,
            },
            {
                'site': 'UO SRML Portland OR PV 30 deg tilt',
                'observation': 'Portland OR PV 30 deg tilt ac_power Evergreen',
                'from': '2019-01-01T08:00Z',
                'until': None,
            },
            {
                'site': 'UO SRML Portland OR PV 30 deg tilt',
                'observation': 'Portland OR PV 30 deg tilt ac_power Photowatt',
                'from': '2019-01-01T08:00Z',
                'until': None,
            },
            {
                'site': 'UO SRML Portland OR PV 30 deg tilt',
                'observation': 'Portland OR PV 30 deg tilt ac_power Kaneka',
                'from': '2019-01-01T08:00Z',
                'until': None,
            },
            {
                'site': 'UO SRML Portland OR PV 30 deg tilt',
                'observation': 'Portland OR PV 30 deg tilt ac_power Sanyo',
                'from': '2019-01-01T08:00Z',
                'until': None,
            },
        ]
    }
]


def generate_aggregate(observations, agg_def):
    """Generate an aggregate object.

    Parameters
    ----------
    observations: list of datamodel.Observation
    agg_def: dict
        Text metadata to create a datamodel.Aggregate.
        'observation' field names will be matched against names
        of datamodel.Observation in ``observations``.

    Returns
    -------
    datamodel.Aggregate

    Raises
    ------
    ValueError
        If an observation does not exist in the API for an aggregate
        or multiple observations match the given name and site name.
    """
    agg_obs = []
    limited_obs = list(filter(lambda x: x.variable == agg_def['variable'],
                              observations))
    # go through each agg_dev.observations[*] and find the Observation
    # from the API corresponding to the site name and observation name
    # to make AggregateObservations and Aggregate
    for obs in agg_def['observations']:
        candidates = list(filter(
            lambda x: x.name == obs['observation'] and
            x.site.name == obs['site'],
            limited_obs
        ))
        if len(candidates) == 0:
            raise ValueError(
                f'No observations match site: {obs["site"]},'
                f' name: {obs["observation"]}')
        elif len(candidates) > 1:
            raise ValueError(
                f'Multiple observations match site: {obs["site"]},'
                f' name: {obs["observation"]}'
            )
        else:
            agg_obs.append(
                datamodel.AggregateObservation(
                    observation=candidates[0],
                    effective_from=pd.Timestamp(obs['from']),
                    effective_until=(
                        None if obs.get('until') is None
                        else pd.Timestamp(obs['until'])),
                    observation_deleted_at=(
                        None if obs.get('deleted_at') is None
                        else pd.Timestamp(obs['deleted_at'])
                    )
                )
            )
    agg_dict = agg_def.copy()
    agg_dict['observations'] = tuple(agg_obs)
    agg = datamodel.Aggregate(**agg_dict)
    return agg


def make_reference_aggregates(token, provider, base_url,
                              aggregates=None):
    """Create the reference aggregates in the API.

    Parameters
    ----------
    token: str
        Access token for the API
    provider: str
        Provider name to filter all API observations on
    base_url: str
        URL of the API to list objects and create aggregate at
    aggregates: list or None
        List of dicts that describes each aggregate. Defaults to
        REF_AGGREGATES if None.

    Raises
    ------
    ValueError
        If an observation does not exist in the API for an aggregate
        or multiple observations match the given name and site name.
    """
    session = api.APISession(token, base_url=base_url)
    observations = list(filter(lambda x: x.provider == provider,
                               session.list_observations()))
    existing_aggregates = {ag.name for ag in session.list_aggregates()}
    if aggregates is None:
        aggregates = REF_AGGREGATES
    for agg_def in aggregates:
        if agg_def['name'] in existing_aggregates:
            logger.warning('Aggregate %s already exists', agg_def['name'])
            # TODO: update the aggregate if the definition has changed
            continue
        logger.info('Creating aggregate %s', agg_def['name'])
        agg = generate_aggregate(observations, agg_def)
        # allow create to raise any API errors
        session.create_aggregate(agg)
