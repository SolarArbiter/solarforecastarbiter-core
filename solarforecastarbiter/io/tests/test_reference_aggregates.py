import pandas as pd
import pytest
from solarforecastarbiter.io import reference_aggregates


AGGDEF = {
    'name': 'Test Aggregate ghi',
    'description': 'ghi agg',
    'variable': 'ghi',
    'aggregate_type': 'mean',
    'interval_length': pd.Timedelta('1h'),
    'interval_label': 'ending',
    'timezone': 'America/Denver',
    'observations': [
        {
            'site': 'Tracking plant',
            'observation': 'GHI Instrument 1',
            'from': '2019-01-01T00:00Z',
            'until': '2020-01-01T00:00Z',
        },
        {
            'site': 'Weather Station 1',
            'observation': 'GHI Instrument 2',
            'from': '2019-01-01T00:00Z',
        },
        {
            'site': 'Weather Station 1',
            'observation': 'Sioux Falls, ghi',
            'from': '2019-01-01T00:00Z',
            'until': None,
            'deleted_at': None
        },
    ]
}


def test_generate_aggregate(aggregate, many_observations):
    new_agg = reference_aggregates.generate_aggregate(many_observations,
                                                      AGGDEF)
    ignore = ('extra_parameters', 'provider', 'aggregate_id')
    for k in new_agg.to_dict().keys():
        if k in ignore:
            continue
        assert getattr(aggregate, k) == getattr(new_agg, k)
    assert len(new_agg.observations) == 3


def test_generate_aggregate_fail(many_observations):
    with pytest.raises(ValueError):
        reference_aggregates.generate_aggregate([], AGGDEF)

    manymany = many_observations + many_observations
    with pytest.raises(ValueError):
        reference_aggregates.generate_aggregate(manymany, AGGDEF)


def test_make_reference_aggregates(mocker, many_observations, aggregate):
    sess = mocker.MagicMock()
    mocker.patch(
        'solarforecastarbiter.io.reference_aggregates.api.APISession',
        return_value=sess
    )
    sess.list_observations.return_value = many_observations
    sess.list_aggregates.return_value = []
    reference_aggregates.make_reference_aggregates('token', 'Organization 1',
                                                   'base', [AGGDEF])

    new_agg = sess.create_aggregate.call_args_list[0][0][0]
    ignore = ('extra_parameters', 'provider', 'aggregate_id')
    for k in new_agg.to_dict().keys():
        if k in ignore:
            continue
        assert getattr(aggregate, k) == getattr(new_agg, k)


def test_make_reference_aggregates_exists(mocker, many_observations,
                                          aggregate):
    sess = mocker.MagicMock()
    mocker.patch(
        'solarforecastarbiter.io.reference_aggregates.api.APISession',
        return_value=sess
    )
    sess.list_observations.return_value = many_observations
    sess.list_aggregates.return_value = [aggregate]
    reference_aggregates.make_reference_aggregates('token', 'Organization 1',
                                                   'base', [AGGDEF])
    sess.create_aggregate.assert_not_called()
