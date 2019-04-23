import json
import pandas as pd
import pandas.testing as pdt
import pytest

from solarforecastarbiter.io import utils

# data for test Dataframe
TEST_DICT = {'var_1': [2.0, 43.9, 338.0, -199.7, 0.32],
             'var_1_flag': [0, 0, 1, 3, 3],
             'var_2': [33.88, 3.22, 1e-2, 248.71, 48.7],
             'var_2_flag': [1, 1, 1, 1, 1]}

DF_INDEX = pd.DatetimeIndex(start=pd.Timestamp('2019-01-24T00:00'),
                            freq='1min',
                            periods=5,
                            tz='UTC', name='timestamp')
TEST_DATA = pd.DataFrame(TEST_DICT, index=DF_INDEX)


@pytest.mark.parametrize('df,var_label,flag_label,flag_value', [
    (TEST_DATA, 'var_1', 'var_1_flag', 0),
    (TEST_DATA, 'var_1', None, 0),
    (TEST_DATA, 'var_2', 'var_2_flag', 1),
])
def test_obs_df_to_json(df, var_label, flag_label, flag_value):
    converted = utils.observation_df_to_json_payload(df, var_label, flag_label)
    converted_dict = json.loads(converted)
    assert 'values' in converted_dict
    values = converted_dict['values']
    assert len(values) == 5
    assert values[0]['timestamp'] == '2019-01-24T00:00:00Z'
    assert values[0]['questionable'] == flag_value
    assert isinstance(values[0]['value'], float)


TEST_JSON = """
{
    "_links": {},
    "observation_id": "OBSID",
    "values": [
        {
            "quality_flag": 0,
            "timestamp": "2019-01-24T00:00:00Z",
            "value": 2.0
        },
        {
            "quality_flag": 0,
            "timestamp": "2019-01-24T00:01:00Z",
            "value": 43.9
        },
        {
            "quality_flag": 1,
            "timestamp": "2019-01-24T00:02:00Z",
            "value": 338.0
        },
        {
            "quality_flag": 3,
            "timestamp": "2019-01-24T00:03:00Z",
            "value": -199.7
        },
        {
            "quality_flag": 3,
            "timestamp": "2019-01-23T17:04:00-07:00",
            "value": 0.32
        }
    ]
}
"""


def test_json_payload_to_observation_df():
    expected = pd.DataFrame({'value': TEST_DICT['var_1'],
                             'quality_flag': TEST_DICT['var_1_flag']},
                            index=DF_INDEX)
    out = utils.json_payload_to_observation_df(json.loads(TEST_JSON))
    pdt.assert_frame_equal(expected, out)
