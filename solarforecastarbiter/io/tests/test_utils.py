import json
import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter.io import utils

# data for test Dataframe
test_dict = {'var_1': np.random.rand(5),
             'var_1_flag': np.zeros(5),
             'var_2': np.random.rand(5),
             'var_2_flag': np.ones(5)}

df_index = pd.DatetimeIndex(start=pd.Timestamp('2019-01-24T00:00'),
                            freq='1min',
                            periods=5,
                            tz='UTC')
test_data = pd.DataFrame(test_dict, index=df_index)


@pytest.mark.parametrize('df,var_label,flag_label,flag_value', [
    (test_data, 'var_1', 'var_1_flag', 0),
    (test_data, 'var_1', None, 0),
    (test_data, 'var_2', 'var_2_flag', 1),
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
