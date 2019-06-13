from dataclasses import replace
import datetime as dt


import pandas as pd
import pytest


from solarforecastarbiter import utils


@pytest.mark.parametrize('issuetime,rl,lt,expected', [
    ('06:00', '1h', '1h', [dt.time(i) for i in range(6, 24)]),
    ('00:00', '4h', '1h', [dt.time(0), dt.time(4), dt.time(8), dt.time(12),
                           dt.time(16), dt.time(20)]),
    ('16:00', '8h', '3h', [dt.time(16)]),
    ('00:30', '4h', '120h', [dt.time(0, 30), dt.time(4, 30), dt.time(8, 30),
                             dt.time(12, 30), dt.time(16, 30),
                             dt.time(20, 30)])
])
def test_issue_times(single_forecast, issuetime, rl, lt, expected):
    fx = replace(
        single_forecast,
        issue_time_of_day=dt.datetime.strptime(issuetime, '%H:%M').time(),
        run_length=pd.Timedelta(rl),
        lead_time_to_start=pd.Timedelta(lt))
    out = utils.issue_times(fx)
    assert out == expected


@pytest.mark.parametrize('issuetime,rl,lt,start,expected', [
    ('05:00', '6h', '1h', pd.Timestamp('20190101T0000Z'),
     [pd.Timestamp('20190101T0500Z'), pd.Timestamp('20190101T1100Z'),
      pd.Timestamp('20190101T1700Z'), pd.Timestamp('20190101T2300Z')]),
    ('11:00', '12h', '3h', pd.Timestamp('20190101T2300Z'),
     [pd.Timestamp('20190101T1100Z'), pd.Timestamp('20190101T2300Z')])
])
def test_issue_times_start(single_forecast, issuetime, rl, lt, start,
                           expected):
    fx = replace(
        single_forecast,
        issue_time_of_day=dt.datetime.strptime(issuetime, '%H:%M').time(),
        run_length=pd.Timedelta(rl),
        lead_time_to_start=pd.Timedelta(lt))
    out = utils.issue_times(fx, start)
    assert out == expected
