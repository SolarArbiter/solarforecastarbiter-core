import datetime as dt
import pytest


from solarforecastarbiter.io.reference_observations import srml


@pytest.mark.parametrize('start,end,exp', [
    (dt.datetime(2019, 1, 10), dt.datetime(2019, 3, 11),
     [(2019, 1), (2019, 2), (2019, 3)]),
    (dt.datetime(2019, 11, 10), dt.datetime(2020, 2, 11),
     [(2019, 11), (2019, 12), (2020, 1), (2020, 2)]),
    (dt.datetime(2019, 11, 10), dt.datetime(2019, 11, 11),
     [(2019, 11)]),
    (dt.datetime(2019, 11, 10), dt.datetime(2021, 2, 11),
     [(2019, 11), (2019, 12), (2020, 1), (2020, 2), (2020, 3),
      (2020, 4), (2020, 5), (2020, 6), (2020, 7), (2020, 8),
      (2020, 9), (2020, 10), (2020, 11), (2020, 12), (2021, 1),
      (2021, 2)]),
    (dt.datetime(2019, 10, 1), dt.datetime(2019, 9, 1), []),
    (dt.datetime(2019, 10, 1), dt.datetime(2018, 11, 1), [])
])
def test_fetch(mocker, single_site, start, end, exp):
    rd = mocker.patch(
        'solarforecastarbiter.io.reference_observations.srml.request_data',
        return_value=None)
    srml.fetch('', single_site, start, end)
    year_month = [(ca[0][1], ca[0][2]) for ca in rd.call_args_list]
    assert year_month == exp
