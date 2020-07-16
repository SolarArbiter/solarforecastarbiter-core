import logging


import pandas as pd
import pandas.testing as pdt
import pytest


from solarforecastarbiter import utils


def _make_aggobs(obsid, ef=pd.Timestamp('20191001T1100Z'),
                 eu=None, oda=None):
    return {
        'observation_id': obsid,
        'effective_from': ef,
        'effective_until': eu,
        'observation_deleted_at': oda
    }


nindex = pd.date_range(start='20191004T0000Z',
                       freq='1h', periods=10)


@pytest.fixture()
def ids():
    return ['f2844284-ea0a-11e9-a7da-f4939feddd82',
            'f3e310ba-ea0a-11e9-a7da-f4939feddd82',
            '09ed7cf6-ea0b-11e9-a7da-f4939feddd82',
            '0fe9f2ba-ea0b-11e9-a7da-f4939feddd82',
            '67ea9200-ea0e-11e9-832b-f4939feddd82']


@pytest.fixture()
def aggobs(ids):
    return tuple([
        _make_aggobs(ids[0]),
        _make_aggobs(ids[1], pd.Timestamp('20191004T0501Z')),
        _make_aggobs(ids[2], eu=pd.Timestamp('20191004T0400Z')),
        _make_aggobs(ids[2], pd.Timestamp('20191004T0700Z'),
                     eu=pd.Timestamp('20191004T0800Z')),
        _make_aggobs(ids[2], pd.Timestamp('20191004T0801Z')),
        _make_aggobs(ids[3], oda=pd.Timestamp('20191005T0000Z')),
        _make_aggobs(ids[4], oda=pd.Timestamp('20191009T0000Z'),
                     eu=pd.Timestamp('20191003T0000Z'))
    ])


def test_compute_aggregate(aggobs, ids):
    data = {id_: pd.DataFrame({'value': [1] * 10, 'quality_flag': [0] * 10},
                              index=nindex)
            for id_ in ids[:3]}
    agg = utils.compute_aggregate(data, '1h', 'ending',
                                  'UTC', 'sum', aggobs[:-2])
    pdt.assert_frame_equal(agg, pd.DataFrame(
        {'value': pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                            index=nindex),
         'quality_flag':  pd.Series([0]*10, index=nindex)})
        )


def test_compute_aggregate_missing_from_data(aggobs, ids):
    data = {id_: pd.DataFrame({'value': [1] * 10, 'quality_flag': [0] * 10},
                              index=nindex)
            for id_ in ids[:3]}
    aggobs = list(aggobs[:-2]) + [
        _make_aggobs('09ed7cf6-ea0b-11e9-a7da-f4939fed889')]
    with pytest.raises(KeyError):
        utils.compute_aggregate(data, '1h', 'ending',
                                'UTC', 'sum', aggobs)


def test_compute_aggregate_empty_data(aggobs, ids):
    data = {}
    with pytest.raises(KeyError):
        utils.compute_aggregate(data, '1h', 'ending',
                                'UTC', 'sum', aggobs[:2])


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_compute_aggregate_missing_data(aggobs, ids):
    data = {id_: pd.DataFrame({'value': [1] * 10, 'quality_flag': [0] * 10},
                              index=nindex)
            for id_ in ids[:3]}
    data[ids[-1]] = pd.DataFrame({'value': [1] * 8, 'quality_flag': [0] * 8},
                                 index=nindex[:-2])
    aggobs = list(aggobs[:-2]) + [_make_aggobs(ids[-1])]
    agg = utils.compute_aggregate(data, '1h', 'ending',
                                  'UTC', 'sum', aggobs)
    pdt.assert_frame_equal(agg, pd.DataFrame(
        {'value': pd.Series(
            [3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 4.0, None, None],
            index=nindex),
         'quality_flag':  pd.Series([0]*10, index=nindex)})
        )


def test_compute_aggregate_deleted_not_removed(aggobs, ids):
    data = {id_: pd.DataFrame({'value': [1] * 10, 'quality_flag': [0] * 10},
                              index=nindex)
            for id_ in ids}
    with pytest.raises(ValueError):
        utils.compute_aggregate(data, '1h', 'ending',
                                'UTC', 'sum', aggobs)


def test_compute_aggregate_deleted_not_removed_yet(aggobs, ids):
    data = {id_: pd.DataFrame({'value': [1] * 10, 'quality_flag': [0] * 10},
                              index=nindex)
            for id_ in ids[:3]}
    # with last aggobs, would try and get data before effective_until,
    # but was deleted, so raise error
    aggobs = list(aggobs[:-2]) + [
        _make_aggobs(ids[4], oda=pd.Timestamp('20191009T0000Z'),
                     eu=pd.Timestamp('20191004T0700Z'))]
    with pytest.raises(ValueError):
        utils.compute_aggregate(data, '1h', 'ending',
                                'UTC', 'sum', aggobs)


def test_compute_aggregate_deleted_but_removed_before(aggobs, ids):
    data = {id_: pd.DataFrame({'value': [1] * 10, 'quality_flag': [0] * 10},
                              index=nindex)
            for id_ in ids[:3]}
    # aggobs[-1] properly removed
    aggobs = list(aggobs[:-2]) + [aggobs[-1]]
    agg = utils.compute_aggregate(data, '1h', 'ending',
                                  'UTC', 'sum', aggobs)
    pdt.assert_frame_equal(agg, pd.DataFrame(
        {'value': pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 3.0, 3.0],
                            index=nindex),
         'quality_flag':  pd.Series([0]*10, index=nindex)}))


def test_compute_aggregate_mean(aggobs, ids):
    data = {id_: pd.DataFrame({'value': [1] * 10, 'quality_flag': [0] * 10},
                              index=nindex)
            for id_ in ids[:3]}
    agg = utils.compute_aggregate(data, '1h', 'ending',
                                  'UTC', 'mean', aggobs[:-2])
    pdt.assert_frame_equal(agg, pd.DataFrame(
        {'value': pd.Series([1.0] * 10, index=nindex),
         'quality_flag':  pd.Series([0]*10, index=nindex)})
        )


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_compute_aggregate_no_overlap(ids):
    data = {ids[0]: pd.DataFrame(
        {'value': [1, 2, 3], 'quality_flag': [2, 10, 338]},
        index=pd.DatetimeIndex([
            '20191002T0100Z', '20191002T0130Z', '20191002T0230Z'])),
            ids[1]: pd.DataFrame(
        {'value': [3, 2, 1], 'quality_flag': [9, 880, 10]},
        index=pd.DatetimeIndex([
            '20191002T0200Z', '20191002T0230Z', '20191002T0300Z']))}
    aggobs = [_make_aggobs(ids[0]),
              _make_aggobs(ids[1], pd.Timestamp('20191002T0200Z'))]
    agg = utils.compute_aggregate(data, '30min', 'ending',
                                  'UTC', 'median', aggobs)
    expected = pd.DataFrame(
        {'value': [1.0, 2.0, None, 2.5, None],
         'quality_flag': [2, 10, 9, 338 | 880, 10]},
        index=pd.DatetimeIndex([
            '20191002T0100Z', '20191002T0130Z', '20191002T0200Z',
            '20191002T0230Z', '20191002T0300Z']))
    pdt.assert_frame_equal(agg, expected)


def test_compute_aggregate_missing_before_effective(ids):
    data = {ids[0]: pd.DataFrame(
        {'value': [1, 2, 3, 0, 0], 'quality_flag': [2, 10, 338, 0, 0]},
        index=pd.DatetimeIndex([
            '20191002T0100Z', '20191002T0130Z', '20191002T0200Z',
            '20191002T0230Z', '20191002T0300Z'])),
            ids[1]: pd.DataFrame(
        {'value': [None, 2.0, 1.0], 'quality_flag': [0, 880, 10]},
        index=pd.DatetimeIndex([
            '20191002T0200Z', '20191002T0230Z', '20191002T0300Z']))}
    aggobs = [_make_aggobs(ids[0]),
              _make_aggobs(ids[1], pd.Timestamp('20191002T0201Z'))]
    agg = utils.compute_aggregate(data, '30min', 'ending',
                                  'UTC', 'max', aggobs)
    expected = pd.DataFrame(
        {'value': [1.0, 2.0, 3.0, 2.0, 1.0],
         'quality_flag': [2, 10, 338, 880, 10]},
        index=pd.DatetimeIndex([
            '20191002T0100Z', '20191002T0130Z', '20191002T0200Z',
            '20191002T0230Z', '20191002T0300Z']))
    pdt.assert_frame_equal(agg, expected)


def test_compute_aggregate_bad_cols():
    data = {'a': pd.DataFrame([0], index=pd.DatetimeIndex(
        ['20191001T1200Z']))}
    with pytest.raises(KeyError):
        utils.compute_aggregate(data, '1h', 'ending', 'UTC',
                                'mean', [_make_aggobs('a')])


def test__observation_valid(aggobs):
    out = utils._observation_valid(
        nindex, 'f2844284-ea0a-11e9-a7da-f4939feddd82', aggobs)
    pdt.assert_series_equal(out, pd.Series(True, index=nindex))


def test__observation_valid_ended(aggobs):
    out = utils._observation_valid(
        nindex, 'f3e310ba-ea0a-11e9-a7da-f4939feddd82', aggobs)
    pdt.assert_series_equal(out, pd.Series([False] * 6 + [True] * 4,
                                           index=nindex))


def test__observation_valid_many(aggobs):
    out = utils._observation_valid(
        nindex, '09ed7cf6-ea0b-11e9-a7da-f4939feddd82', aggobs)
    pdt.assert_series_equal(out, pd.Series(
        [True, True, True, True, True, False, False, True, True, True],
        index=nindex))


def test__observation_valid_deleted(aggobs):
    with pytest.raises(ValueError):
        utils._observation_valid(
            nindex, '0fe9f2ba-ea0b-11e9-a7da-f4939feddd82', aggobs)


def test__observation_valid_deleted_before(aggobs):
    out = utils._observation_valid(
        nindex, '67ea9200-ea0e-11e9-832b-f4939feddd82', aggobs)
    pdt.assert_series_equal(out, pd.Series(False, index=nindex))


@pytest.mark.parametrize('length,label,expected', [
    ('15min', 'ending', pd.date_range(start='20191004T0700Z',
                                      end='20191004T0745Z',
                                      freq='15min')),
    ('15min', 'beginning', pd.date_range(
        start='20191004T0700Z', end='20191004T0745Z',
        freq='15min')),
    ('1h', 'ending', pd.DatetimeIndex(['20191004T0700Z', '20191004T0800Z'])),
    ('1h', 'beginning', pd.DatetimeIndex(['20191004T0700Z'])),
    ('20min', 'ending', pd.DatetimeIndex([
        '20191004T0700Z', '20191004T0720Z', '20191004T0740Z',
        '20191004T0800Z'])),
    ('20min', 'beginning', pd.DatetimeIndex([
        '20191004T0700Z', '20191004T0720Z', '20191004T0740Z'])),
])
def test__make_aggregate_index(length, label, expected):
    test_data = {
        0: pd.DataFrame(range(5), index=pd.date_range(
            '20191004T0700Z', freq='7min', periods=5)),  # end 35
        1: pd.DataFrame(range(4), index=pd.date_range(
            '20191004T0015-0700', freq='10min', periods=4))}  # end 45
    out = utils._make_aggregate_index(test_data, length, label, 'UTC')
    pdt.assert_index_equal(out, expected)


@pytest.mark.parametrize('length,label,expected', [
    ('15min', 'ending', pd.date_range(start='20191004T0715Z',
                                      end='20191004T0745Z',
                                      freq='15min')),
    ('15min', 'beginning', pd.date_range(
        start='20191004T0700Z', end='20191004T0730Z',
        freq='15min')),
    ('1h', 'ending', pd.DatetimeIndex(['20191004T0800Z'])),
    ('1h', 'beginning', pd.DatetimeIndex(['20191004T0700Z'])),
    ('20min', 'ending', pd.DatetimeIndex([
        '20191004T0720Z', '20191004T0740Z'])),
    ('20min', 'beginning', pd.DatetimeIndex([
        '20191004T0700Z', '20191004T0720Z'])),
])
def test__make_aggregate_index_offset_right(length, label, expected):
    test_data = {
        0: pd.DataFrame(range(6), index=pd.date_range(
            '20191004T0701Z', freq='7min', periods=6))  # end 35
        }
    out = utils._make_aggregate_index(test_data, length, label, 'UTC')
    pdt.assert_index_equal(out, expected)


@pytest.mark.parametrize('length,label,expected', [
    ('15min', 'ending', pd.date_range(start='20191004T0700Z',
                                      end='20191004T0745Z',
                                      freq='15min')),
    ('15min', 'beginning', pd.date_range(
        start='20191004T0645Z', end='20191004T0730Z',
        freq='15min')),
    ('1h', 'ending', pd.DatetimeIndex(['20191004T0700Z', '20191004T0800Z'])),
    ('1h', 'beginning', pd.DatetimeIndex(['20191004T0600Z',
                                          '20191004T0700Z'])),
    ('20min', 'ending', pd.DatetimeIndex([
        '20191004T0700Z', '20191004T0720Z', '20191004T0740Z'])),
    ('20min', 'beginning', pd.DatetimeIndex([
        '20191004T0640Z', '20191004T0700Z', '20191004T0720Z'])),
    ('36min', 'ending', pd.DatetimeIndex(['20191004T0712Z',
                                          '20191004T0748Z'])),
    ('36min', 'beginning', pd.DatetimeIndex(['20191004T0636Z',
                                             '20191004T0712Z'])),
])
def test__make_aggregate_index_offset_left(length, label, expected):
    test_data = {
        0: pd.DataFrame(range(6), index=pd.date_range(
            '20191004T0658Z', freq='7min', periods=6))  # end 32
        }
    out = utils._make_aggregate_index(test_data, length, label, 'UTC')
    pdt.assert_index_equal(out, expected)


def test__make_aggregate_index_tz():
    length = '30min'
    label = 'beginning'
    test_data = {
        0: pd.DataFrame(range(6), index=pd.date_range(
            '20190101T1600Z', freq='5min', periods=6))  # end 30
        }
    expected = pd.DatetimeIndex(['20190101T0900'],
                                tz='America/Denver')
    out = utils._make_aggregate_index(test_data, length, label,
                                      'America/Denver')
    pdt.assert_index_equal(out, expected)


def test__make_aggregate_index_invalid_length():
    length = '33min'
    label = 'beginning'
    test_data = {
        0: pd.DataFrame(range(6), index=pd.date_range(
            '20190101T0158Z', freq='7min', periods=6))  # end 32
        }
    with pytest.raises(ValueError):
        utils._make_aggregate_index(test_data, length, label, 'UTC')


def test__make_aggregate_index_instant():
    length = '30min'
    label = 'instant'
    test_data = {
        0: pd.DataFrame(range(6), index=pd.date_range(
            '20190101T0100Z', freq='10min', periods=6))  # end 32
        }
    with pytest.raises(ValueError):
        utils._make_aggregate_index(test_data, length, label, 'UTC')


@pytest.mark.parametrize('start,end', [
    (pd.Timestamp('20190101T0000Z'), pd.Timestamp('20190102T0000')),
    (pd.Timestamp('20190101T0000'), pd.Timestamp('20190102T0000Z')),
    (pd.Timestamp('20190101T0000'), pd.Timestamp('20190102T0000')),
])
def test__make_aggregate_index_localization(start, end):
    length = '30min'
    label = 'ending'
    test_data = {
        0: pd.DataFrame(range(1), index=pd.DatetimeIndex([start])),
        1: pd.DataFrame(range(1), index=pd.DatetimeIndex([end])),
        }
    with pytest.raises(TypeError):
        utils._make_aggregate_index(test_data, length, label, 'UTC')


@pytest.mark.parametrize('inp,oup', [
    (pd.DataFrame(dtype=float), pd.Series(dtype=float)),
    (pd.DataFrame(index=pd.DatetimeIndex([]), dtype=float),
     pd.DataFrame(dtype=float)),
    (pd.Series([0, 1]), pd.Series([0, 1])),
    (pd.DataFrame([[0, 1], [1, 2]]), pd.DataFrame([[0, 1], [1, 2]])),
    pytest.param(
        pd.Series([0, 1]),
        pd.Series([0, 1], index=pd.date_range(start='now', freq='1min',
                                              periods=2)),
        marks=pytest.mark.xfail(type=AssertionError, strict=True)),
    pytest.param(
        pd.Series([0, 1]),
        pd.Series([1, 0]),
        marks=pytest.mark.xfail(type=AssertionError, strict=True))
])
def test_sha256_pandas_object_hash(inp, oup):
    assert utils.sha256_pandas_object_hash(inp) == utils.sha256_pandas_object_hash(oup)  # NOQA


def test_listhandler():
    logger = logging.getLogger('testlisthandler')
    handler = utils.ListHandler()
    logger.addHandler(handler)
    logger.setLevel('DEBUG')
    logger.warning('Test it')
    logger.debug('What?')
    out = handler.export_records()
    assert len(out) == 1
    assert out[0].message == 'Test it'
    assert len(handler.export_records(logging.DEBUG)) == 2


def test_listhandler_recreate():
    logger = logging.getLogger('testlisthandler')
    handler = utils.ListHandler()
    logger.addHandler(handler)
    logger.setLevel('DEBUG')
    logger.warning('Test it')
    logger.debug('What?')
    out = handler.export_records()
    assert len(out) == 1
    assert out[0].message == 'Test it'
    assert len(handler.export_records(logging.DEBUG)) == 2

    l2 = logging.getLogger('testlist2')
    h2 = utils.ListHandler()
    l2.addHandler(h2)
    l2.error('Second fail')
    out = h2.export_records()
    assert len(out) == 1
    assert out[0].message == 'Second fail'


def test_hijack_loggers(mocker):
    old_handler = mocker.MagicMock()
    new_handler = mocker.MagicMock()
    mocker.patch('solarforecastarbiter.utils.ListHandler',
                 return_value=new_handler)
    logger = logging.getLogger('testhijack')
    logger.addHandler(old_handler)
    assert logger.handlers[0] == old_handler
    with utils.hijack_loggers(['testhijack']):
        assert logger.handlers[0] == new_handler
    assert logger.handlers[0] == old_handler


@pytest.mark.parametrize('data,freq,expected', [
    (pd.Series(index=pd.DatetimeIndex([]), dtype=float), '5min',
     [pd.Series(index=pd.DatetimeIndex([]), dtype=float)]),
    (pd.Series([1.0], index=pd.DatetimeIndex(['2020-01-01T00:00Z'])),
     '5min',
     [pd.Series([1.0], index=pd.DatetimeIndex(['2020-01-01T00:00Z']))]),
    (pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range('2020-01-01T00:00Z', freq='1h', periods=3)),
     '1h',
     [pd.Series(
         [1.0, 2.0, 3.0],
         index=pd.date_range('2020-01-01T00:00Z', freq='1h', periods=3))]),
    (pd.Series(
        [1.0, 2.0, 4.0],
        index=pd.DatetimeIndex(['2020-01-01T01:00Z', '2020-01-01T02:00Z',
                                '2020-01-01T04:00Z'])),
     '1h',
     [pd.Series(
         [1.0, 2.0],
         index=pd.DatetimeIndex(['2020-01-01T01:00Z', '2020-01-01T02:00Z'])),
      pd.Series(
          [4.0],
          index=pd.DatetimeIndex(['2020-01-01T04:00Z'])),
      ]),
    (pd.Series(
        [1.0, 3.0, 5.0],
        index=pd.DatetimeIndex(['2020-01-01T01:00Z', '2020-01-01T03:00Z',
                                '2020-01-01T05:00Z'])),
     '1h',
     [pd.Series(
         [1.0],
         index=pd.DatetimeIndex(['2020-01-01T01:00Z'])),
      pd.Series(
         [3.0],
         index=pd.DatetimeIndex(['2020-01-01T03:00Z'])),
      pd.Series(
          [5.0],
          index=pd.DatetimeIndex(['2020-01-01T05:00Z'])),
      ]),
    (pd.DataFrame(index=pd.DatetimeIndex([]), dtype=float), '1h',
     [pd.DataFrame(index=pd.DatetimeIndex([]), dtype=float)]),
    (pd.DataFrame(
        {'a': [1.0, 2.0, 4.0], 'b': [11.0, 12.0, 14.0]},
        index=pd.DatetimeIndex(['2020-01-01T01:00Z', '2020-01-01T02:00Z',
                                '2020-01-01T04:00Z'])),
     '1h',
     [pd.DataFrame(
         {'a': [1.0, 2.0], 'b': [11.0, 12.0]},
         index=pd.DatetimeIndex(['2020-01-01T01:00Z', '2020-01-01T02:00Z'])),
      pd.DataFrame(
          {'a': [4.0], 'b': [14.0]},
          index=pd.DatetimeIndex(['2020-01-01T04:00Z'])),
      ]),
    (pd.DataFrame(
        {'_cid': [1.0, 2.0, 4.0], '_cid0': [11.0, 12.0, 14.0]},
        index=pd.DatetimeIndex(['2020-01-01T01:00Z', '2020-01-01T02:00Z',
                                '2020-01-01T04:00Z'])),
     '1h',
     [pd.DataFrame(
         {'_cid': [1.0, 2.0], '_cid0': [11.0, 12.0]},
         index=pd.DatetimeIndex(['2020-01-01T01:00Z', '2020-01-01T02:00Z'])),
      pd.DataFrame(
          {'_cid': [4.0], '_cid0': [14.0]},
          index=pd.DatetimeIndex(['2020-01-01T04:00Z'])),
      ]),
    (pd.DataFrame(
        [[0.0, 1.0], [2.0, 3.0]],
        columns=pd.MultiIndex.from_product([[0], ['a', 'b']]),
        index=pd.DatetimeIndex(['2020-01-01T00:00Z', '2020-01-02T00:00Z'])),
     '12h',
     [pd.DataFrame(
         [[0.0, 1.0]],
         columns=pd.MultiIndex.from_product([[0], ['a', 'b']]),
         index=pd.DatetimeIndex(['2020-01-01T00:00Z'])),
      pd.DataFrame(
          [[2.0, 3.0]],
          columns=pd.MultiIndex.from_product([[0], ['a', 'b']]),
          index=pd.DatetimeIndex(['2020-01-02T00:00Z']))]),
])
def test_generate_continuous_chunks(data, freq, expected):
    lg = list(utils.generate_continuous_chunks(data, freq))
    assert len(lg) == len(expected)
    for i, g in enumerate(lg):
        if isinstance(expected[i], pd.DataFrame):
            pdt.assert_frame_equal(expected[i], g, check_column_type=False)
        else:
            pdt.assert_series_equal(expected[i], g)


@pytest.mark.parametrize('data,freq,err', [
    (pd.Series(dtype=float), '5min', TypeError),
    (pd.DataFrame(dtype=float), '5min', TypeError),
    ([], '5min', TypeError),
    ([], 'no', TypeError),
    (pd.Series(index=pd.DatetimeIndex([]), dtype=float), 'no', ValueError),
    pytest.param(pd.Series(index=pd.DatetimeIndex([]), dtype=float),
                 '5min', TypeError,
                 marks=pytest.mark.xfail(strict=True))
])
def test_generate_continuous_chunks_errs(data, freq, err):
    with pytest.raises(err):
        list(utils.generate_continuous_chunks(data, freq))
