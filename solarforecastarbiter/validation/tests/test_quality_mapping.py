from itertools import product


import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import pytest


from solarforecastarbiter.validation import quality_mapping


def test_ok_user_flagged():
    assert quality_mapping.DESCRIPTION_MASK_MAPPING['OK'] == 0
    assert quality_mapping.DESCRIPTION_MASK_MAPPING['USER FLAGGED'] == 1


def test_description_dict_version_compatibility():
    for dict_ in quality_mapping.BITMASK_DESCRIPTION_DICT.values():
        assert dict_['VERSION IDENTIFIER 0'] == 1 << 1
        assert dict_['VERSION IDENTIFIER 1'] == 1 << 2
        assert dict_['VERSION IDENTIFIER 2'] == 1 << 3


def test_latest_version_flag():
    # test valid while only identifiers 0 - 2 present
    last_identifier = max(
        int(vi.split(' ')[-1]) for vi in
        quality_mapping.DESCRIPTION_MASK_MAPPING.keys() if
        vi.startswith('VERSION IDENTIFIER'))
    assert last_identifier == 2
    assert (quality_mapping.LATEST_VERSION_FLAG ==
            quality_mapping.LATEST_VERSION << 1)


@pytest.mark.parametrize(
    'flag_val', quality_mapping.DESCRIPTION_MASK_MAPPING.items())
def test_convert_bool_flags_to_flag_mask(flag_val):
    flag, mask = flag_val
    mask |= quality_mapping.LATEST_VERSION_FLAG
    ser = pd.Series([0, 0, 1, 0, 1])
    flags = quality_mapping.convert_bool_flags_to_flag_mask(ser, flag, True)
    assert_series_equal(flags, pd.Series([
        mask, mask, quality_mapping.LATEST_VERSION_FLAG, mask,
        quality_mapping.LATEST_VERSION_FLAG]))


@pytest.mark.parametrize('flag_invert', product(
    quality_mapping.DESCRIPTION_MASK_MAPPING.keys(), [True, False]))
def test_convert_bool_flags_to_flag_mask_none(flag_invert):
    assert quality_mapping.convert_bool_flags_to_flag_mask(
        None, *flag_invert) is None


@pytest.mark.parametrize('flag_invert', product(
    quality_mapping.DESCRIPTION_MASK_MAPPING.keys(), [True, False]))
def test_convert_bool_flags_to_flag_mask_adds_latest_version(flag_invert):
    ser = pd.Series([0, 0, 0, 1, 1])
    flags = quality_mapping.convert_bool_flags_to_flag_mask(
        ser, *flag_invert)
    assert (flags & quality_mapping.LATEST_VERSION_FLAG).all()


@pytest.fixture()
def ignore_latest_version(mocker):
    mocker.patch(
        'solarforecastarbiter.validation.quality_mapping.LATEST_VERSION_FLAG',
        0)


@pytest.mark.parametrize(
    'flag_val', quality_mapping.DESCRIPTION_MASK_MAPPING.items())
def test_convert_bool_flags_to_flag_mask_invert(flag_val,
                                                ignore_latest_version):
    flag, mask = flag_val
    ser = pd.Series([0, 0, 1, 0, 1])
    flags = quality_mapping.convert_bool_flags_to_flag_mask(ser, flag, True)
    assert_series_equal(flags, pd.Series([mask, mask, 0, mask, 0]))


@pytest.mark.parametrize(
    'flag_val', quality_mapping.DESCRIPTION_MASK_MAPPING.items())
def test_convert_bool_flags_to_flag_mask_no_invert(flag_val,
                                                   ignore_latest_version):
    flag, mask = flag_val
    ser = pd.Series([0, 0, 1, 0, 1])
    flags = quality_mapping.convert_bool_flags_to_flag_mask(ser, flag, False)
    assert_series_equal(flags, pd.Series([0, 0, mask, 0, mask]))


@pytest.mark.parametrize(
    'flag_val', quality_mapping.DESCRIPTION_MASK_MAPPING.items())
def test_mask_flags(flag_val):
    flag, mask = flag_val
    latest = quality_mapping.LATEST_VERSION_FLAG
    mask |= latest

    @quality_mapping.mask_flags(flag)
    def f():
        return pd.Series([True, True, False, False])

    out = f(_return_mask=True)
    assert_series_equal(out, pd.Series([latest, latest, mask, mask]))


@pytest.mark.parametrize(
    'flag_val', quality_mapping.DESCRIPTION_MASK_MAPPING.items())
def test_mask_flags_tuple(flag_val):
    flag, mask = flag_val
    latest = quality_mapping.LATEST_VERSION_FLAG
    mask |= latest

    @quality_mapping.mask_flags(flag)
    def f():
        return pd.Series([True, True, False, False]), None

    out = f(_return_mask=True)
    assert_series_equal(out[0], pd.Series([latest, latest, mask, mask]))
    assert out[1] is None


@pytest.mark.parametrize(
    'flag_val', quality_mapping.DESCRIPTION_MASK_MAPPING.items())
def test_mask_flags_noop(flag_val):
    flag, mask = flag_val
    latest = quality_mapping.LATEST_VERSION_FLAG
    mask |= latest

    inp = pd.Series([True, True, False, False])

    @quality_mapping.mask_flags(flag)
    def f():
        return inp

    out = f()
    assert_series_equal(out, inp)


@pytest.mark.parametrize('flag,expected', [
    (0b10, 1),
    (0b11, 1),
    (0b10010, 1),
    (0b10010010, 1),
    (0b100, 2),
    (0b110, 3),
    (0b1110001011111, 7)
])
def test_get_version(flag, expected):
    assert quality_mapping.get_version(flag) == expected


def test_has_data_been_validated():
    flags = pd.Series([0, 1, 2, 7])
    out = quality_mapping.has_data_been_validated(flags)
    assert_series_equal(out, pd.Series([False, False, True, True]))


@pytest.mark.parametrize('flag,desc,result', [
    (0, 'OK', True),
    (1, 'OK', False),
    (2, 'OK', True),
    (3, 'OK', False),
    (0, 'USER FLAGGED', False),
    (3, 'USER FLAGGED', True),
    (0, 'CLEARSKY', False),
    (16, 'OK', False),
    (1, 'USER FLAGGED', True),
    (16, 'NIGHTTIME', True),
    (33, 'CLEARSKY', True),
    (33, 'NIGHTTIME', False),
    (33, ['OK', 'NIGHTTIME'], False),
    (33, ('OK', 'CLEARSKY', 'USER FLAGGED'), True),
    (2, ('OK', 'NIGHTTIME'), True),
    (9297, 'USER FLAGGED', True)
])
def test_check_if_single_value_flagged(flag, desc, result):
    flag |= quality_mapping.LATEST_VERSION_FLAG
    out = quality_mapping.check_if_single_value_flagged(flag, desc)
    assert out == result


@pytest.mark.parametrize('flag', [0, 1])
def test_check_if_single_value_flagged_validation_error(flag):
    with pytest.raises(ValueError):
        quality_mapping.check_if_single_value_flagged(flag, 'OK')


@pytest.mark.parametrize('desc', [33, b'OK', [1, 2], []])
def test_check_if_single_value_flagged_type_error(desc):
    with pytest.raises(TypeError):
        quality_mapping.check_if_single_value_flagged(2, desc)


@pytest.mark.parametrize('desc', ['NOPE', 'MAYBE', ['YES', 'NO']])
def test_check_if_single_value_flagged_key_error(desc):
    with pytest.raises(KeyError):
        quality_mapping.check_if_single_value_flagged(2, desc)


@pytest.mark.parametrize('flags,expected', [
    (pd.Series([0, 1, 0]), pd.Series([False, False, False])),
    (pd.Series([2, 2, 2]), pd.Series([True, True, True])),
    (pd.Series([3, 2, 2]), pd.Series([False, True, True])),
    (pd.Series([3, 34, 130]), pd.Series([False, False, False]))
])
def test_which_data_is_ok(flags, expected):
    out = quality_mapping.which_data_is_ok(flags)
    assert_series_equal(out, expected)


DESCRIPTIONS = ['USER FLAGGED', 'NIGHTTIME', 'CLEARSKY',
                'SHADED', 'UNEVEN FREQUENCY', 'LIMITS EXCEEDED',
                'CLEARSKY EXCEEDED', 'STALE VALUES', 'INTERPOLATED VALUES',
                'CLIPPED VALUES', 'INCONSISTENT IRRADIANCE COMPONENTS',
                'DAILY VALIDATION APPLIED']


DERIVED_DESCRIPTIONS = ['DAYTIME', 'DAYTIME STALE VALUES',
                        'DAYTIME INTERPOLATED VALUES']


@pytest.mark.parametrize('flag,expected', [
    (2, pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  index=DESCRIPTIONS, dtype=bool)),
    (3, pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  index=DESCRIPTIONS, dtype=bool)),
    (35, pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   index=DESCRIPTIONS, dtype=bool)),
    (2 | 1 << 13 | 1 << 12 | 1 << 10,
     pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
               index=DESCRIPTIONS, dtype=bool))

])
def test_check_for_all_descriptions(flag, expected):
    out = quality_mapping.check_for_all_descriptions(flag)
    assert_series_equal(out, expected)


@pytest.mark.parametrize('flag', [0, 1])
def test_check_for_all_validation_fail(flag):
    with pytest.raises(ValueError):
        quality_mapping.check_for_all_descriptions(flag)


def test_convert_mask_into_dataframe():
    flags = (pd.Series([0, 0, 1, 1 << 12, 1 << 9 | 1 << 7 | 1 << 5]) |
             quality_mapping.LATEST_VERSION_FLAG)
    columns = DESCRIPTIONS + ['NOT VALIDATED'] + DERIVED_DESCRIPTIONS
    expected = pd.DataFrame([[0] * 13 + [1, 0, 0],
                             [0] * 13 + [1, 0, 0],
                             [1] + [0] * 12 + [1, 0, 0],
                             [0] * 9 + [1, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
                            columns=columns,
                            dtype=bool)
    out = quality_mapping.convert_mask_into_dataframe(flags)
    assert_frame_equal(out, expected)


def test_convert_mask_into_dataframe_w_unvalidated():
    flags = (pd.Series([0, 0, 1, 1 << 12, 1 << 9 | 1 << 7 | 1 << 5]) |
             quality_mapping.LATEST_VERSION_FLAG)
    flags.iloc[0] = 0
    columns = DESCRIPTIONS + ['NOT VALIDATED'] + DERIVED_DESCRIPTIONS
    expected = pd.DataFrame([[0] * 12 + [1, 0, 0, 0],
                             [0] * 13 + [1, 0, 0],
                             [1] + [0] * 12 + [1, 0, 0],
                             [0] * 9 + [1, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
                            columns=columns,
                            dtype=bool)
    out = quality_mapping.convert_mask_into_dataframe(flags)
    assert_frame_equal(out, expected, check_like=True)


def test_convert_mask_into_dataframe_all_unvalidated():
    flags = pd.Series([0, 0, 1, 1, 0])
    columns = ['NOT VALIDATED']
    expected = pd.DataFrame([[1]] * 5,
                            columns=columns,
                            dtype=bool)
    out = quality_mapping.convert_mask_into_dataframe(flags)
    assert_frame_equal(out, expected, check_like=True)


def test_convert_flag_frame_to_strings():
    frame = pd.DataFrame({'FIRST': [True, False, False],
                          'SECOND': [False, False, True],
                          'THIRD': [True, False, True]})
    expected = pd.Series(['FIRST, THIRD', 'OK', 'SECOND, THIRD'])
    out = quality_mapping.convert_flag_frame_to_strings(frame)
    assert_series_equal(expected, out)


@pytest.mark.parametrize('expected,desc', [
    (pd.Series([1, 0, 0, 0], dtype=bool), 'OK'),
    (pd.Series([0, 1, 0, 1], dtype=bool), 'USER FLAGGED'),
    (pd.Series([0, 0, 1, 0], dtype=bool), 'CLEARSKY EXCEEDED'),
    (pd.Series([0, 0, 0, 1], dtype=bool), 'CLEARSKY'),
    (pd.Series([0, 0, 0, 1], dtype=bool), 'CLIPPED VALUES'),
    (pd.Series([0, 0, 0, 0], dtype=bool), 'STALE VALUES'),
])
def test_check_if_series_flagged(expected, desc):
    flags = pd.Series([2, 3, 2 | 1 << 9, 2 | 1 << 5 | 1 << 12 | 1])
    out = quality_mapping.check_if_series_flagged(flags, desc)
    assert_series_equal(out, expected)


def test_check_if_series_flagged_validated_fail():
    with pytest.raises(ValueError):
        quality_mapping.check_if_series_flagged(pd.Series([0, 1, 0]), 'OK')


def test_check_if_series_flagged_type_fail():
    with pytest.raises(TypeError):
        quality_mapping.check_if_series_flagged(pd.Series([2, 3, 35]),
                                                ['OK', b'CLEARSKY', []])


def test_check_if_series_flagged_key_fail():
    with pytest.raises(KeyError):
        quality_mapping.check_if_series_flagged(pd.Series([2, 3, 35]),
                                                ['NOK'])
