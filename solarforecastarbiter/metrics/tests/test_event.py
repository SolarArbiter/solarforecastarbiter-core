import pytest
from solarforecastarbiter.metrics import event


@pytest.mark.parametrize("obs,fx,result", [
    # one event
    ([True], [False], (0, 0, 0, 1)),
    ([False], [True], (0, 1, 0, 0)),
    ([True], [True], (1, 0, 0, 0)),
    ([False], [False], (0, 0, 1, 0)),

    # two events
    ([True, False], [True, False], (1, 0, 1, 0)),
    ([True, True], [True, False], (1, 0, 0, 1)),
    ([False, False], [True, False], (0, 1, 1, 0)),
    ([True, True], [True, True], (2, 0, 0, 0)),
    ([False, False], [False, False], (0, 0, 2, 0)),
    ([True, True], [False, False], (0, 0, 0, 2)),

    # bad data
    pytest.param([True], [], 0,
                 marks=pytest.mark.xfail(raises=RuntimeError, strict=True)),
    pytest.param([], [True], 0,
                 marks=pytest.mark.xfail(raises=RuntimeError, strict=True)),
    pytest.param([True], [True, True], 0,
                 marks=pytest.mark.xfail(raises=RuntimeError, strict=True))
])
def test__event2count(obs, fx, result):
    tp, fp, tn, fn = event._event2count(obs, fx)
    assert (tp, fp, tn, fn) == result
    assert (tp + fp + tn + fn) == len(obs)


@pytest.mark.parametrize("obs,fx,result", [
    ([True], [False], 0.0),
    ([False], [True], 0.0),
    ([True, True], [True, True], 1.0),
    ([True, True], [True, False], 0.5),
])
def test_probability_of_detection(obs, fx, result):
    assert event.probability_of_detection(obs, fx) == result


@pytest.mark.parametrize("obs,fx,result", [
    ([True], [False], 0.0),
    ([False], [True], 1.0),
    ([True, True], [True, False], 0.0),
    ([False, True], [True, True], 0.5),
])
def test_false_alarm_ratio(obs, fx, result):
    assert event.false_alarm_ratio(obs, fx) == result


@pytest.mark.parametrize("obs,fx,result", [
    ([True], [False], 0.0),
    ([False], [True], 1.0),
    ([False, False], [True, False], 0.5),
])
def test_probability_of_false_detection(obs, fx, result):
    assert event.probability_of_false_detection(obs, fx) == result


@pytest.mark.parametrize("obs,fx,result", [
    ([True], [False], 0.0),
    ([True], [True], 1.0),
    ([False], [False], 0.0),
    ([True, False], [True, True], 0.5),
    ([True, False, True], [True, True, False], 1 / 3),
])
def test_critical_success_index(obs, fx, result):
    assert event.critical_success_index(obs, fx) == result


@pytest.mark.parametrize("obs,fx,result", [
    ([True], [False], 0.0),
    ([False], [False], 0.0),
    ([True, True], [True, False], 0.5),
    ([True, False], [True, True], 2.0),
])
def test_event_bias(obs, fx, result):
    assert event.event_bias(obs, fx) == result


@pytest.mark.parametrize("obs,fx,result", [
    ([True], [False], 0.0),
    ([False], [True], 0.0),
    ([True, True], [True, True], 1.0),
    ([True, False], [True, False], 1.0),
    ([True, False], [True, True], 0.5),
])
def test_event_accuracy(obs, fx, result):
    assert event.event_accuracy(obs, fx) == result
