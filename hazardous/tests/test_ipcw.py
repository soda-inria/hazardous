import numpy as np
import pytest
from numpy.testing import assert_allclose

from .._ipcw import IPCWEstimator


@pytest.mark.parametrize("seed", range(10))
def test_ipcw_invariant_properties(seed):
    rng = np.random.default_rng(seed)
    n_samples = rng.choice(np.logspace(1, 3, 100).astype(np.int32))
    t_max = rng.choice(np.logspace(1, 3, 100))
    n_events = rng.choice([1, 2, 5, 10])
    y_competing = dict(
        event=rng.choice(range(n_events + 1), size=n_samples),
        duration=rng.uniform(1, t_max, size=n_samples),
    )
    y_any_event = y_competing.copy()
    y_any_event["event"][y_competing["event"] >= 1] = 1

    test_times = np.linspace(0, t_max, num=100)

    est_competing = IPCWEstimator().fit(y_competing)
    ipcw_values = est_competing.compute_ipcw_at(test_times)

    est_any_event = IPCWEstimator().fit(y_any_event)
    ipcw_values_any_event = est_any_event.compute_ipcw_at(test_times)

    # The IPCW should be the same for any event and competing risk: they
    # are defined by the censoring distribution only.
    assert_allclose(ipcw_values, ipcw_values_any_event)

    # The IPCW should be greater than 1 as they are inverse probabilities.
    assert np.all(ipcw_values >= 1.0), ipcw_values

    # The IPCW values should be finite.
    assert np.all(np.isfinite(ipcw_values)), ipcw_values

    # The IPCW at time 0 should be 1: there cannot be any censoring there.
    assert_allclose(ipcw_values[0], 1.0)

    # The IPCW should be monotonically increasing as the censoring survival
    # probability is monotonically decreasing.
    assert np.all(np.diff(ipcw_values) >= 0.0), ipcw_values

    # The IPCW values should be strictly larger than 1 for all times larger
    # than the first censored observation.
    first_censored = y_competing["duration"][y_competing["event"] == 0].min()
    after_first_censored_mask = test_times > first_censored
    ipcw_after_first_censored = ipcw_values[after_first_censored_mask]
    assert np.all(ipcw_after_first_censored > 1.0), ipcw_after_first_censored

    # The IPCW are extrapolated with a constant value equal to the last IPCW
    # value beyond the last observed time.
    #
    # XXX: not sure if this is the best behavior when using IPCW to debias a
    # performance metric evaluated on a test dataset that has data points
    # beyond the maximum time observed on the training dataset.
    extrapolated_ipcw = est_competing.compute_ipcw_at(
        np.linspace(1.1 * t_max, 10 * t_max, num=5)
    )
    assert_allclose(
        extrapolated_ipcw, np.full_like(extrapolated_ipcw, fill_value=ipcw_values[-1])
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ipcw_deterministic_censoring_weights(seed):
    rng = np.random.default_rng(seed)
    n_samples = 1000
    t_max = 10000
    threshold = t_max // 3
    durations = rng.integers(1, t_max, size=n_samples)
    events = np.ones(n_samples, dtype=np.int32)
    events[durations > threshold] = 0
    y = dict(
        event=events,
        duration=durations,
    )
    est = IPCWEstimator().fit(y)
    before_censoring = np.arange(threshold)
    ipcw_before_censoring = est.compute_ipcw_at(before_censoring)
    assert_allclose(ipcw_before_censoring, np.ones_like(ipcw_before_censoring))

    after_censoring = np.linspace(threshold, t_max, num=100)
    ipcw_after_censoring = est.compute_ipcw_at(after_censoring)
    assert np.all(np.isfinite(ipcw_after_censoring)), ipcw_after_censoring
    assert np.all(ipcw_after_censoring[1:] > 1.0), ipcw_after_censoring

    # Beyond the last observed time, the IPCW should be extrapolated with a
    # constant value equal to the last IPCW value.
    extrapolated_ipcw = est.compute_ipcw_at(np.linspace(t_max, 10 * t_max, num=5))
    assert_allclose(
        extrapolated_ipcw,
        np.full_like(extrapolated_ipcw, fill_value=ipcw_after_censoring[-1]),
    )


@pytest.mark.parametrize("competing_risk", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ipcw_no_censoring(competing_risk, seed):
    # IPCW should be 1 when there is no censoring
    rng = np.random.default_rng(seed)
    n_samples = 100
    t_max = 1000
    durations = rng.integers(1, t_max, size=n_samples)
    if competing_risk:
        y_uncensored = dict(
            event=rng.choice([1, 2], size=n_samples),
            duration=durations,
        )
    else:
        y_uncensored = dict(
            event=np.ones(n_samples, dtype=np.int32),
            duration=durations,
        )

    est = IPCWEstimator().fit(y_uncensored)
    test_times = np.arange(t_max)
    ipcw_values = est.compute_ipcw_at(test_times)
    assert_allclose(ipcw_values, np.ones_like(ipcw_values))


@pytest.mark.parametrize("competing_risk", [True, False])
def test_ipcw_consistent_with_sksurv(competing_risk):
    # Compare IPCW values with scikit-survival on hardcoded random
    # data with censoring: we do not want to depend on scikit-survival
    # for the tests, but we want to make sure that we get the same
    # results.
    rng = np.random.default_rng(0)
    n_samples = 100
    t_max = 1000
    events = rng.choice([0, 1, 2, 3], size=n_samples)
    if not competing_risk:
        # Collapse any competing risk into a single event so that
        # the censored samples are the same in both cases.
        events[events >= 1] = 1

    y = dict(
        event=events,
        duration=rng.uniform(1, t_max, size=n_samples),
    )

    max_observed_duration = y["duration"][y["event"] != 0].max()
    test_times = np.linspace(0, max_observed_duration, num=5)

    est = IPCWEstimator().fit(y)
    ipcw_values = est.compute_ipcw_at(test_times)

    # Expected values computed with scikit-survival:
    #
    # from sksurv.nonparametric import CensoringDistributionEstimator
    # from hazardous.utils import _dict_to_recarray
    # from pprint import pprint

    # cens = CensoringDistributionEstimator()
    # y_bool_event = {
    #     "duration": y["duration"],
    #     "event": (y["event"] != 0).astype(bool),
    # }
    # cens.fit(_dict_to_recarray(y_bool_event))
    # y_test = _dict_to_recarray(
    #     {
    #         "duration": test_times,
    #         "event": np.ones_like(test_times).astype(bool),
    #     }
    # )
    # pprint(cens.predict_ipcw(y_test).tolist())

    expected_ipcw_values = np.array(
        [
            1.0,
            1.0579567311513456,
            1.0930129482665436,
            1.1861020893150818,
            2.1586149135044597,
        ]
    )
    assert_allclose(ipcw_values, expected_ipcw_values)
