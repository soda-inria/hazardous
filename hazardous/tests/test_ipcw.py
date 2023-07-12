import numpy as np
import pytest
from numpy.testing import assert_allclose

from .._ipcw import IPCWEstimator


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
