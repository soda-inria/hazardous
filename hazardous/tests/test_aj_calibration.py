"""Tests for aj_calibration_at_t, aj_calibration_per_event, aj_calibration."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hazardous._km_sampler import _AalenJohansenSampler, _KaplanMeierSampler
from hazardous.metrics._aj_calibration import (
    aj_calibration,
    aj_calibration_at_t,
    aj_calibration_per_event,
)


def _make_competing_risks_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    duration = rng.exponential(scale=10.0, size=n)
    event = rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3], size=n)
    return {"event": event, "duration": duration}


def _perfect_inc_predictions(y, times):
    """Tile AJ/KM marginal estimates to get a perfectly calibrated prediction."""
    n = len(y["event"])
    aj = _AalenJohansenSampler().fit(y)
    km = _KaplanMeierSampler().fit(y)

    surv = np.tile(km.survival_func_(times), (n, 1))
    cif1 = np.tile(aj.incidence_func_[1](times), (n, 1))
    cif2 = np.tile(aj.incidence_func_[2](times), (n, 1))

    # Shape: (n_samples, n_events+1, n_times)
    return np.stack([surv, cif1, cif2], axis=1)


@pytest.fixture
def y():
    return _make_competing_risks_data(n=500, seed=7)


@pytest.fixture
def times():
    return np.linspace(1, 20, 40)


# ---------------------------------------------------------------------------
# aj_calibration_at_t
# ---------------------------------------------------------------------------


class TestAJCalibrationAtT:
    def test_perfect_calibration_returns_zero(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        diffs = aj_calibration_at_t(y, times, inc_pred)
        for diff in diffs.values():
            assert_allclose(diff, 0.0, atol=1e-10)

    def test_miscalibrated_model_has_nonzero_diff(self, y, times):
        n = len(y["event"])
        inc_pred = np.full((n, 3, len(times)), fill_value=0.5)
        diffs = aj_calibration_at_t(y, times, inc_pred)
        for diff in diffs.values():
            assert not np.allclose(diff, 0.0)

    def test_keys_match_event_ids(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        diffs = aj_calibration_at_t(y, times, inc_pred)
        assert set(diffs.keys()) == {0, 1, 2}

    def test_output_shape(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        diffs = aj_calibration_at_t(y, times, inc_pred)
        for diff in diffs.values():
            assert diff.shape == times.shape

    def test_event_of_interest_returns_array(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        diff = aj_calibration_at_t(y, times, inc_pred, event_of_interest=1)
        assert isinstance(diff, np.ndarray)
        assert diff.shape == times.shape

    def test_invariant_to_time_ordering(self, y):
        times_sorted = np.linspace(1, 20, 30)
        rng = np.random.default_rng(11)
        perm = rng.permutation(len(times_sorted))
        times_shuffled = times_sorted[perm]

        n = len(y["event"])
        inc_pred = rng.random((n, 3, len(times_sorted)))
        inc_pred_shuffled = inc_pred[:, :, perm]

        diffs_sorted = aj_calibration_at_t(y, times_sorted, inc_pred)
        diffs_shuffled = aj_calibration_at_t(y, times_shuffled, inc_pred_shuffled)

        for event_id in diffs_sorted:
            assert_allclose(
                diffs_sorted[event_id], diffs_shuffled[event_id], atol=1e-10
            )


# ---------------------------------------------------------------------------
# aj_calibration_per_event
# ---------------------------------------------------------------------------


class TestAJCalibrationPerEvent:
    def test_perfect_calibration_returns_zero(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        scores = aj_calibration_per_event(y, times, inc_pred)
        for score in scores.values():
            assert score == pytest.approx(0.0, abs=1e-10)

    def test_miscalibrated_model_has_nonzero_scores(self, y, times):
        n = len(y["event"])
        inc_pred = np.full((n, 3, len(times)), fill_value=0.5)
        scores = aj_calibration_per_event(y, times, inc_pred)
        for score in scores.values():
            assert score != pytest.approx(0.0)

    def test_keys_match_event_ids(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        scores = aj_calibration_per_event(y, times, inc_pred)
        assert set(scores.keys()) == {0, 1, 2}

    def test_event_of_interest_returns_float(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        score = aj_calibration_per_event(y, times, inc_pred, event_of_interest=1)
        assert isinstance(score, float)

    def test_alpha_changes_scores(self, y, times):
        n = len(y["event"])
        inc_pred = np.full((n, 3, len(times)), fill_value=0.4)
        scores2 = aj_calibration_per_event(y, times, inc_pred, alpha=2)
        scores4 = aj_calibration_per_event(y, times, inc_pred, alpha=4)
        for event_id in [1, 2]:
            assert scores2[event_id] != pytest.approx(scores4[event_id])

    def test_consistent_with_at_t(self, y, times):
        """Score should equal trapz(diff**alpha) / t_max computed from at_t."""
        n = len(y["event"])
        rng = np.random.default_rng(3)
        inc_pred = rng.random((n, 3, len(times)))

        scores = aj_calibration_per_event(y, times, inc_pred, alpha=2)

        order = np.argsort(times)
        times_sorted = times[order]
        inc_sorted = inc_pred[:, :, order]
        diffs = aj_calibration_at_t(y, times_sorted, inc_sorted)

        t_max = times_sorted[-1]
        for event_id, diff in diffs.items():
            expected = np.trapezoid(diff**2, times_sorted) / t_max
            assert scores[event_id] == pytest.approx(expected, rel=1e-10)

    def test_invariant_to_time_ordering(self, y):
        times_sorted = np.linspace(1, 20, 30)
        rng = np.random.default_rng(5)
        perm = rng.permutation(len(times_sorted))
        times_shuffled = times_sorted[perm]

        n = len(y["event"])
        inc_pred = rng.random((n, 3, len(times_sorted)))
        inc_pred_shuffled = inc_pred[:, :, perm]

        scores_sorted = aj_calibration_per_event(y, times_sorted, inc_pred)
        scores_shuffled = aj_calibration_per_event(y, times_shuffled, inc_pred_shuffled)

        for event_id in scores_sorted:
            assert scores_sorted[event_id] == pytest.approx(
                scores_shuffled[event_id], rel=1e-6
            )


# ---------------------------------------------------------------------------
# aj_calibration
# ---------------------------------------------------------------------------


class TestAJCalibration:
    def test_perfect_calibration_returns_zero(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        score = aj_calibration(y, times, inc_pred)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_returns_float(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        score = aj_calibration(y, times, inc_pred)
        assert isinstance(score, float)

    def test_mean_reduction(self, y, times):
        n = len(y["event"])
        rng = np.random.default_rng(7)
        inc_pred = rng.random((n, 3, len(times)))

        per_event = aj_calibration_per_event(y, times, inc_pred)
        expected = float(np.mean(list(per_event.values())))
        score = aj_calibration(y, times, inc_pred, reduction="mean")
        assert score == pytest.approx(expected, rel=1e-10)

    def test_sum_reduction(self, y, times):
        n = len(y["event"])
        rng = np.random.default_rng(9)
        inc_pred = rng.random((n, 3, len(times)))

        per_event = aj_calibration_per_event(y, times, inc_pred)
        expected = float(np.sum(list(per_event.values())))
        score = aj_calibration(y, times, inc_pred, reduction="sum")
        assert score == pytest.approx(expected, rel=1e-10)

    def test_mean_vs_sum(self, y, times):
        n = len(y["event"])
        inc_pred = np.full((n, 3, len(times)), fill_value=0.4)
        mean_score = aj_calibration(y, times, inc_pred, reduction="mean")
        sum_score = aj_calibration(y, times, inc_pred, reduction="sum")
        assert sum_score == pytest.approx(mean_score * 3, rel=1e-10)

    def test_invalid_reduction_raises(self, y, times):
        inc_pred = _perfect_inc_predictions(y, times)
        with pytest.raises(ValueError, match="reduction must be"):
            aj_calibration(y, times, inc_pred, reduction="bad")

    def test_alpha_changes_score(self, y, times):
        n = len(y["event"])
        inc_pred = np.full((n, 3, len(times)), fill_value=0.4)
        score2 = aj_calibration(y, times, inc_pred, alpha=2)
        score4 = aj_calibration(y, times, inc_pred, alpha=4)
        assert score2 != pytest.approx(score4)


# ---------------------------------------------------------------------------
# Single-event degradation
# ---------------------------------------------------------------------------


class TestAJCalibrationSingleEvent:
    def test_single_event_only(self, times):
        rng = np.random.default_rng(3)
        n = 300
        duration = rng.exponential(scale=10.0, size=n)
        event = rng.binomial(1, p=0.6, size=n)
        y = {"event": event, "duration": duration}

        km = _KaplanMeierSampler().fit(y)
        surv = np.tile(km.survival_func_(times), (n, 1))
        cif1 = 1 - surv
        inc_pred = np.stack([surv, cif1], axis=1)

        diffs = aj_calibration_at_t(y, times, inc_pred)
        assert 0 in diffs
        assert 1 in diffs
        assert 2 not in diffs

        scores = aj_calibration_per_event(y, times, inc_pred)
        assert set(scores.keys()) == {0, 1}
