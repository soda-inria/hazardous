"""Tests for AJCalibration and aj_calibration."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hazardous._km_sampler import _AalenJohansenSampler, _KaplanMeierSampler
from hazardous.metrics._aj_calibration import AJCalibration, aj_calibration


def _make_competing_risks_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    duration = rng.exponential(scale=10.0, size=n)
    event = rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3], size=n)
    return {"event": event, "duration": duration}


def _perfect_inc_predictions(y, times):
    """Build predictions that exactly match the AJ/KM marginal estimates."""
    n = len(y["event"])
    aj = _AalenJohansenSampler().fit(y)
    km = _KaplanMeierSampler().fit(y)

    surv = np.tile(km.survival_func_(times), (n, 1))  # (n, T)
    cif1 = np.tile(aj.incidence_func_[1](times), (n, 1))
    cif2 = np.tile(aj.incidence_func_[2](times), (n, 1))

    # Shape: (n_samples, n_events+1, n_times)  — axes: sample, event_id, time
    return np.stack([surv, cif1, cif2], axis=1)


@pytest.fixture
def competing_data():
    return _make_competing_risks_data(n=500, seed=7)


@pytest.fixture
def times():
    return np.linspace(1, 20, 40)


class TestAJCalibrationClass:
    def test_perfect_calibration_returns_zero(self, competing_data, times):
        y = competing_data
        inc_pred = _perfect_inc_predictions(y, times)

        cal = AJCalibration().fit(y)
        scores = cal.score(times, inc_pred)

        for event_id, score in scores.items():
            assert score == pytest.approx(
                0.0, abs=1e-10
            ), f"Expected zero calibration error for event {event_id}, got {score}"

    def test_miscalibrated_model_has_nonzero_scores(self, competing_data, times):
        y = competing_data
        n = len(y["event"])
        # Constant over-prediction for all CIFs
        inc_pred = np.full((n, 3, len(times)), fill_value=0.5)

        cal = AJCalibration().fit(y)
        scores = cal.score(times, inc_pred)

        for event_id, score in scores.items():
            assert score != pytest.approx(
                0.0
            ), f"Expected non-zero calibration error for event {event_id}"

    def test_scores_keys_match_event_ids(self, competing_data, times):
        y = competing_data
        inc_pred = _perfect_inc_predictions(y, times)

        cal = AJCalibration().fit(y)
        scores = cal.score(times, inc_pred)

        assert set(scores.keys()) == {0, 1, 2}

    def test_fit_returns_self(self, competing_data):
        cal = AJCalibration()
        result = cal.fit(competing_data)
        assert result is cal

    def test_fit_sets_attributes(self, competing_data):
        cal = AJCalibration().fit(competing_data)
        assert hasattr(cal, "aalen_johansen_sampler_")
        assert hasattr(cal, "km_calibration_")
        assert hasattr(cal, "event_ids_")

    def test_event_ids_include_zero(self, competing_data):
        cal = AJCalibration().fit(competing_data)
        assert 0 in cal.event_ids_

    def test_alpha_changes_scores(self, competing_data, times):
        y = competing_data
        n = len(y["event"])
        inc_pred = np.full((n, 3, len(times)), fill_value=0.4)

        scores2 = AJCalibration(alpha=2).fit(y).score(times, inc_pred)
        scores4 = AJCalibration(alpha=4).fit(y).score(times, inc_pred)

        for event_id in [1, 2]:
            assert scores2[event_id] != pytest.approx(scores4[event_id])

    def test_difference_at_t_perfect_calibration(self, competing_data, times):
        y = competing_data
        inc_pred = _perfect_inc_predictions(y, times)

        cal = AJCalibration().fit(y)
        diffs = cal.difference_at_t(times, inc_pred)

        for event_id, diff in diffs.items():
            assert_allclose(diff, 0.0, atol=1e-10)

    def test_difference_at_t_shape(self, competing_data, times):
        y = competing_data
        inc_pred = _perfect_inc_predictions(y, times)

        cal = AJCalibration().fit(y)
        diffs = cal.difference_at_t(times, inc_pred)

        for event_id, diff in diffs.items():
            assert diff.shape == times.shape

    def test_difference_at_t_keys(self, competing_data, times):
        y = competing_data
        inc_pred = _perfect_inc_predictions(y, times)

        cal = AJCalibration().fit(y)
        diffs = cal.difference_at_t(times, inc_pred)

        assert set(diffs.keys()) == {0, 1, 2}

    def test_score_invariant_to_time_ordering(self, competing_data):
        y = competing_data
        times_sorted = np.linspace(1, 20, 30)
        rng = np.random.default_rng(11)
        perm = rng.permutation(len(times_sorted))
        times_shuffled = times_sorted[perm]

        n = len(y["event"])
        inc_pred = rng.random((n, 3, len(times_sorted)))
        inc_pred_shuffled = inc_pred[:, :, perm]

        cal = AJCalibration().fit(y)
        scores_sorted = cal.score(times_sorted, inc_pred)
        scores_shuffled = cal.score(times_shuffled, inc_pred_shuffled)

        for event_id in scores_sorted:
            assert scores_sorted[event_id] == pytest.approx(
                scores_shuffled[event_id], rel=1e-6
            )


class TestAJCalibrationFunction:
    def test_perfect_calibration_returns_zero(self, competing_data, times):
        y = competing_data
        inc_pred = _perfect_inc_predictions(y, times)

        scores = aj_calibration(y, times, inc_pred)

        for event_id, score in scores.items():
            assert score == pytest.approx(
                0.0, abs=1e-10
            ), f"Event {event_id}: expected 0, got {score}"

    def test_return_diff_at_t(self, competing_data, times):
        y = competing_data
        inc_pred = _perfect_inc_predictions(y, times)

        scores, diffs = aj_calibration(y, times, inc_pred, return_diff_at_t=True)

        assert isinstance(scores, dict)
        assert isinstance(diffs, dict)
        assert set(scores.keys()) == set(diffs.keys())
        for event_id, diff in diffs.items():
            assert diff.shape == times.shape

    def test_function_matches_class_api(self, competing_data, times):
        y = competing_data
        n = len(y["event"])
        rng = np.random.default_rng(13)
        inc_pred = rng.random((n, 3, len(times)))

        scores_func = aj_calibration(y, times, inc_pred)
        scores_class = AJCalibration().fit(y).score(times, inc_pred)

        for event_id in scores_func:
            assert scores_func[event_id] == pytest.approx(
                scores_class[event_id], rel=1e-10
            )

    def test_alpha_parameter_forwarded(self, competing_data, times):
        y = competing_data
        n = len(y["event"])
        rng = np.random.default_rng(17)
        inc_pred = rng.random((n, 3, len(times)))

        scores_a2 = aj_calibration(y, times, inc_pred, alpha=2)
        scores_a4 = aj_calibration(y, times, inc_pred, alpha=4)

        for event_id in [1, 2]:
            assert scores_a2[event_id] != pytest.approx(scores_a4[event_id])


class TestAJCalibrationSingleEvent:
    """Test that AJCalibration degrades gracefully to single-event survival."""

    def test_single_event_only(self, times):
        rng = np.random.default_rng(3)
        n = 300
        duration = rng.exponential(scale=10.0, size=n)
        event = rng.binomial(1, p=0.6, size=n)
        y = {"event": event, "duration": duration}

        km = _KaplanMeierSampler().fit(y)
        surv = np.tile(km.survival_func_(times), (n, 1))
        # Shape (n, 2, T): axis 1 index 0 = survival, index 1 = CIF event 1
        cif1 = 1 - surv
        inc_pred = np.stack([surv, cif1], axis=1)

        cal = AJCalibration().fit(y)
        scores = cal.score(times, inc_pred)

        # Only event IDs 0 and 1 present
        assert 0 in scores
        assert 1 in scores
        assert 2 not in scores
