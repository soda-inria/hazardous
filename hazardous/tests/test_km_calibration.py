"""Tests for _KMCalibration and km_calibration."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from .._km_sampler import _KaplanMeierSampler
from ..metrics._km_calibration import (
    _KMCalibration,
    km_calibration,
    recalibrate_survival_function,
    recalibrate_survival_function_predictions,
)


def _make_survival_data(n=400, scale=10.0, seed=0):
    rng = np.random.default_rng(seed)
    duration = rng.exponential(scale=scale, size=n)
    event = rng.binomial(1, p=0.7, size=n)
    return {"event": event, "duration": duration}


def _km_survival_predictions(y, times, n):
    """Tile the KM survival estimate to simulate a perfectly calibrated model."""
    km = _KaplanMeierSampler().fit(y)
    return np.tile(km.survival_func_(times), (n, 1))


@pytest.fixture
def survival_data():
    return _make_survival_data(n=400, seed=42)


@pytest.fixture
def times():
    return np.linspace(1, 20, 40)


class TestKMCalibrationClass:
    def test_perfect_calibration_returns_zero(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        surv_pred = _km_survival_predictions(y, times, n)

        cal = _KMCalibration().fit(y)
        score = cal.score(times, surv_pred)

        assert score == pytest.approx(0.0, abs=1e-10)

    def test_miscalibrated_model_has_positive_score(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        # Constant over-prediction of survival
        surv_pred = np.full((n, len(times)), fill_value=0.9)

        cal = _KMCalibration().fit(y)
        score = cal.score(times, surv_pred)

        assert score > 0.0

    def test_alpha_changes_score(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        surv_pred = np.full((n, len(times)), fill_value=0.7)

        score_alpha2 = _KMCalibration(alpha=2).fit(y).score(times, surv_pred)
        score_alpha4 = _KMCalibration(alpha=4).fit(y).score(times, surv_pred)

        # Both should be non-zero and different
        assert score_alpha2 > 0.0
        assert score_alpha4 > 0.0
        assert score_alpha2 != pytest.approx(score_alpha4)

    def test_fit_returns_self(self, survival_data):
        cal = _KMCalibration()
        result = cal.fit(survival_data)
        assert result is cal

    def test_fit_sets_sampler_attribute(self, survival_data):
        cal = _KMCalibration().fit(survival_data)
        assert hasattr(cal, "kaplan_meier_sampler_")

    def test_difference_at_t_perfect_calibration(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        surv_pred = _km_survival_predictions(y, times, n)

        cal = _KMCalibration().fit(y)
        diff = cal.difference_at_t(times, surv_pred)

        assert_allclose(diff, 0.0, atol=1e-10)

    def test_difference_at_t_shape(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        surv_pred = np.random.default_rng(0).random((n, len(times)))

        cal = _KMCalibration().fit(y)
        diff = cal.difference_at_t(times, surv_pred)

        assert diff.shape == times.shape

    def test_score_invariant_to_time_ordering(self, survival_data):
        y = survival_data
        n = len(y["event"])
        times_sorted = np.linspace(1, 20, 30)
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(times_sorted))
        times_shuffled = times_sorted[perm]

        surv_pred = rng.random((n, len(times_sorted)))
        surv_pred_shuffled = surv_pred[:, perm]

        cal = _KMCalibration().fit(y)
        score_sorted = cal.score(times_sorted, surv_pred)
        score_shuffled = cal.score(times_shuffled, surv_pred_shuffled)

        assert score_sorted == pytest.approx(score_shuffled, rel=1e-6)


class TestKMCalibrationFunction:
    def test_perfect_calibration_returns_zero(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        surv_pred = _km_survival_predictions(y, times, n)

        score = km_calibration(y, times, surv_pred)

        assert score == pytest.approx(0.0, abs=1e-10)

    def test_return_diff_at_t_shape(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        surv_pred = np.random.default_rng(3).random((n, len(times)))

        score, diff = km_calibration(y, times, surv_pred, return_diff_at_t=True)

        assert isinstance(score, float)
        assert diff.shape == times.shape

    def test_function_matches_class_api(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        surv_pred = np.random.default_rng(5).random((n, len(times)))

        score_func = km_calibration(y, times, surv_pred)
        score_class = _KMCalibration().fit(y).score(times, surv_pred)

        assert score_func == pytest.approx(score_class, rel=1e-10)

    def test_alpha_parameter_forwarded(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        surv_pred = np.random.default_rng(9).random((n, len(times)))

        score2 = km_calibration(y, times, surv_pred, alpha=2)
        score4 = km_calibration(y, times, surv_pred, alpha=4)

        assert score2 != pytest.approx(score4)

    def test_difference_at_t_is_signed_and_preserves_time_order(
        self, survival_data, times
    ):
        y = survival_data
        n = len(y["event"])
        km_surv = _KaplanMeierSampler().fit(y).survival_func_(times)

        pred_over = np.clip(np.tile(km_surv + 0.15, (n, 1)), 0.0, 1.0)
        pred_under = np.clip(np.tile(km_surv - 0.15, (n, 1)), 0.0, 1.0)

        _, diff_over = km_calibration(y, times, pred_over, return_diff_at_t=True)
        _, diff_under = km_calibration(y, times, pred_under, return_diff_at_t=True)

        assert_allclose(diff_over, pred_over.mean(axis=0) - km_surv, atol=1e-12)
        assert_allclose(diff_under, pred_under.mean(axis=0) - km_surv, atol=1e-12)
        assert np.all(diff_over >= 0.0)
        assert np.any(diff_over > 0.0)
        assert np.all(diff_under <= 0.0)
        assert np.any(diff_under < 0.0)

        rng = np.random.default_rng(11)
        perm = rng.permutation(len(times))
        _, diff_shuffled = km_calibration(
            y, times[perm], pred_under[:, perm], return_diff_at_t=True
        )
        assert_allclose(diff_shuffled, diff_under[perm], atol=1e-12)


class TestRecalibrateSurvivalFunction:
    @pytest.mark.parametrize("shift", [0.15, -0.15])
    def test_recalibrate_predictions_corrects_over_and_under_prediction(
        self, survival_data, times, shift
    ):
        y = survival_data
        n = len(y["event"])
        km_surv = _KaplanMeierSampler().fit(y).survival_func_(times)
        pred = np.clip(np.tile(km_surv + shift, (n, 1)), 0.0, 1.0)

        score_before = km_calibration(y, times, pred)
        recalibrated = recalibrate_survival_function_predictions(
            pred_test=pred,
            pred_calibration=pred,
            y_calibration=y,
            times=times,
        )
        score_after = km_calibration(y, times, recalibrated)

        assert score_before > 0.0
        assert score_after == pytest.approx(0.0, abs=1e-10)
        assert_allclose(recalibrated.mean(axis=0), km_surv, atol=1e-10)

    def test_recalibrate_predictions_unsorted_times(self, survival_data, times):
        y = survival_data
        n = len(y["event"])
        km_surv = _KaplanMeierSampler().fit(y).survival_func_(times)
        bias = np.linspace(-0.15, 0.15, len(times))
        pred_sorted = np.clip(np.tile(km_surv + bias, (n, 1)), 0.0, 1.0)

        rng = np.random.default_rng(17)
        perm = rng.permutation(len(times))
        times_shuffled = times[perm]
        pred_shuffled = pred_sorted[:, perm]

        recalibrated = recalibrate_survival_function_predictions(
            pred_test=pred_shuffled,
            pred_calibration=pred_shuffled,
            y_calibration=y,
            times=times_shuffled,
        )
        score_after = km_calibration(y, times_shuffled, recalibrated)

        assert score_after == pytest.approx(0.0, abs=1e-10)
        assert_allclose(recalibrated.mean(axis=0), km_surv[perm], atol=1e-10)

    def test_recalibrate_survival_function_with_estimator(self, survival_data, times):
        y = survival_data
        km_surv = _KaplanMeierSampler().fit(y).survival_func_(times)

        class _UnderPredictingEstimator:
            def predict_survival_function(self, X, eval_times):
                base = _KaplanMeierSampler().fit(y).survival_func_(eval_times)
                return np.clip(np.tile(base - 0.15, (len(X), 1)), 0.0, 1.0)

        X_conf = np.zeros((len(y["event"]), 2))
        X_test = np.zeros((25, 2))
        recalibrated = recalibrate_survival_function(
            X_conf=X_conf,
            y_calibration=y,
            times=times,
            estimator=_UnderPredictingEstimator(),
            X=X_test,
        )

        assert recalibrated.shape == (25, len(times))
        assert_allclose(recalibrated.mean(axis=0), km_surv, atol=1e-10)
