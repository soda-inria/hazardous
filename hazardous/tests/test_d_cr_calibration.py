import numpy as np
import pytest

from hazardous.metrics._d_calibration import (
    d_calibration,
    d_cr_calibration,
    d_cr_calibration_ks_test,
    d_cr_calibration_per_event,
)


def _make_competing_risks_data(n=500, seed=0):
    """Generate synthetic competing risks data."""
    rng = np.random.default_rng(seed)
    duration = rng.exponential(scale=10.0, size=n)
    event = rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3], size=n)
    return {"event": event, "duration": duration}


@pytest.fixture
def y():
    return _make_competing_risks_data(n=500, seed=7)


@pytest.fixture
def y_single_event():
    """Single event (no competing risks)."""
    rng = np.random.default_rng(42)
    n = 300
    return {
        "event": rng.binomial(1, p=0.6, size=n),
        "duration": rng.exponential(scale=10.0, size=n),
    }


# ============================================================================
# d_calibration: per-bucket calibration values
# ============================================================================


class TestDCalibration:
    """Tests for d_calibration (per-bucket values)."""

    def test_returns_dataframe(self, y):
        """Output should be a pandas DataFrame."""
        import pandas as pd

        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_calibration(fk, fk_infty, s_t, y)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_shape(self, y):
        """Output shape should match n_buckets."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk
        n_buckets = 50

        result = d_calibration(fk, fk_infty, s_t, y, n_buckets=n_buckets)
        assert result.shape == (n_buckets, 1)
        assert result.index.max() == n_buckets

    def test_cumsum_property(self, y):
        """Values should be monotonically increasing (cumulative)."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_calibration(fk, fk_infty, s_t, y, n_buckets=50)
        values = result.iloc[:, 0].values
        # Check monotonic increase
        assert np.all(np.diff(values) >= -1e-10)

    def test_output_positive(self, y):
        """Output values should be non-negative."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_calibration(fk, fk_infty, s_t, y, n_buckets=50)
        # All values should be non-negative
        assert (result.values >= 0).all()

    def test_event_of_interest_any(self, y):
        """event_of_interest='any' should work."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_calibration(fk, fk_infty, s_t, y, event_of_interest="any")
        assert isinstance(result, type(d_calibration(fk, fk_infty, s_t, y)))

    def test_event_of_interest_specific(self, y):
        """event_of_interest=int should work."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_calibration(fk, fk_infty, s_t, y, event_of_interest=1)
        assert isinstance(result, type(d_calibration(fk, fk_infty, s_t, y)))

    def test_numerical_stability(self, y):
        """Should handle zero survival with epsilon."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = np.zeros(n) + 1e-6

        result = d_calibration(fk, fk_infty, s_t, y, epsilon=1e-3)
        assert not np.any(np.isnan(result.values))
        assert not np.any(np.isinf(result.values))


# ============================================================================
# d_cr_calibration_per_event: integrated calibration with alpha exponent
# ============================================================================


class TestDCRCalibrationPerEvent:
    """Tests for d_cr_calibration_per_event (integrated per-event score)."""

    def test_returns_dict(self, y):
        """Output should be a dict with event ids as keys."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_cr_calibration_per_event(fk, fk_infty, s_t, y)
        assert isinstance(result, dict)
        assert all(isinstance(k, (int, np.integer)) for k in result.keys())
        assert all(isinstance(v, float) for v in result.values())

    def test_scores_are_non_negative(self, y):
        """Scores should always be non-negative."""
        n = len(y["event"])
        fk = np.random.uniform(0.1, 0.4, n)
        fk_infty = np.random.uniform(0.5, 0.9, n)
        s_t = 1 - fk

        result = d_cr_calibration_per_event(fk, fk_infty, s_t, y, event_of_interest=1)
        assert result >= 0

    def test_event_of_interest_returns_float(self, y):
        """event_of_interest should return a float."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_cr_calibration_per_event(fk, fk_infty, s_t, y, event_of_interest=1)
        assert isinstance(result, float)

    def test_alpha_changes_scores(self, y):
        """Different alpha values should give different scores."""
        n = len(y["event"])
        fk = np.random.uniform(0.1, 0.4, n)
        fk_infty = np.random.uniform(0.5, 0.9, n)
        s_t = 1 - fk

        score2 = d_cr_calibration_per_event(fk, fk_infty, s_t, y, alpha=2)
        score4 = d_cr_calibration_per_event(fk, fk_infty, s_t, y, alpha=4)

        assert score2 != pytest.approx(score4)

    def test_alpha_parameter_affects_score(self, y):
        """Different alpha values produce different scores."""
        n = len(y["event"])
        rng = np.random.default_rng(99)
        fk = rng.uniform(0.1, 0.4, n)
        fk_infty = rng.uniform(0.5, 0.9, n)
        s_t = 1 - fk

        score1 = d_cr_calibration_per_event(
            fk, fk_infty, s_t, y, alpha=1, event_of_interest=1
        )
        score2 = d_cr_calibration_per_event(
            fk, fk_infty, s_t, y, alpha=2, event_of_interest=1
        )
        score4 = d_cr_calibration_per_event(
            fk, fk_infty, s_t, y, alpha=4, event_of_interest=1
        )

        # All three should be different (alpha affects the integral)
        assert score1 != score2
        assert score2 != score4


# ============================================================================
# d_cr_calibration: overall aggregated calibration
# ============================================================================


class TestDCRCalibration:
    """Tests for d_cr_calibration (overall aggregated score)."""

    def test_returns_float(self, y):
        """Output should be a float."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_cr_calibration(fk, fk_infty, s_t, y)
        assert isinstance(result, float)

    def test_mean_reduction(self, y):
        """Mean reduction should match manual computation."""
        n = len(y["event"])
        rng = np.random.default_rng(42)
        fk = rng.uniform(0.1, 0.4, n)
        fk_infty = rng.uniform(0.5, 0.9, n)
        s_t = 1 - fk

        per_event = d_cr_calibration_per_event(fk, fk_infty, s_t, y)
        expected = float(np.mean(list(per_event.values())))
        score = d_cr_calibration(fk, fk_infty, s_t, y, reduction="mean")
        assert score == pytest.approx(expected, rel=1e-10)

    def test_sum_reduction(self, y):
        """Sum reduction should match manual computation."""
        n = len(y["event"])
        rng = np.random.default_rng(43)
        fk = rng.uniform(0.1, 0.4, n)
        fk_infty = rng.uniform(0.5, 0.9, n)
        s_t = 1 - fk

        per_event = d_cr_calibration_per_event(fk, fk_infty, s_t, y)
        expected = float(np.sum(list(per_event.values())))
        score = d_cr_calibration(fk, fk_infty, s_t, y, reduction="sum")
        assert score == pytest.approx(expected, rel=1e-10)

    def test_max_reduction(self, y):
        """Max reduction should match manual computation."""
        n = len(y["event"])
        rng = np.random.default_rng(44)
        fk = rng.uniform(0.1, 0.4, n)
        fk_infty = rng.uniform(0.5, 0.9, n)
        s_t = 1 - fk

        per_event = d_cr_calibration_per_event(fk, fk_infty, s_t, y)
        expected = float(np.max(list(per_event.values())))
        score = d_cr_calibration(fk, fk_infty, s_t, y, reduction="max")
        assert score == pytest.approx(expected, rel=1e-10)

    def test_invalid_reduction_raises(self, y):
        """Invalid reduction should raise ValueError."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        with pytest.raises(ValueError, match="reduction must be"):
            d_cr_calibration(fk, fk_infty, s_t, y, reduction="bad")


# ============================================================================
# d_cr_calibration_ks_test: KS test for calibration
# ============================================================================


class TestDCRCalibrationKSTest:
    """Tests for d_cr_calibration_ks_test (KS test for calibration)."""

    def test_returns_dict_with_results(self, y):
        """Output should be a dict with statistic and pvalue."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_cr_calibration_ks_test(fk, fk_infty, s_t, y)
        assert isinstance(result, dict)
        for event_id, test_result in result.items():
            assert "statistic" in test_result
            assert "pvalue" in test_result
            assert isinstance(test_result["statistic"], float)
            assert isinstance(test_result["pvalue"], float)

    def test_statistic_in_valid_range(self, y):
        """KS statistic should be in [0, 1]."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_cr_calibration_ks_test(fk, fk_infty, s_t, y)
        for test_result in result.values():
            assert 0 <= test_result["statistic"] <= 1

    def test_pvalue_in_valid_range(self, y):
        """p-values should be in [0, 1]."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_cr_calibration_ks_test(fk, fk_infty, s_t, y)
        for test_result in result.values():
            assert 0 <= test_result["pvalue"] <= 1

    def test_event_of_interest_returns_single_result(self, y):
        """event_of_interest should return a single result dict."""
        n = len(y["event"])
        fk = np.random.uniform(0, 0.5, n)
        fk_infty = np.random.uniform(0.4, 1.0, n)
        s_t = 1 - fk

        result = d_cr_calibration_ks_test(fk, fk_infty, s_t, y, event_of_interest=1)
        assert isinstance(result, dict)
        assert "statistic" in result
        assert "pvalue" in result

    def test_ks_test_consistency(self, y):
        """KS statistic should be consistent with d_calibration differences."""
        n = len(y["event"])
        fk = np.random.uniform(0.1, 0.4, n)
        fk_infty = np.random.uniform(0.5, 0.9, n)
        s_t = 1 - fk

        # Compute calibration curves
        calib_df = d_calibration(fk, fk_infty, s_t, y, event_of_interest=1)
        b_hat = calib_df.values.flatten()
        rho_values = np.linspace(1 / 100, 1, 100)

        # Maximum deviation should match KS statistic
        max_deviation = np.max(np.abs(b_hat - rho_values))

        result = d_cr_calibration_ks_test(fk, fk_infty, s_t, y, event_of_interest=1)

        # KS statistic should be close to manual computation
        assert result["statistic"] == pytest.approx(max_deviation, rel=0.01)

    def test_miscalibrated_has_low_pvalue(self, y):
        """Miscalibrated predictions should have lower p-values."""
        n = len(y["event"])
        # Predictions all at extreme values (bad calibration)
        fk = np.ones(n) * 0.9
        fk_infty = np.ones(n) * 0.95
        s_t = np.ones(n) * 0.05

        result = d_cr_calibration_ks_test(fk, fk_infty, s_t, y)
        # At least some should have low p-values
        pvalues = [r["pvalue"] for r in result.values()]
        assert any(p < 0.5 for p in pvalues)  # Some miscalibration signals
