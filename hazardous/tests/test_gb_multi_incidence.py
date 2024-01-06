import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.interpolate import interp1d

from hazardous._gb_multi_incidence import WeightedMultiClassTargetSampler
from hazardous._ipcw import AlternatingCensoringEst
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

SEED_RANGE = range(3)


@pytest.mark.parametrize("seed", SEED_RANGE)
def test_simple_weight_sampler_draw(seed):
    _, y = make_synthetic_competing_weibull(
        independent_censoring=False,
        complex_features=True,
        return_X_y=True,
        target_rounding=None,
        random_state=seed,
    )
    y_sample = y.head(10)
    event, duration = y_sample["event"].to_numpy(), y_sample["duration"].to_numpy()

    ws = WeightedMultiClassTargetSampler(y_sample, random_state=seed)
    sampled_times, y_targets, sample_weight = ws.draw()

    before_horizon = duration < sampled_times.ravel()

    # Check that y_targets is multiclass.
    assert_array_equal(y_targets[before_horizon], event[before_horizon])

    # Check that all events after horizon are censored.
    assert all(y_targets[~before_horizon] == 0)

    # Check that all sample weights for previous censoring are 0.
    previous_censoring = before_horizon & (event == 0)
    assert all(sample_weight[previous_censoring] == 0)
    assert all(sample_weight[~previous_censoring] > 0)


class KaplanMeierEstimator:
    def fit(self, y):
        self.km_ = KaplanMeierFitter()
        self.km_.fit(
            durations=y["duration"],
            event_observed=y["event"] > 0,
        )

        df = self.km_.cumulative_density_
        self.cumulative_density_ = df.values[:, 0]
        self.unique_times_ = df.index

        self.cumulative_density_func_ = interp1d(
            self.unique_times_,
            self.cumulative_density_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return self

    def predict_proba(self, times):
        if times.ndim == 2:
            times = times[:, 0]
        any_event = self.cumulative_density_func_(times).reshape(-1, 1)
        # The class 0 is the survival to any event S(t).
        # The class 1 is the any event incidence.
        return np.hstack([1 - any_event, any_event])


def test_weighted_binary_targets():
    y_sample = pd.DataFrame(
        dict(
            event=[0, 0, 1, 2, 2],
            duration=[400, 1200, 700, 1100, 950],
        )
    )
    sampled_times = np.array([1000, 400, 1200, 700, 1400])

    duration = y_sample["duration"]

    # Multi incidence estimation
    wb = WeightedMultiClassTargetSampler(y_sample)
    y_binary, sample_weight = wb._weighted_binary_targets(
        wb.any_event_train,
        duration,
        sampled_times,
        ipcw_y_duration=wb.ipcw_train,
        ipcw_training=False,
    )
    y_targets = (y_binary * wb.event_train).to_numpy()

    expected_y_targets = np.array([0, 0, 1, 0, 2])
    expected_sample_weight = np.array([0.0, 1.25, 1.25, 1.25, 1.25])

    assert_array_equal(y_targets, expected_y_targets)
    assert_array_equal(sample_weight, expected_sample_weight)

    # Censoring estimation
    incidence_est = KaplanMeierEstimator()
    ipcw_est = AlternatingCensoringEst(incidence_est=incidence_est)
    wb = WeightedMultiClassTargetSampler(y_sample, ipcw_est=ipcw_est)

    # fit a regular Kaplan Meier estimator to estimate S(t).
    incidence_est.fit(y_sample)

    # compute 1 / S(t_i)
    wb.inv_any_survival_train = wb.ipcw_est.compute_ipcw_at(
        wb.duration_train,
        ipcw_training=True,
        X=None,
    )

    all_time_censoring = wb.any_event_train == 0
    y_targets, sample_weight = wb._weighted_binary_targets(
        all_time_censoring,
        duration,
        sampled_times,
        ipcw_y_duration=wb.inv_any_survival_train,
        ipcw_training=True,
        X=None,
    )

    expected_y_targets = np.array([1, 0, 0, 0, 0])
    expected_sample_weight = np.array([1.0, 1.0, 0, 1.3333333, 0])

    assert_array_equal(y_targets, expected_y_targets)
    assert_almost_equal(sample_weight, expected_sample_weight)


def test_weight_sampler_fit(seed=0):
    X, y = make_synthetic_competing_weibull(
        n_events=3,
        n_samples=3000,
        n_features=2,
        independent_censoring=False,
        complex_features=True,
        return_X_y=True,
        target_rounding=None,
        random_state=seed,
    )

    incidence_est = KaplanMeierEstimator()
    ipcw_est = AlternatingCensoringEst(incidence_est=incidence_est)
    ws = WeightedMultiClassTargetSampler(
        y,
        ipcw_est=ipcw_est,
        random_state=seed,
        n_iter_before_feedback=10,
    )

    # Mock the training of the tree-based incidence estimator.
    incidence_est.fit(y)

    ws.fit(X)

    expected_classes = np.array([0, 1])
    assert_array_equal(ipcw_est.censoring_est_.classes_, expected_classes)
