from copy import deepcopy
from warnings import warn

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import check_array, check_is_fitted
from sklearn.ensemble._bagging import BaseBagging
from sklearn.utils._param_validation import HasMethods

from ._survival_boost import SurvivalBoost
from .base import SurvivalMixin
from .metrics import mean_integrated_brier_score
from .utils import (
    _dict_to_recarray,
    check_y_survival,
    get_unique_events,
    make_time_grid,
)


class BaggingSurvival(BaseBagging, SurvivalMixin):
    """TODO"""

    _parameter_constraints = deepcopy(BaseBagging._parameter_constraints)
    _parameter_constraints["estimator"] = [
        HasMethods(["fit", "score", "predict_cumulative_incidence"])
    ]

    def __init__(
        self,
        estimator=None,
        n_estimators=3,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def _get_estimator(self):
        """Resolve which estimator to return"""
        if self.estimator is None:
            return SurvivalBoost(show_progressbar=False)
        return self.estimator

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_events_ = self.n_events_
        n_time_steps_ = self.time_grid_.shape[0]

        y_pred = np.zeros((n_samples, n_events_, n_time_steps_))

        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            y_pred[mask, :] += estimator.predict_proba((X[mask, :])[:, features])

        if (y_pred.sum(axis=(1, 2)) == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few estimators were used "
                "to compute any reliable oob estimates."
            )

        self.oob_score_ = -mean_integrated_brier_score(
            y_train=self.weighted_targets_.y_train,
            y_test=y,
            y_pred=y_pred,
            time_grid=self.time_grid_,
        )

    def _validate_y(self, y):
        event, duration = check_y_survival(y)
        self.event_ids_ = get_unique_events(event)
        self.n_events_ = len(self.event_ids_)

        base_estimator = self._get_estimator()
        self.time_grid_ = make_time_grid(
            event,
            duration,
            base_estimator.n_time_grid_steps,
        )
        self.y_train_ = y  # XXX: Used by SurvivalMixin.score()
        self.time_horizon_ = base_estimator.time_horizon

        return y

    def fit(self, X, y, **fit_params):
        y = _dict_to_recarray(y)
        return super().fit(X, y, **fit_params)

    def predict_cumulative_incidence(self, X, times=None):
        """TODO"""
        check_is_fitted(self)

        # Check data
        X = check_array(X, force_all_finite="allow-nan")

        # Parallel loop
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        # Get time grid
        times = times or self.time_grid_

        all_proba = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_predict_cumulative_incidence)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
                times,
                self.n_events_,
            )
            for i in range(n_jobs)
        )

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba

    def predict_survival_function(self, X, times=None):
        return self.predict_cumulative_incidence(X, times=times)[:, 0, :]

    def predict_proba(self, X, time_horizon=None):
        """TODO"""
        check_is_fitted(self)

        # Check data
        X = check_array(X, force_all_finite="allow-nan")

        # Parallel loop
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)

        # Get time grid
        time_horizon = time_horizon or self.time_horizon_

        all_proba = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i] : starts[i + 1]],
                self.estimators_features_[starts[i] : starts[i + 1]],
                X,
                time_horizon,
                self.n_events_,
            )
            for i in range(n_jobs)
        )

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba


def _parallel_predict_cumulative_incidence(
    estimators,
    estimators_features,
    X,
    times,
    n_events,
):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    n_time_steps = times.shape[0]
    proba = np.zeros((n_samples, n_events, n_time_steps))

    for estimator, features in zip(estimators, estimators_features):
        proba_estimator = estimator.predict_cumulative_incidence(
            X[:, features], times=times
        )

        if n_events == len(estimator.event_ids_):
            proba += proba_estimator

        else:
            proba[:, estimator.event_ids_] += proba_estimator[
                :, range(len(estimator.event_ids_))
            ]

    return proba


def _parallel_predict_proba(
    estimators,
    estimators_features,
    X,
    time_horizon,
    n_events,
):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_events))

    for estimator, features in zip(estimators, estimators_features):
        proba_estimator = estimator.predict_proba(
            X[:, features], time_horizon=time_horizon
        )

        if n_events == len(estimator.event_ids_):
            proba += proba_estimator

        else:
            proba[:, estimator.event_ids_] += proba_estimator[
                :, range(len(estimator.event_ids_))
            ]

    return proba


# Vendored from a private module in sklearn.
def indices_to_mask(indices, mask_length):
    """Convert list of indices to boolean mask.

    Parameters
    ----------
    indices : list-like
        List of integers treated as indices.
    mask_length : int
        Length of boolean mask to be generated.
        This parameter must be greater than max(indices).

    Returns
    -------
    mask : 1d boolean nd-array
        Boolean array that is True where indices are present, else False.
    """
    if mask_length <= np.max(indices):
        raise ValueError("mask_length must be greater than max(indices)")

    mask = np.zeros(mask_length, dtype=bool)
    mask[indices] = True

    return mask


# Vendored from a private module in sklearn.
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs, dtype=int)
    n_estimators_per_job[: n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
