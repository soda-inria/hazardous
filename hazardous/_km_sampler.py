import numpy as np
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator

from .utils import check_y_survival


class _KaplanMeierSampler(BaseEstimator):
    """Wrapper around the Kaplan-Meier estimator to estimate the
    censoring survival function
    and the inverse of the survival function.


    Attributes
    ----------
    min_censoring_prob_ : float
        The effective minimal probability used, defined as the max between
        min_censoring_prob and the minimum predicted probability.

    unique_times_ : ndarray of shape (n_unique_times,)
        The observed censoring durations from the training target.

    censoring_survival_probs_ : ndarray of shape (n_unique_times,)
        The estimated censoring survival probabilities.

    censoring_survival_func_ : callable
        The linear interpolation function defined with unique_times_ (x) and
        censoring_survival_probs_ (y).
    """

    def fit(self, y):
        """Marginal estimation of the censoring survival function

        In addition to running the Kaplan-Meier estimator on the negated event
        labels (1 for censoring, 0 for any event), this methods also fits
        interpolation function to be able to make prediction at any time.

        Parameters
        ----------
        y : array-like of shape (n_samples, 2)
            The target data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        event, duration = check_y_survival(y)

        km = KaplanMeierFitter()
        km.fit(
            durations=duration,
            event_observed=event,
        )

        df = km.survival_function_
        self.unique_times_ = df.index
        self.survival_probs_ = df.values[:, 0]

        self.survival_func_ = interp1d(
            x=self.unique_times_,
            y=self.survival_probs_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.inverse_surv_func_ = interp1d(
            x=self.survival_probs_[::-1],
            y=self.unique_times_[::-1],
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.min_survival_prob_ = self.survival_probs_.min()
        self.min_positive_survival_prob_ = self.survival_probs_[
            self.survival_probs_ > 0
        ].min()

        return self


class _AalenJohansenSampler(BaseEstimator):
    def fit(self, y):
        event, duration = check_y_survival(y)
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))

        self.incidence_func_ = {}
        for event_id in self.event_ids_[1:]:
            aj = AalenJohansenFitter(calculate_variance=False)
            aj.fit(
                durations=duration,
                event_observed=event,
                event_of_interest=event_id,
            )

            df = aj.cumulative_density_
            times_event = df.index
            y_pred = df.values[:, 0]

            times_event = np.hstack([[0], times_event, [np.inf]])
            y_pred = np.hstack([[0], y_pred, [y_pred[-1]]])

            self.incidence_func_[event_id] = interp1d(
                x=times_event,
                y=y_pred,
                kind="previous",
                bounds_error=False,
                fill_value="extrapolate",
            )
        y_pred_surv = 1 - np.sum(
            [
                self.incidence_func_[event_id](times_event)
                for event_id in self.event_ids_[1:]
            ],
            axis=0,
        )

        self.survival_func_ = interp1d(
            x=times_event,
            y=y_pred_surv,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return self
