import numpy as np
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from scipy.interpolate import interp1d

from .utils import check_y_survival


class _KaplanMeierSampler:
    # TODO docstring
    """Estimate the Inverse Probability of Censoring Weight (IPCW).

    This class estimates the inverse probability of 'survival' to censoring using the
    Kaplan-Meier estimator applied to a binary indicator for censoring, defined as the
    negation of the binary indicator for any event occurrence. This estimator assumes
    that the censoring distribution is independent of the covariates X. If this
    assumption is violated, the estimator may be biased, and a conditional estimator
    might be more appropriate.

    This approach is useful for correcting the bias introduced by right censoring in
    survival analysis, particularly when computing model evaluation metrics such as
    the Brier score or the concordance index.

    Note that the term 'IPCW' can be somewhat misleading: IPCW values represent the
    inverse of the probability of remaining censor-free (or uncensored) at a given time.
    For instance, at t=0, the probability of being censored is 0, so the probability of
    being uncensored is 1.0, and its inverse is also 1.0.

    By construction, IPCW values are always greater than or equal to 1.0 and can only
    increase over time. If no observations are censored, the IPCW values remain
    uniformly at 1.0.

    Note: This estimator extrapolates by maintaining a constant value equal to the last
    observed IPCW value beyond the last recorded time point.

    Parameters
    ----------
    epsilon_censoring_prob : float, default=0.05
        Lower limit of the predicted censoring probabilities. It helps avoiding
        instabilities during the division to obtain IPCW.

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
            x=self.survival_probs_,
            y=self.unique_times_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.min_survival_prob_ = self.survival_probs_.min()
        self.min_positive_survival_prob_ = self.survival_probs_[
            self.survival_probs_ > 0
        ].min()

        return self


class _AalenJohansenSampler:
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

        return self
