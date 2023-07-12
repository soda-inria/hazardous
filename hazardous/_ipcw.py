import numpy as np
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .utils import check_y_survival


class IPCWEstimator(BaseEstimator):
    """Estimate the Inverse Probability Censoring Weight (IPCW).

    Estimate the inverse of the probability of "survival" to censoring using
    the Kaplan-Meier estimator on a binary indicator for censoring, that is the
    negative of the binary indicator for any-event occurrence.

    Note that the name IPCW name is a bit misleading: IPCW values are the
    inverse of the probability of remaining censoring-free (or uncensored) at a
    given time: at t=0, the probability of being censored is 0, therefore the
    probability of being uncensored is 1.0, and its inverse is also 1.0.

    By construction, IPCW values are always larger or equal to 1.0 and can only
    increase with time. If no observations are censored, the IPCW values are
    uniformly 1.0.

    Parameters
    ----------
    min_censoring_survival_prob: float, default=1e-30
        Lower bound of the censoring survival probability used to avoid
        zero-division when taking its inverse to get the IPCW values. As a
        result, IPCW values are upper bounded by the inverse of this value.
    """

    def __init__(self, min_censoring_survival_prob=1e-30):
        # XXX: study the effect and maybe set a better default value.
        self.min_censoring_survival_prob = min_censoring_survival_prob

    def fit(self, y):
        """Compute the censoring survival function using Kaplan Meier
        and store it as an interpolation function.

        Parameters
        ----------
        y : np.array, dictionnary or dataframe
            The target, consisting in the 'event' and 'duration' columns.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        event, duration = check_y_survival(y)

        km = KaplanMeierFitter()
        censoring = event == 0
        km.fit(
            durations=duration,
            event_observed=censoring,
        )

        df = km.survival_function_
        self.unique_times_ = df.index
        self.censoring_survival_probs_ = df.values[:, 0]
        self.censoring_survival_func_ = interp1d(
            self.unique_times_,
            self.censoring_survival_probs_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return self

    def predict(self, times):
        """Predict inverse censoring weight probability for times.

        Predict the linearly interpolated censoring survival function
        and return the inverse.

        Parameters
        ----------
        times : np.ndarray of shape (n_times,)
            The input times for which to predict the IPCW.

        Returns
        -------
        ipcw : np.ndarray of shape (n_times,)
            The IPCW for times
        """
        check_is_fitted(self, "censoring_survival_func_")

        last_censoring = self.unique_times_[-1]
        is_beyond_last = times > last_censoring

        if any(is_beyond_last):
            raise ValueError(
                "'times' can't be higher than the last observed "
                f"duration: {last_censoring}"
            )

        censoring_survival_probs = self.censoring_survival_func_(times)
        censoring_survival_probs = np.clip(
            censoring_survival_probs, self.min_censoring_survival_prob, 1
        )
        return 1 / censoring_survival_probs
