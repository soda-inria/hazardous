import numpy as np
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .utils import check_y_survival


class IpcwEstimator(BaseEstimator):
    """Estimate the Inverse Probability Censoring Weight (IPCW).

    This estimator compute the inverse censoring probability,
    using the Kaplan Meier estimator on the censoring
    instead of the event.

    Parameters
    ----------
    min_censoring_prob: float, default=1e-30
        Lower bound of the censoring probability used to avoid zero-division
    """

    def __init__(self, min_censoring_prob=1e-30):
        self.min_censoring_prob = (
            min_censoring_prob  # XXX: study the effect and set a better default
        )

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
        self.censor_probs_ = df.values[:, 0]
        self.censor_probs_func_ = interp1d(
            self.unique_times_,
            self.censor_probs_,
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
        check_is_fitted(self, "censor_probs_func_")

        last_censoring = self.unique_times_[-1]
        is_beyond_last = times > last_censoring

        if any(is_beyond_last):
            raise ValueError(
                "'times' can't be higher than the last observed "
                f"duration: {last_censoring}"
            )

        censor_probs = self.censor_probs_func_(times)
        censor_probs = np.clip(censor_probs, self.min_censoring_prob, 1)

        return 1 / censor_probs
