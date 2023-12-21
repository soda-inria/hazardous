import numpy as np
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from scipy.stats import weibull_min
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .utils import check_y_survival


class IPCWEstimator(BaseEstimator):
    """Estimate the Inverse Probability of Censoring Weight (IPCW).

    This class estimates the inverse of the probability of "survival" to
    censoring using the Kaplan-Meier estimator on a binary indicator for
    censoring, that is the negative of the binary indicator for any-event
    occurrence.

    This is useful to correct for the bias introduced by right censoring in
    survival analysis when computing model evaluation metrics such as the Brier
    score or the concordance index.

    Note that the name IPCW name is a bit misleading: IPCW values are the
    inverse of the probability of remaining censoring-free (or uncensored) at a
    given time: at t=0, the probability of being censored is 0, therefore the
    probability of being uncensored is 1.0, and its inverse is also 1.0.

    By construction, IPCW values are always larger or equal to 1.0 and can only
    increase with time. If no observations are censored, the IPCW values are
    uniformly 1.0.

    Note: this estimator extrapolates with a constant value equal to the last
    IPCW value beyond the last observed time.
    """

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
        self.min_censoring_prob_ = self.censoring_survival_probs_[
            self.censoring_survival_probs_ > 0
        ].min()
        return self

    def compute_ipcw_at(self, times):
        """Estimate inverse censoring weight probability at times.

        Linearly interpolate the censoring survival function and return the
        inverse values.

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

        cs_prob = self.censoring_survival_func_(times)
        cs_prob = np.clip(cs_prob, self.min_censoring_prob_, 1)
        return 1 / cs_prob


class IPCWSampler(BaseEstimator):
    """Compute the True survival probabilities based on the \
        distribution parameters.

    Parameters
    ----------
    shape : float or ndarray of shape (n_samples,)
        Weibull distribution shape parameter.

    scale : float or ndarray of shape (n_samples,)
        Weibull distribution scale parameter.
    """

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def fit(self, y):
        """No-op"""
        return self

    def compute_ipcw_at(self, times):
        """Compute the IPCW for a given array of time.

        Parameters
        ----------
        times : ndarray of shape (n_samples, 1)

        Returns
        -------
        ipcw_y : pandas.DataFrame of shape (n_samples, )
            True Survival Probability at each t_i| x_i in times
            G^*(t_i|x_i).
        """
        return 1 - weibull_min.cdf(times, self.shape, scale=self.scale)
