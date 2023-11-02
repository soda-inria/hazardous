import numpy as np
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .utils import check_event_of_interest, check_y_survival


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

    Parameters
    ----------
    event_of_interest : int or any
        If integer, this estimator estimates the aggregate survival function to
        either censoring or any other competing event.
        If "any", this estimator estimates to censoring.
    """

    def __init__(self, event_of_interest="any"):
        self.event_of_interest = event_of_interest

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
        check_event_of_interest(self.event_of_interest)

        km_censoring = KaplanMeierFitter()
        km_censoring.fit(
            durations=duration,
            event_observed=event == 0,
        )
        censoring_survival_df = km_censoring.survival_function_
        self.unique_times_ = censoring_survival_df.index
        self.censoring_survival_probs_ = censoring_survival_df.values[:, 0]

        if self.event_of_interest != "any":
            # Adjust for competing events by multiplying by the event specific
            # "survival" functions (complement of cumulated density functions)
            # for each competing event.
            competing_event_ids = np.unique(event[event != 0])
            competing_event_ids = competing_event_ids[
                competing_event_ids != self.event_of_interest
            ]
            for event_id in competing_event_ids:
                # TODO: unhardcode the jitter seed for tied times handling by
                # passing a random state argument to IPWEstimator.
                #
                # TODO: the AJ estimators recompute the censoring survival
                # using KM internally: we could avoid redundant computation by
                # reimplmenting the AJ estimators directly instead of relying
                # on lifelines.
                ajf = AalenJohansenFitter(calculate_variance=False, seed=0)
                ajf.fit(
                    durations=duration,
                    event_observed=event,
                    event_of_interest=event_id,
                )
                cif_df = ajf.cumulative_density_
                assert (self.unique_times_ == cif_df.index.values).all()
                competing_cif = cif_df[f"CIF_{event_id}"].values
                self.censoring_survival_probs_ *= 1 - competing_cif

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
