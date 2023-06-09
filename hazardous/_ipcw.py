import numpy as np
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class IpcwEstimator(BaseEstimator):
    """Inverse Probability Censoring Weight Estimator.

    This estimator compute the inverse censoring probability,
    using the Kaplan Meier estimator on the censoring
    instead of the event.
    """

    def __init__(self, min_censoring_prob=1e-30):
        self.min_censoring_prob = (
            min_censoring_prob  # XXX: study the effect and set a better default
        )

    def fit(self, y):
        km = KaplanMeierFitter()
        censoring = y["event"] == 0
        km.fit(
            durations=y["duration"],
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
