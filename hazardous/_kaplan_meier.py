import numpy as np
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d


class KaplanMeierEstimator:
    """A utils class for the Kaplan Meier estimator."""

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

        self.inverse_surv_func_ = interp1d(
            1 - self.cumulative_density_,
            self.unique_times_,
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

    def predict_quantile(self, quantiles):
        return self.inverse_surv_func_(quantiles)
