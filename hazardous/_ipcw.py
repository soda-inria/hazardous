import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.interpolate import interp1d
from scipy.stats import weibull_min
from sklearn.base import BaseEstimator
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_is_fitted

from .utils import check_y_survival


class BaseIPCW(BaseEstimator):
    def fit(self, y, X=None):
        del X
        event, duration = check_y_survival(y)
        censoring = event == 0

        km = KaplanMeierFitter()
        km.fit(
            durations=duration,
            event_observed=censoring,
        )

        df = km.survival_function_
        self.unique_times_ = df.index
        self.censoring_survival_probs_ = df.values[:, 0]
        self.min_censoring_prob_ = self.censoring_survival_probs_[
            self.censoring_survival_probs_ > 0
        ].min()

        return self

    def compute_ipcw_at(self, times, X=None):
        """Estimate inverse censoring weight probability at times.

        Linearly interpolate the censoring survival function and return the
        inverse values.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data for conditional estimators.

        times : np.ndarray of shape (n_samples,)
            The input times for which to predict the IPCW for each sample.

        Returns
        -------
        ipcw : np.ndarray of shape (n_samples,)
            The IPCW for each sample at each sample time.
        """
        check_is_fitted(self, "min_censoring_prob_")

        cs_prob = self.compute_censoring_survival_proba(times, X=X)
        cs_prob = np.clip(cs_prob, self.min_censoring_prob_, 1)
        return 1 / cs_prob

    def compute_censoring_survival_proba(self, times, X=None):
        raise NotImplementedError()


class IPCWEstimator(BaseIPCW):
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

    def fit(self, y, X=None):
        """Compute the censoring survival function using Kaplan Meier
        and store it as an interpolation function.

        Parameters
        ----------
        y : np.array, dictionnary or dataframe
            The target, consisting in the 'event' and 'duration' columns.

        X : None

        Returns
        -------
        self : object
            Fitted estimator.
        """
        del X
        super().fit(y)
        self.censoring_survival_func_ = interp1d(
            self.unique_times_,
            self.censoring_survival_probs_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return self

    def compute_censoring_survival_proba(self, times, X=None):
        """Estimate inverse censoring weight probability at times.

        Linearly interpolate the censoring survival function and return the
        inverse values.

        Parameters
        ----------
        times : np.ndarray of shape (n_times,)
            The input times for which to predict the IPCW.

        X : None

        Returns
        -------
        ipcw : np.ndarray of shape (n_times,)
            The IPCW for times
        """
        del X
        return self.censoring_survival_func_(times)


class IPCWSampler(BaseIPCW):
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

    def compute_censoring_survival_proba(self, times, X=None):
        """Compute the censoring survival proba G for a given array of time.

        Parameters
        ----------
        times : ndarray of shape (n_samples,)
            The time step to consider for each sample.

        X : None

        Returns
        -------
        cs_prob : ndarray of shape (n_samples)
            The censoring survival probability of each sample.
        """
        del X
        return 1 - weibull_min.cdf(times, self.shape, scale=self.scale)


class IPCWCoxEstimator(BaseIPCW):
    def __init__(
        self,
        transformer=None,
        cox_estimator=None,
        n_time_grid_steps=100,
    ):
        self.transformer = transformer
        self.cox_estimator = cox_estimator
        self.n_time_grid_steps = n_time_grid_steps

    def fit(self, y, X=None):
        """Fit a nonlinear transformer and a CoxPHFitter estimator.

        Parameters
        ----------
        y : pandas.DataFrame of shape (n_samples, 2)
            The target, with 'event' and 'duration' columns.

        X : pandas.DataFrame of shape (n_samples, n_features)
            The covariates to be transformed and fitted.

        Returns
        -------
        self : fitted instance of IPCWCoxEstimator
        """
        super().fit(y)
        self.check_transformer_estimator()

        X_trans = self.transformer_.fit_transform(X)

        frame = X_trans.copy()
        frame["duration"] = y["duration"]
        frame["censoring"] = y["event"] == 0

        # XXX: This could be integrated in the pipeline by using the scikit-learn
        # interface.
        self.cox_estimator_.fit(frame, event_col="censoring", duration_col="duration")

        return self

    def compute_censoring_survival_proba(self, times, X=None):
        """Compute the conditional censoring survival probability.

        A time grid is used to reduce the size of the prediction matrix
        returned by lifelines CoxPHFitter. Each sample matches a time of
        evaluation in the array 'times', so for each sample we only return
        the closest bin to the time of evaluation.

        Parameters
        ----------
        times : ndarray of shape (n_samples,)
            Each time step corresponds to a single sample to be evaluated.

        X : pandas.DataFrame of shape (n_samples, n_features)
            Covariate used by the conditional estimator.

        Returns
        -------
        cs_prob : ndarray of shape (n_samples,)
            The censoring survival probability of each sample evaluated
            at their corresponding time step.
        """
        X_trans = self.transformer_.transform(X)

        unique_times = np.sort(np.unique(times))
        if len(unique_times) > self.n_time_grid_steps:
            t_min, t_max = min(self.unique_times_), max(self.unique_times_)
            time_grid = np.linspace(t_min, t_max, self.n_time_grid_steps)
        else:
            time_grid = unique_times

        # shape (n_unique_times, n_samples)
        cs_prob_all_t = self.cox_estimator_.predict_survival_function(
            X_trans,
            times=time_grid,
        )

        indices = np.searchsorted(time_grid, times)
        cs_prob = cs_prob_all_t.to_numpy()[indices, np.arange(X.shape[0])]

        return cs_prob

    def check_transformer_estimator(self):
        if self.transformer is None:
            self.transformer_ = make_pipeline(
                SplineTransformer(),
                Nystroem(),
            )
        else:
            self.transformer_ = self.transformer
        self.transformer_.set_output(transform="pandas")

        if self.cox_estimator is None:
            self.cox_estimator_ = CoxPHFitter(penalizer=5)
        else:
            self.cox_estimator_ = self.cox_estimator
