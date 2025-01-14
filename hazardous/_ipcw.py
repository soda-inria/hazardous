import numpy as np
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted

from .utils import check_y_survival


class KaplanMeierIPCW:
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

    def __init__(self, epsilon_censoring_prob=0.05):
        self.epsilon_censoring_prob = epsilon_censoring_prob

    def fit(self, y, X=None):
        """Marginal estimation of the censoring survival function

        In addition to running the Kaplan-Meier estimator on the negated event
        labels (1 for censoring, 0 for any event), this methods also fits
        interpolation function to be able to make prediction at any time.

        Parameters
        ----------
        y : array-like of shape (n_samples, 2)
            The target data.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        Returns
        -------
        self : object
            Fitted estimator.
        """
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

        min_censoring_prob = self.censoring_survival_probs_[
            self.censoring_survival_probs_ > 0
        ].min()

        self.min_censoring_prob_ = max(
            min_censoring_prob,
            self.epsilon_censoring_prob,
        )
        self.censoring_survival_func_ = interp1d(
            self.unique_times_,
            self.censoring_survival_probs_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return self

    def compute_ipcw_at(self, times, X=None, ipcw_training=False):
        """Estimate the inverse probability of censoring weights at given time horizons.

        Compute the inverse of the linearly interpolated censoring survival
        function.

        Parameters
        ----------
        times : np.ndarray of shape (n_samples,)
            The input times for which to predict the IPCW for each sample.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        Returns
        -------
        ipcw : np.ndarray of shape (n_samples,)
            The IPCW for each sample at each time horizon.
        """
        check_is_fitted(self, "min_censoring_prob_")

        cs_prob = self.compute_censoring_survival_proba(
            times,
            X=X,
            ipcw_training=ipcw_training,
        )
        cs_prob = np.clip(cs_prob, self.min_censoring_prob_, 1)
        return 1 / cs_prob

    def compute_censoring_survival_proba(self, times, X=None, ipcw_training=False):
        """Estimate probability of not experiencing censoring at times.

        Linearly interpolate the censoring survival function.

        Parameters
        ----------
        times : np.ndarray of shape (n_times,)
            The input times for which to predict the IPCW.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        ipcw_training : bool, default=False
            Unused.

        Returns
        -------
        ipcw : np.ndarray of shape (n_times,)
            The IPCW for times
        """
        return self.censoring_survival_func_(times)


class AlternatingCensoringEstimator(KaplanMeierIPCW):
    r"""IPCW estimator for the alternating censoring estimation strategy used by \
    SurvivalBoost.

    Predicts :math:`\hat{G}(t | X = x) = P(C > t | X = x)` using
    :math:`1/\hat{S}(t | X = x) = 1/P(T^* > t | X = x)` as its IPCWs.

    Like ``SurvivalBoost``, this class uses a Histogram-based Gradient Boosting (HGB)
    classifier with the log-loss to estimate the probability of "survival" to censoring.

    Parameters
    ----------
    cold_start_ipcw_estimator : object, default=None
        The estimator considered to compute the probabilities of survival to censoring
        before the censoring estimator has been fit.
        If set to ``None``, the ``KaplanMeierIPCW`` is used, as it doesn't require an
        extensive training.

    incidence_estimator : object, default=None
        The antagonist estimator to the censoring estimator, where the former predicts
        the probability of survival to any event, as well as the cause-specific
        cumulative incidence functions (CIFs).
        The incidence estimator is fit outside of this class, and its predictions are
        used as sample weights to fit the censoring estimator.

    learning_rate : float, default=0.05
        The learning rate, similar to the argument in SurvivalBoost constructor.

    max_depth : int, default=None
        The maximum depth of each tree, similar to the argument in SurvivalBoost
        constructor.

    min_samples_leaf : int, default=50
        The minimum number of samples per leaf.

    epsilon_censoring_prob : float, default=0.05
        Lower limit of the predicted censoring probabilities. It helps avoiding
        instabilities during the division to obtain IPCW

    Attributes
    ----------
    cold_start_ipcw_estimator_ : object
        The fitted cold start censoring probability estimator.
    """

    def __init__(
        self,
        cold_start_ipcw_estimator=None,
        incidence_estimator=None,
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        epsilon_censoring_prob=0.05,
    ):
        self.cold_start_ipcw_estimator = cold_start_ipcw_estimator
        self.incidence_estimator = incidence_estimator
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        super().__init__(epsilon_censoring_prob=epsilon_censoring_prob)

    def fit(self, y, X=None):
        """Fit the cold start IPCW estimator.

        This methods should be called only once for the whole training of the
        incidence estimator.

        Parameters
        ----------
        y : array-like of shape (n_samples, 2)
            The target dataframe with 'event' and 'duration' columns.

        X : array-like of shape (n_samples, n_features), default=None
            The input samples. Ignored if the cold start estimator is
            the ``KaplanMeierIPCW``.

        Returns
        -------
        self : object
            Fitted estimator
        """
        super().fit(y)

        self.check_cold_start_ipcw_estimator()
        self.cold_start_ipcw_estimator_.fit(y, X=X)

        return self

    def fit_censoring_estimator(self, X, y_binary, times, sample_weight):
        """Fit the censoring classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y_binary : ndarray of shape (n_samples,)
            Binary censoring indicator (1 for censoring, 0 for any event).

        times : ndarray of shape (n_samples,)
            Times of observation for each sample.

        sample_weight : ndarray of shape (n_samples,)
            Inverse probability of survival to any event :math:`P(T^* > t)`.
            Antagonist to the IPCW.

        Return
        ------
        self : object
            Fitted estimator.
        """
        if not hasattr(self, "censoring_estimator_"):
            self.censoring_estimator_ = self._build_censoring_estimator()

        X_with_time = np.hstack([times, X])
        self.censoring_estimator_.max_iter += 1
        self.censoring_estimator_.fit(
            X_with_time, y_binary, sample_weight=sample_weight
        )

        return self

    def compute_censoring_survival_proba(self, times, X=None, ipcw_training=False):
        r"""Predict the censoring probability at some given times.

        The probabilities returned by the incidence estimator are
        :math:`\hat{S}(t| X = x) = P(T^* > t | X = x)` and
        :math:`\hat{F}_k(t| X = x) = P(T^* \leq t \cap \Delta = k | X = x)`
        for each :math:`k` competing event.

        The probabilities returned by the censoring estimator are
        :math:`\hat{G}(t) = P(C > t | X = x)` and
        :math:`\hat{G}(t) = P(C \leq t | X = x)`.

        Parameters
        ----------
        times : ndarray of shape (n_samples,)
            The time horizons used to predict the censoring survival probabilities.

        X : array-like of shape (n_samples, n_features), default=None
            The input samples.

        ipcw_training : bool, default=False
            * If set to True, returns the predicted probability of survival to
            any event, using the external 'incidence_estimator'.
            * If set to False (default), returns the predicted probability of survival
            to censoring, using the internal 'censoring_estimator_'
            (or using the cold start IPCW estimator for the first training iteration).
        """
        if ipcw_training:
            # incidence_estimator is trained to predict the survival to any event S(t)
            # (class 0) and the incidence of all events (class 1 to K).
            if X is None:
                X = times
            else:
                X = np.hstack([times.reshape(-1, 1), X])
            return self.incidence_estimator.predict_proba(X)[:, 0]

        else:
            if not hasattr(self, "censoring_estimator_"):
                return self.cold_start_ipcw_estimator_.compute_censoring_survival_proba(
                    times
                )

            X_with_time = np.hstack([times.reshape(-1, 1), X])
            return self.censoring_estimator_.predict_proba(X_with_time)[:, 0]

    def check_cold_start_ipcw_estimator(self):
        if self.cold_start_ipcw_estimator is None:
            self.cold_start_ipcw_estimator_ = KaplanMeierIPCW(
                epsilon_censoring_prob=self.epsilon_censoring_prob
            )
        else:
            self.cold_start_ipcw_estimator_ = clone(
                self.cold_start_ipcw_estimator, safe=False
            )

    def _build_censoring_estimator(self):
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=1,
            warm_start=True,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )
