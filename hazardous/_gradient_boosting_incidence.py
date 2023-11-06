import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.utils.validation import check_array, check_random_state
from tqdm import tqdm

from .metrics._brier_score import IncidenceScoreComputer
from .utils import check_y_survival


class WeightedBinaryTargetSampler(IncidenceScoreComputer):
    """Weighted binary targets for censoring-adjusted incidence estimation.

    Cast a cumulative incidence estimation problem with censored event times as
    a binary classification problem with weighted targets.

    This class samples time-horizons uniformly, and for each event (censored or
    not) and each time horizon, computes for each of the the IPCW weights and
    the expected binary targets. By optimizing the stochastic average of a
    proper-scoring rule such as the time-dependent Brier score or binary
    cross-entropy, we obtain an estimator of the cumulative incidence function
    that minimizes the time-integrated Brier Score (IBS) or time-integrated
    Negative Log Likelihood (INLL).

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the uniform time sampler
    """

    def __init__(
        self,
        y_train,
        event_of_interest="any",
        hard_zero_fraction=0.01,
        random_state=None,
    ):
        self.rng = check_random_state(random_state)
        self.hard_zero_fraction = hard_zero_fraction
        super().__init__(y_train, event_of_interest)

    def draw(self):
        # Sample time horizons uniformly on the observed time range:
        duration = self.duration_train
        n_samples = duration.shape[0]

        # Sample from t_min=0 event if never observed in the training set
        # because we want to make sure that the model learns to predict a 0
        # incidence at t=0.
        t_min = 0.0
        t_max = duration.max()
        times = self.rng.uniform(t_min, t_max, n_samples)

        # Add some some hard zeros to make sure that the model learns to
        # predict 0 incidence at t=0.
        #
        # TODO: theoretically or empirically study what kind of bias
        # oversampling exact zeros introduces, w.r.t. the stochastically
        # time-integrated Brier score objective.
        n_hard_zeros = max(int(self.hard_zero_fraction * n_samples), 1)
        hard_zero_indices = self.rng.choice(n_samples, n_hard_zeros, replace=False)
        times[hard_zero_indices] = 0.0

        if self.event_of_interest == "any":
            # Collapse all event types together.
            event = self.any_event_train
        else:
            event = self.event_train

        y_binary, sample_weight = self._weighted_binary_targets(
            event,
            duration,
            times,
            ipcw_y_duration=self.ipcw_train,
        )
        return times.reshape(-1, 1), y_binary, sample_weight


class GradientBoostingIncidence(BaseEstimator, ClassifierMixin):
    """Cause-specific Cumulative Incidence Function (CIF) with GBDT.

    This internally relies on the histogram-based gradient boosting classifier
    or regressor implementation of scikit-learn.

    Estimate a cause-specific CIF by iteratively minimizing a stochastic time
    integrated proper scoring rule (Brier score or binary cross-entropy) for
    the kth cause of failure from [Kretowska2018]_.

    Under the hood, this class uses randomly sampled reference time horizons
    concatenated as an extra input column to the underlying HGB binary
    classification model. At boosting iteration, a new tree is trained on a
    copy of the original feature matrix X augmented with a new independent sample
    of time horizons.

    One can obtain the survival probabilities for any event by summing all
    cause-specific CIF curves and computing 1 - "sum of CIF curves".

    Parameters
    ----------
    event_of_interest : int or "any", default="any"
        The event to compute the CIF for. When passed as an integer, it should
        match one of the values observed in `y_train["event"]`. Note: 0 always
        represents censoring and cannot be used as a valid event of interest.

        "any" means that all events are collapsed together and the resulting
        model can be used for any event survival analysis: the any event
        survival function can be estimated as the complement of the any event
        cumulative incidence function.

        In single event settings, "any" and 1 are equivalent.

    loss : {'ibs', 'inll'}, default='ibs'
        The objective of the model. In practise, both objective yields
        comparable results.

        - 'ibs' : integrated brier score. Use a `HistGradientBoostedRegressor`
          with the 'squared_error' loss. As we have no guarantee that the
          regression yields a survival function belonging to [0, 1], we clip
          the probabilities to this range.
        - 'inll' : integrated negative log likelihood (also known as integrated
          binary cross-entropy). Use a `HistGradientBoostedClassifier` with
          'log_loss'.

    time_horizon : float or int, default=None
        A specific time horizon `t_horizon` to treat the model as a
        probabilistic classifier to estimate `E[T < t_horizon, E = k|X]` where
        `T` is a random variable representing the (uncensored) event and `E`
        a random categorical variable representing the (uncensored) event type.

        When specified, the `predict_proba` method returns an estimate of
        `E[T < t_horizon, E = k|X]` for each provided realisation of `X`.

    monotonic_incidence : str or False, default=False
        Whether to constrain the CIF to be monotonic with respect to time.
        If left to `False`, the CIF is not constrained to be monotonic and
        can randomly oscillate around the true CIF.

        If set to 'at_training_time', the CIF is constrained to be monotonically
        increasing at training time.

        TODO: implement 'at_prediction_time' option with isotonic regression.

        Note: constraining the CIF to be monotonic can lead to a biased estimate
        of the CIF: the CIF is typically overestimated for the longest time
        horizons, especially for large number of boosting iterations and deep
        trees.

    The remaining hyper-parameters match those of the underlying
    `HistGradientBoostedClassifier` or `HistGradientBoostedRegressor` models.

    References
    ----------

    .. [Graf1999] E. Graf, C. Schmoor, W. Sauerbrei, M. Schumacher, "Assessment
       and comparison of prognostic classification schemes for survival data",
       1999

    .. [Kretowska2018] M. Kretowska, "Tree-based models for survival data with
       competing risks", 2018

    .. [Gerds2006] T. Gerds and M. Schumacher, "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times", 2006

    .. [Edwards2016] J. Edwards, L. Hester, M. Gokhale, C. Lesko,
       "Methodologic Issues When Estimating Risks in Pharmacoepidemiology.",
       2016, doi:10.1007/s40471-016-0089-1
    """

    def __init__(
        # TODO: run a grid search on a few datasets to find good defaults.
        self,
        event_of_interest="any",
        loss="ibs",
        monotonic_incidence=False,
        hard_zero_fraction=0.1,
        # TODO: implement convergence criterion and use max_iter instead of
        # n_iter.
        n_iter=100,
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        show_progressbar=True,
        n_time_grid_steps=100,
        time_horizon=None,
        random_state=None,
    ):
        self.event_of_interest = event_of_interest
        self.loss = loss
        self.monotonic_incidence = monotonic_incidence
        self.hard_zero_fraction = hard_zero_fraction
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.show_progressbar = show_progressbar
        self.n_time_grid_steps = n_time_grid_steps
        self.time_horizon = time_horizon
        self.random_state = random_state

    def fit(self, X, y, times=None):
        X = check_array(X)
        event, duration = check_y_survival(y)

        self.event_ids_ = np.unique(event)

        # The time horizon is concatenated as an additional input feature
        # before the features of X and we constrain the prediction function
        # (that estimates the CIF) to monotically increase with the time
        # horizon feature.
        if self.monotonic_incidence == "at_training_time":
            monotonic_cst = np.zeros(X.shape[1] + 1)
            monotonic_cst[0] = 1
        elif self.monotonic_incidence is not False:
            raise ValueError(
                f"Invalid value for monotonic_incidence: {self.monotonic_incidence}."
                " Expected either 'at_training_time' or False."
            )
        else:
            monotonic_cst = None

        self.estimator_ = self._build_base_estimator(monotonic_cst)

        # Compute the default time grid used at prediction time.
        any_event_mask = event > 0
        observed_times = duration[any_event_mask]

        if times is None:
            if observed_times.shape[0] > self.n_time_grid_steps:
                self.time_grid_ = np.quantile(
                    observed_times, np.linspace(0, 1, num=self.n_time_grid_steps)
                )
            else:
                self.time_grid_ = observed_times.copy()
                self.time_grid_.sort()
        else:
            # XXX: do we really want to allow to pass this at training time if
            # we already allow to pass it at prediction time?
            self.time_grid_ = times.copy()
            self.time_grid_.sort()

        self.weighted_targets_ = WeightedBinaryTargetSampler(
            y,
            event_of_interest=self.event_of_interest,
            hard_zero_fraction=self.hard_zero_fraction,
            random_state=self.random_state,
        )

        iterator = range(self.n_iter)
        if self.show_progressbar:
            iterator = tqdm(iterator)

        for _ in iterator:
            (
                sampled_times,
                y_binary,
                sample_weight,
            ) = self.weighted_targets_.draw()
            X_with_time = np.hstack([sampled_times, X])
            self.estimator_.max_iter += 1
            self.estimator_.fit(X_with_time, y_binary, sample_weight=sample_weight)

            # XXX: implement verbose logging with a version of IBS that
            # can handle competing risks.

        # To be use at a fixed horizon classifier when setting time_horizon.
        if self.event_of_interest == "any":
            self.classes_ = np.array(["no_event", "any_event"])
        else:
            self.classes_ = np.array(
                ["other_or_no_event", f"event_{self.event_of_interest}"]
            )
        return self

    def predict_proba(self, X, time_horizon=None):
        """Estimate the probability of incidence for a specific time horizon.

        See the docstring for the `time_horizon` parameter for more details.

        Returns a 2d array with shape (X.shape[0], 2). The second column holds
        the cumulative incidence probability and the first column its
        complement.

        When `event_of_interest == "any"` the second column therefore holds the
        sum all individual events cumulative incidece and the first column
        holds the probability of remaining event free at `time_horizon`, that
        is, the survival probability.

        When `event_of_interest != "any"`, the values in the first column do
        not have an intuitive meaning.
        """
        if time_horizon is None:
            if self.time_horizon is None:
                raise ValueError(
                    "The time_horizon parameter is required to use "
                    f"{self.__class__.__name__} as a classifier. "
                    "This parameter can either be passed as constructor "
                    "or method parameter."
                )
            else:
                time_horizon = self.time_horizon

        times = np.asarray([time_horizon])
        cif = self.predict_cumulative_incidence(X, times=times)

        # Reshape to be consistent with the expected shape returned by
        # the predict_proba method of scikit-learn binary classifiers.
        cif = cif.reshape(-1, 1)
        return np.hstack([1 - cif, cif])

    def predict_cumulative_incidence(self, X, times=None):
        if times is None:
            times = self.time_grid_

        if self.show_progressbar:
            times = tqdm(times)

        predictions_at_all_times = []
        for t in times:
            t = np.full((X.shape[0], 1), fill_value=t)
            X_with_time = np.hstack([t, X])
            if self.loss == "ibs":
                predictions_at_t = self.estimator_.predict(X_with_time)
            else:
                predictions_at_t = self.estimator_.predict_proba(X_with_time)[:, 1]
            predictions_at_all_times.append(predictions_at_t)

        predicted_curves = np.vstack(predictions_at_all_times).T

        if self.loss == "ibs":
            # HistGradientBoostingRegressor does not guarantee that the
            # predictions are in [0, 1] (no identity link function when using
            # the squared_error loss).
            predicted_curves = np.clip(predicted_curves, 0, 1)

        return predicted_curves

    def predict_survival_function(self, X, times=None):
        """Compute the event specific survival function.

        Warning: this metric only makes sense when y_train["event"] is binary
        (single event) or when setting event_of_interest='any'.
        """
        if (self.event_ids_ > 0).sum() > 1 and self.event_of_interest != "any":
            warnings.warn(
                "Values returned by predict_survival_function only make "
                "sense when the model is trained with a binary event "
                "indicator or when setting event_of_interest='any'. "
                "Instead this model was fit on data with event ids "
                f"{self.event_ids_.tolist()} and with "
                f"event_of_interest={self.event_of_interest}."
            )
        return 1 - self.predict_cumulative_incidence(X, times=times)

    def predict_quantile(self, X, quantile=0.5, times=None):
        """Estimate the conditional median (or other quantile) time to event.

        Note: this can return np.inf values when the estimated CIF does not
        reach the `quantile` value at the maximum time horizon observed on
        the training set.
        """
        if times is None:
            times = self.time_grid_
        cif_curves = self.predict_cumulative_incidence(X, times=times)
        quantile_idx = np.apply_along_axis(
            lambda a: a.searchsorted(quantile, side="right"), 1, cif_curves
        )
        inf_mask = quantile_idx == cif_curves.shape[1]
        # Change quantile_idx to avoid out-of-bound index in the subsequent
        # line.
        quantile_idx[inf_mask] = cif_curves.shape[1] - 1
        results = times[quantile_idx]
        # Mark out-of-index results as np.inf
        results[inf_mask] = np.inf
        return results

    def _build_base_estimator(self, monotonic_cst):
        if self.loss == "ibs":
            return HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=1,
                warm_start=True,
                monotonic_cst=monotonic_cst,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
            )
        elif self.loss == "inll":
            return HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=1,
                warm_start=True,
                monotonic_cst=monotonic_cst,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        else:
            raise ValueError(
                f"Parameter 'loss' must be either 'ibs' or 'inll', got {self.loss}."
            )

    def score(self, X, y):
        """Return the negative time-integrated Brier score (IBS) or INLL.

        This returns the negative of a proper scoring rule, so that the higher
        the value, the better the model to be consistent with the scoring
        convention of scikit-learn to make it possible to use this class with
        scikit-learn model selection utilities such as GridSearchCV and
        RandomizedSearchCV.

        The `loss` parameter passed to the constructor determines whether the
        negative IBS or negative INLL is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : dict with keys "event" and "duration"
            The target values. "event" is a boolean array of shape (n_samples,)
            indicating whether the event was observed or not. "duration" is a
            float array of shape (n_samples,) indicating the time of the event
            or the time of censoring.

        Returns
        -------
        score : float
            The time-integrated Brier score (IBS) or INLL.
        """
        predicted_curves = self.predict_cumulative_incidence(X)
        if self.weighted_targets_.event_of_interest != self.event_of_interest:
            raise ValueError(
                "The event_of_interest parameter passed to the score method "
                f"({self.event_of_interest}) does not match the one used at "
                f"training time ({self.weighted_targets_.event_of_interest})."
            )

        if self.loss == "ibs":
            return -self.weighted_targets_.integrated_brier_score_incidence(
                y,
                predicted_curves,
                times=self.time_grid_,
            )
        elif self.loss == "inll":
            raise NotImplementedError("implement me!")
        else:
            raise ValueError(
                f"Parameter 'loss' must be either 'ibs' or 'inll', got {self.loss}."
            )
