import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_array, check_random_state
from tqdm import tqdm

from ._comp_risks_loss import MultinomialBinaryLoss
from .metrics._brier_score import IncidenceScoreComputer
from .utils import check_y_survival


class WeightedMultiClassTargetSampler(IncidenceScoreComputer):
    """Weighted targets for censoring-adjusted incidence estimation.

    Parameters
    ----------
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the uniform time sampler
    """

    def __init__(
        self,
        y_train,
        hard_zero_fraction=0.01,
        random_state=None,
    ):
        self.rng = check_random_state(random_state)
        self.hard_zero_fraction = hard_zero_fraction
        super().__init__(y_train, event_of_interest="any")

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

        n_hard_zeros = max(int(self.hard_zero_fraction * n_samples), 1)
        hard_zero_indices = self.rng.choice(n_samples, n_hard_zeros, replace=False)
        times[hard_zero_indices] = 0.0

        y_binary, sample_weight = self._weighted_binary_targets(
            self.any_event_train,
            duration,
            times,
            ipcw_y_duration=self.ipcw_train,
        )
        y_before_horizon = y_binary * self.event_train

        return times.reshape(-1, 1), y_before_horizon, sample_weight


class GBMultiIncidence(BaseEstimator, ClassifierMixin):
    """Cause-specific Cumulative Incidence Function (CIF) with GBDT.

    This model returns the cause-specific CIFs for each event type as well as
    the survival function.

    Cumulative Incidence Function for each event type :math:`k`:

    .. math::

        \hat{F}_k(t) \approx \mathbb{P}(T \leq t, E=k)
    where :math:`T` is a random variable for the uncensored time to first event
    and :math:`E` is a random variable over the :math:`[1, K]` domain for the
    uncensored event type.

    .. math::

        S(t) = \mathbb{P}(T > t) = 1 - \mathbb{P}(T \leq t)
        = 1 - \sum_{k=1}^K \mathbb{P}(T \leq t, E=k)
        \approx 1 - \sum_{k=1}^K \hat{F}_k(t)


    Under the hood, this class uses randomly sampled reference time horizons
    concatenated as an extra input column to the underlying HGB classifier.
    At boosting iteration, a new tree is trained on a
    copy of the original feature matrix X augmented with a new independent sample
    of time horizons.

    """

    def __init__(
        # TODO: run a grid search on a few datasets to find good defaults.
        self,
        loss="inll",
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

        # Add 0 as a special event id for the survival function.
        self.event_ids_ = np.array(list(set([0]) | set(event)))
        # The time horizon is concatenated as an additional input feature
        # before the features of X

        self.estimator_ = self._build_base_estimator()

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

        self.weighted_targets_ = WeightedMultiClassTargetSampler(
            y,
            hard_zero_fraction=self.hard_zero_fraction,
            random_state=self.random_state,
        )

        iterator = range(self.n_iter)
        if self.show_progressbar:
            iterator = tqdm(iterator)

        for _ in iterator:
            (
                sampled_times,
                y_targets,
                sample_weight,
            ) = self.weighted_targets_.draw()

            X_with_time = np.hstack([sampled_times, X])
            self.estimator_.max_iter += 1
            self.estimator_.fit(X_with_time, y_targets, sample_weight=sample_weight)

            # XXX: implement verbose logging with a version of IBS that
            # can handle competing risks.

        # To be use at a fixed horizon classifier when setting time_horizon.
        events_names = [f"event_{i}" for i in range(1, len(self.event_ids_))]
        self.classes_ = np.array(["no_event"] + events_names)
        return self

    def predict_proba(self, X, time_horizon=None):
        """Estimate the probability of all incidences for a specific time horizon.

        See the docstring for the `time_horizon` parameter for more details.

        Returns a (n_events + 1)d array with shape (X.shape[0], n_events + 1).
        The first column holds the survival probability to any event and others the
        incicence probabilities for each event.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        time_horizon : float or list, default=None
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

        return cif

    def predict_cumulative_incidence(self, X, times=None):
        """Estimate the cumulative incidence function for all events.

        .. math::

        \sum_{k=1}^K \mathbb{P}(T^* \leq t \cap \Delta = k) + \mathbb{P}(T^* > t) = 1

        Returns
        -------
        predicted_curves : array-like of shape (n_samples, n_events + 1)
            The first column holds the survival probability to any event and others the
            incicence probabilities for each event.
        """
        if times is None:
            times = self.time_grid_

        if self.show_progressbar:
            times = tqdm(times)

        predictions_at_all_times = []

        for t in times:
            t = np.full((X.shape[0], 1), fill_value=t)
            X_with_time = np.hstack([t, X])
            predictions_at_t = self.estimator_.predict_proba(X_with_time)
            predictions_at_all_times.append(predictions_at_t)

        predicted_curves = np.array(predictions_at_all_times).swapaxes(2, 0)

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

    def _build_base_estimator(self):
        if self.loss == "inll":
            return HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=1,
                warm_start=True,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        elif self.loss == "competing_risks":
            loss = MultinomialBinaryLoss(n_classes=len(self.event_ids_))
            # class_weight = {0: 0, **{event: 1 for event in self.event_ids_[1:]}}
            return HistGradientBoostingClassifier(
                loss=loss,
                class_weight=None,
                max_iter=1,
                warm_start=True,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        else:
            raise ValueError(
                "Parameter 'loss' must be 'inll' or 'competing_risks', got"
                f" {self.loss}."
            )

    def score(self, X, y):
        """Return INLL or competing_risks loss (proper scoring rule).

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

        #TODO: implement time integrated NLL.
        """

        if self.loss == "inll":
            return -self.estimator_.score(X, y)
        else:
            raise ValueError(f"Parameter 'loss' must be 'inll', got {self.loss}.")
