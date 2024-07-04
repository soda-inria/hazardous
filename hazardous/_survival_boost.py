from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_array, check_random_state
from tqdm import tqdm

from ._ipcw import AlternatingCensoringEstimator, KaplanMeierIPCW
from .metrics._brier_score import (
    IncidenceScoreComputer,
    integrated_brier_score_incidence,
    integrated_brier_score_survival,
)
from .utils import check_y_survival


class WeightedMultiClassTargetSampler(IncidenceScoreComputer):
    """Weighted targets for censoring-adjusted incidence estimation.

    This class samples time horizons and computes the corresponding weighted
    targets for the training of the multi-class incidence estimator. The
    weighted targets are the incidence of each event type at the sampled time
    horizons. The weights are the inverse of the censoring survival function at
    the sampled time horizons.

    Parameters
    ----------
    hard_zero_fraction : float, default=0.1
        The fraction of observations that are assigned a time horizon set to exact
        zeros when doing one epoch of fitting. Increasing this value helps the model
        learn to predict 0 incidence at `t=0` at the cost of reducing the effective
        sample size for the non-zero time horizons.

    ipcw_est : object, default=None
        The estimator used to estimate the Inverse Probability of Censoring Weighting
        (IPCW). If `None`, an instance of `KaplanMeierIPCW` is used.

    n_iter_before_feedback : int, default=20
        The number of iterations used to fit incrementally `ipcw_est`.

    random_state : int, RandomState instance or None, default=None
        Control the randomness of the time horizon sampler.

    Attributes
    ----------
    inv_any_survival_train : ndarray of shape (n_samples,)
        The IPCW for each sample at each time horizon before fitting the IPCW estimator.

    ipcw_train : ndarray of shape (n_samples,)
        The IPCW for each sample at each time horizon after fitting the IPCW estimator.
    """

    def __init__(
        self,
        y_train,
        hard_zero_fraction=0.01,
        random_state=None,
        ipcw_est=None,
        n_iter_before_feedback=20,
    ):
        self.rng = check_random_state(random_state)
        self.hard_zero_fraction = hard_zero_fraction
        self.n_iter_before_feedback = n_iter_before_feedback
        super().__init__(
            y_train,
            event_of_interest="any",
            ipcw_est=ipcw_est,
        )
        # Precompute the censoring probabilities at the time of the events on the
        # training set:
        self.ipcw_train = self.ipcw_est.compute_ipcw_at(self.duration_train)

    def draw(self, ipcw_training=False, X=None):
        # Sample time horizons uniformly on the observed time range:
        observation_durations = self.duration_train
        n_samples = observation_durations.shape[0]

        # Sample from t_min=0 event if never observed in the training set
        # because we want to make sure that the model learns to predict a 0
        # incidence at t=0.
        t_min = 0.0
        t_max = observation_durations.max()
        sampled_time_horizons = self.rng.uniform(t_min, t_max, n_samples)

        # Add some hard zeros to make sure that the model learns to
        # predict 0 incidence at t=0.
        n_hard_zeros = max(int(self.hard_zero_fraction * n_samples), 1)
        hard_zero_indices = self.rng.choice(n_samples, n_hard_zeros, replace=False)
        sampled_time_horizons[hard_zero_indices] = 0.0

        if ipcw_training:
            # During the training of the ICPW, we estimate G(t) = P(C > t)
            # 1 / S(t) = 1 / P(T^* > t) as sample weight. t is an arbitrary
            # time horizon.
            # Since 1 = P(C <= t) + P(C > t), our training target is the censoring
            # incidence, whose value is:
            # * 1 when the observation was censored: no event happened during
            #   the observation period and the observation duration was lower
            #   than the sampled time horizon;
            # * 0 for a censored observation with a duration larger than the
            #   sampled time horizon;
            # * 0 when an event has happened before the sampled time horizon.
            #   The sample weight is zero in that case.

            if not hasattr(self, "inv_any_survival_train"):
                self.inv_any_survival_train = self.ipcw_est.compute_ipcw_at(
                    self.duration_train, ipcw_training=True, X=X
                )

            censored_observations = self.any_event_train == 0
            y_targets, sample_weight = self._weighted_binary_targets(
                censored_observations,
                observation_durations,
                sampled_time_horizons,
                ipcw_y_duration=self.inv_any_survival_train,
                ipcw_training=True,
                X=X,
            )
        else:
            # During the training of the multi incidence estimator, we estimate
            # P(T^* <= t & Delta = k) using 1 / P(C > t) as sample weight.
            # t is an arbitrary time horizon.
            # Since 1 = P(T^* <= t) + P(T^* > t), our training target is the
            # multi event incidence, whose value is:
            # * k when the event k has happened first and before the time
            #   horizon;
            # * 0 when no event has happened at the sampled time horizon;
            # * 0 when the observation was censored with a duration smaller
            #   than the sampled time horizon. The sample weight is zero in
            #   that case.
            y_binary, sample_weight = self._weighted_binary_targets(
                self.any_event_train,
                observation_durations,
                sampled_time_horizons,
                ipcw_y_duration=self.ipcw_train,
                ipcw_training=False,
                X=X,
            )
            y_targets = y_binary * self.event_train

        return sampled_time_horizons.reshape(-1, 1), y_targets, sample_weight

    def fit(self, X):
        self.inv_any_survival_train = self.ipcw_est.compute_ipcw_at(
            self.duration_train, ipcw_training=True, X=X
        )

        for _ in range(self.n_iter_before_feedback):
            sampled_time_horizons, y_targets, sample_weight = self.draw(
                ipcw_training=True,
                X=X,
            )
            self.ipcw_est.fit_censoring_estimator(
                X,
                y_targets,
                times=sampled_time_horizons,
                sample_weight=sample_weight,
            )

        self.ipcw_train = self.ipcw_est.compute_ipcw_at(
            self.duration_train,
            ipcw_training=False,
            X=X,
        )


class SurvivalBoost(BaseEstimator, ClassifierMixin):
    r"""Cause-specific Cumulative Incidence Function (CIF) with GBDT [1]_.

    This model estimates the cause-specific Cumulative Incidence Function (CIF)
    of each event of interest as well the surival funciton to any event using a
    Gradient Boosting Decision Tree (GBDT) classifier. The CIF is the
    probability of observing an event of a specific type before a given time.

    The models handles survival analysis and competing risks data.

    The cumulative incidence function (CIF) for each event type :math:`k` at
    each time horizon `t` is defined as:

    .. math::

        \hat{F}_k(t; x_i) \approx F_k(t; x_i) = \mathbb{P}(T \leq t, E=k | X=x_i)

    where :math:`T` is a random variable for the uncensored time to first event
    and :math:`E` is a random variable over the :math:`[1, K]` domain for the
    (uncensored) event type, and :math:`x_i` is the feature vector of the
    :math:`i`-th observation.

    The (any event) Survival Function can be defined as:

    .. math::

        S(t; x_i) = \mathbb{P}(T > t) = 1 - \mathbb{P}(T \leq t | X=x_i)
        = 1 - \sum_{k=1}^K \mathbb{P}(T \leq t, E=k | X=x_i)
        = 1 - \sum_{k=1}^K F_k(t; x_i)

    Under the hood, this class randomly samples reference time horizons
    concatenated as an extra input column to train an underlying HGB
    classifier. At each boosting iteration, a new tree is trained on a copy of
    the original feature matrix X augmented with a new independent sample of
    time horizons. The number of time horizons sampled at each iteration is
    controlled by the `n_horizons_per_observation` parameter.

    To predict the survival function and the CIF, the model uses an alternating
    optimization. The censoring-adjusted incidence estimator is trained with a
    fixed number of iterations before the feedback loop is triggered. The
    feedback loop is triggered every `n_iter_before_feedback` iterations. The
    feedback loop updates the censoring-adjusted incidence estimator with the
    current model predictions.

    Parameters
    ----------
    hard_zero_fraction : float, default=0.1
        The fraction of observations that are assigned a time horizon set to exact
        zeros when doing one epoch of fitting. Increasing this value helps the model
        learn to predict 0 incidence at `t=0` at the cost of reducing the effective
        sample size for the non-zero time horizons.

    n_iter : int, default=100
        The number of boosting iterations.

    learning_rate : float, default=0.05
        The learning rate, also known as shrinkage. This is used as a multiplicative
        factor for the leaves values. Use 1 for no shrinkage.

    n_iter : int, default=100
        The number of iterations of the boosting process.

    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater than 1. If
        None, there is no maximum limit.

    max_depth : int, default=None
        The maximum depth of each tree. The depth of a tree is the number of edges to go
        from the root to the deepest leaf. Depth isn't constrained by default.

    min_samples_leaf : int, default=50
        The minimum number of samples per leaf.

    show_progressbar : bool, default=True
        Whether to show a progress bar during the training process.

    n_time_grid_steps : int, default=100
        The number of time horizons to sample uniformly between the minimum and maximum
        observed event times. Note that the generated grid `time_grid_` can be
        overridden in the method `predict_cumulative_incidence` and
        `predict_survival_function` by setting the parameter `times`.

    time_horizon : int or float, default=None
        The time horizon at which to estimate the probabilities. If `None`, the
        `time_horizon` should be specified when calling the method `predict_proba`.

    ipcw_strategy : {"alternating", "kaplan-meier"}, default="alternating"
        The method used to estimate the Inverse Probability of Censoring
        Weighting (IPCW).

        If "alternating", the two instances of gradient boosting are trained
        alternatively every `n_iter_before_feedback` iterations: one for the
        CIF + any event survival function and the other for the censoring
        distribution. This makes it possible to estimate IPCW conditionally on
        the covariates without assuming independence between censoring and
        covariates.

        If "kaplan-meier", the censoring estimator is trained using the Kaplan-Meier
        estimator. This estimator is trained only once at the beginning of the
        training process. This estimator is very fast but assumes that the
        censoring is independent of the covariates.

    n_iter_before_feedback : int, default=20
        The number of iterations at which we alternate to fit the Inverse Probability
        of Censoring Weighting (IPCW) estimator before feeding back the weights to the
        incidence estimator.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the uniform time sampler.

    n_horizons_per_observation : int, default=3
        The number of time horizons to sample for each individual in the
        training at each stochastic boosting iteration (epoch).

    Attributes
    ----------
    estimator_ : HistGradientBoostingClassifier
        The base estimator used to fit the CIF and survival function.

    classes_ : ndarray of shape (n_classes,)
        The events seen during training.

    time_grid_ : ndarray of shape (n_time_grid_steps,)
        The time horizons used to predict the survival function and the CIF.

    weighted_targets_ : WeightedMultiClassTargetSampler
        The weighted targets used to train the model.

    References
    ----------
    .. [1]  J. Alberge, V. Maladière, O. Grisel, J. Abécassis, G. Varoquaux,
            "Teaching Models To Survive: Proper Scoring Rule and Stochastic Optimization
            with Competing Risks", 2024.
            https://arxiv.org/pdf/2406.14085

    Examples
    --------
    >>> from hazardous.data import make_synthetic_competing_weibull
    >>> from sklearn.model_selection import train_test_split
    >>> from hazardous import SurvivalBoost
    >>> X, y = make_synthetic_competing_weibull(return_X_y=True, random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> survival_booster = SurvivalBoost(
    ...     n_iter=3, show_progressbar=False, random_state=0
    ... ).fit(X_train, y_train)
    >>> survival_pred = survival_booster.predict_survival_function(X_test)
    """

    def __init__(
        # TODO: run a grid search on a few datasets to find good defaults.
        self,
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
        n_iter_before_feedback=20,
        ipcw_strategy="alternating",
        random_state=None,
        n_horizons_per_observation=3,
    ):
        self.hard_zero_fraction = hard_zero_fraction
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.show_progressbar = show_progressbar
        self.n_time_grid_steps = n_time_grid_steps
        self.time_horizon = time_horizon
        self.n_iter_before_feedback = n_iter_before_feedback
        self.ipcw_strategy = ipcw_strategy
        self.random_state = random_state
        self.n_horizons_per_observation = n_horizons_per_observation

    def fit(self, X, y, times=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : dict, {array-like, dataframe} of shape (n_samples, 2)
            The target values. If a dictionary, it must have keys "event" and
            "duration". If an record array, it must have a dtype with two fields
            named "event" and "duration". If a dataframe, it must have columns
            named "event" and "duration". "event" is an integer array of shape
            (n_samples,) indicating which event was observed (and 0 means that
            the sample was censored). "duration" is a float array of shape
            (n_samples,) indicating the time of the first event or the time of
            censoring.

        times : array-like of shape (n_times,), default=None
            The time horizons used to predict the survival function and the CIF.
            If None, the default time grid is computed from the observed event
            times in the training data.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        X = check_array(X, force_all_finite="allow-nan")
        event, duration = check_y_survival(y)

        # Add 0 as a special event id for the survival function.
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))

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

        if self.ipcw_strategy == "alternating":
            ipcw_est = AlternatingCensoringEstimator(incidence_est=self.estimator_)
        elif self.ipcw_strategy == "kaplan-meier":
            ipcw_est = KaplanMeierIPCW()
        else:
            raise ValueError(
                f"Invalid parameter value: ipcw_strategy={self.ipcw_strategy!r}. "
                "Valid values are 'alternating' and 'kaplan-meier'."
            )

        self.weighted_targets_ = WeightedMultiClassTargetSampler(
            y,
            hard_zero_fraction=self.hard_zero_fraction,
            random_state=self.random_state,
            ipcw_est=ipcw_est,
            n_iter_before_feedback=self.n_iter_before_feedback,
        )

        iterator = range(self.n_iter)
        if self.show_progressbar:
            iterator = tqdm(iterator)

        for idx_iter in iterator:
            X_with_time = np.empty((0, X.shape[1] + 1))
            y_targets = np.empty((0,))
            sample_weight = np.empty((0,))
            for _ in range(self.n_horizons_per_observation):
                (
                    sampled_times_,
                    y_targets_,
                    sample_weight_,
                ) = self.weighted_targets_.draw(X=X, ipcw_training=False)

                X_with_time_ = np.hstack([sampled_times_, X])
                X_with_time = np.vstack([X_with_time, X_with_time_])
                y_targets = np.hstack([y_targets, y_targets_])
                sample_weight = np.hstack([sample_weight, sample_weight_])

            self.estimator_.max_iter += 1
            self.estimator_.fit(X_with_time, y_targets, sample_weight=sample_weight)

            if not np.array_equal(self.estimator_.classes_, self.event_ids_):
                raise ValueError(
                    "The time-horizon resampling of the data has caused some events "
                    f"to be unobserved in the training data at iteration {idx_iter}. "
                    "Consider lowering the value of hard_zero_fraction (currently set "
                    f"to {self.hard_zero_fraction})."
                )

            if (idx_iter % self.n_iter_before_feedback == 0) and isinstance(
                ipcw_est, AlternatingCensoringEstimator
            ):
                self.weighted_targets_.fit(X)

            # XXX: implement verbose logging with a version of IBS that
            # can handle competing risks.

        # To be use at a fixed horizon classifier when setting time_horizon.
        events_names = [f"event_{i}" for i in range(1, len(self.event_ids_))]
        self.classes_ = np.array(["no_event"] + events_names)
        return self

    def predict_proba(self, X, time_horizon=None):
        """Estimate the probability of all incidences for a specific time horizon.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        time_horizon : int or float, default=None
            The time horizon at which to estimate the probabilities. If `None`, the
            `time_horizon` passed at the constructor is used.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_events + 1)
            The estimated probabilities at the given time horizon. The column
            indexed 0 stores the estimated probabilities of staying event-free at
            the requested time horizon for each observation described by the matching
            row of X. The remaining columns store the estimated cumulated incidence
            (or probability) for each event.
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

        if not isinstance(time_horizon, Real):
            raise TypeError(
                "The time_horizon parameter must be a real number. Use "
                "predict_cumulative_incidence instead of predict_proba if you want "
                "to predict at several time horizons."
            )
        times = np.asarray([time_horizon])
        return self.predict_cumulative_incidence(X, times=times).squeeze()

    def predict_cumulative_incidence(self, X, times=None):
        """Estimate conditional cumulative incidence function for each event type.

        Please refer to the docstring of the class for the definitions of the
        conditional survival function and the event-specific cumulative
        incidence functions estimated by this method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature vectors for each observation for which to estimate the
            survival function.

        times : array-like, default=None
            The time horizons at which to estimate the probabilities. If `None`, it uses
            the grid generated during `fit` based on the parameter `n_time_grid_steps`.

        Returns
        -------
        predicted_curves : ndarray of shape (n_samples, n_events + 1, n_times)
            The estimated probabilities at different time horizons. The values at event
            index 0 are the estimated probabilities of staying event-free at
            the requested time horizons for each observation described by the matching
            row of X. The remaining event indices correspond to the estimated cumulated
            incidence (or probability) for each event type.
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

        predicted_curves = np.array(predictions_at_all_times)
        # roll axis to get a shape (n_samples, n_events + 1, n_times)
        return np.transpose(predicted_curves, axes=(1, 2, 0))

    def predict_survival_function(self, X, times=None):
        """Estimate the conditional any-event survival function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature vectors for each observation for which to estimate the
            survival function.

        times : array-like, default=None
            The time horizons at which to estimate the probabilities. If `None`, it uses
            the grid generated during `fit` based on the parameter `n_time_grid_steps`.

        Returns
        -------
        predicted_curves : ndarray of shape (n_samples, n_times)
            The estimated probabilities of staying event-free at different time
            horizons.
        """
        return self.predict_cumulative_incidence(X, times=times)[:, 0, :]

    def _build_base_estimator(self):
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=1,
            warm_start=True,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

    def score(self, X, y):
        """Return the mean of IBS for each event of interest and survival.

        This returns the negative of the mean of the Integrated Brier Score
        (IBS) (a proper scoring rule) of each competing event as well as the IBS
        of the survival to any event. So, the higher the value, the better the
        model to be consistent with the scoring convention of scikit-learn to
        make it possible to use this class with scikit-learn model selection
        utilities such as GridSearchCV and RandomizedSearchCV.

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
            The negative of time-integrated Brier score (IBS) or INLL.

        TODO: implement time integrated NLL and use as the default for the
        .score method to match the objective function used at fit time.
        """
        predicted_curves = self.predict_cumulative_incidence(X)
        ibs_events = []
        for event_idx in self.event_ids_:
            predicted_curves_for_event = predicted_curves[:, event_idx]
            if event_idx == 0:
                ibs_event = integrated_brier_score_survival(
                    y_train=self.weighted_targets_.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.time_grid_,
                )
            else:
                ibs_event = integrated_brier_score_incidence(
                    y_train=self.weighted_targets_.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.time_grid_,
                    event_of_interest=event_idx,
                )
            ibs_events.append(ibs_event)
        return -np.mean(ibs_events)
