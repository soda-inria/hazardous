import warnings
import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

from .metrics.brier_score import BrierScoreComputer
from .survival_mixin import SurvivalMixin
from .utils import _check_y_survival


class BrierScoreSampler(BrierScoreComputer):
    def draw(self):
        # Sample time horizons uniformly on the observed time range:
        min_times = self.y_train["duration"].min()
        max_times = self.y_train["duration"].max()
        times = self.rng.uniform(min_times, max_times, self.y_train.shape[0])

        if self.event_of_interest == "any":
            # Collapse all event types together.
            y = self.y_train_any_event
        else:
            y = self.y_train

        y_binary, sample_weights = self._ibs_components(
            y,
            times,
            ipcw_y=self.ipcw_y_train,
        )
        return times.reshape(-1, 1), y_binary, sample_weights


class GradientBoostedCIF(BaseEstimator, SurvivalMixin, ClassifierMixin):
    """GBDT estimator for cause-specific Cumulative Incidence Function (CIF).

    This internally relies on the histogram-based gradient boosting classifier
    or regressor implementation of scikit-learn.

    Estimate a cause-specific CIF by minimizing the Brier Score for the kth
    cause of failure from _[1] for randomly sampled reference time horizons
    concatenated as extra inputs to the underlying HGB binary classification
    model.

    One can obtain the survival probabilities for any event by summing all
    cause-specific CIF curves and computing 1 - "sum of CIF curves".

    Parameters
    ----------
    event_of_interest : int or "any" default="any"
        The event to compute the CIF for. When passed as an integer, it should
        match one of the values observed in `y_train["event"]`. Note: 0 always
        represents censoring and cannot be used as a valid event of interest.
    
        "any" means that all events are collapsed together and the resulting
        model can be used for any event survival analysis: the any
        event survival function can be estimated as the complement of the
        any event cumulative incidence function.

    objective : {'ibs', 'inll'}, default='ibs'
        The objective of the model. In practise, both objective yields
        comparable results.

        - 'ibs' : integrated brier score. Use a `HistGradientBoostedRegressor`
          with the 'squared_error' loss. As we have no guarantee that the regression
          yields a survival function belonging to [0, 1], we clip the probabilities
          to this range.
        - 'inll' : integrated negative log likelihood. Use a
          `HistGradientBoostedClassifier` with 'log_loss' loss.

    time_horizon : float or int, default=None
        A specific time horizon `t_horizon` to treat the model as a
        probabilistic classifier to estimate `E[T_k < t_horizon|X]` where `T_k`
        is a random variable representing the (uncensored) event for the type
        of interest.

        When specified, the `predict_proba` method returns an estimate of
        `E[T_k < t_horizon|X]` for each provided realisation of `X`.

    TODO: complete the docstring.

    References
    ----------
    
    [1] M. Kretowska, "Tree-based models for survival data with competing risks",
        Computer Methods and Programs in Biomedicine 159 (2018) 185-198.
    """

    name = "GradientBoostedCIF"

    def __init__(
        self,
        event_of_interest="any",
        objective="ibs",
        n_iter=10,
        n_repetitions_per_iter=5,
        learning_rate=0.1,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        show_progressbar=True,
        n_time_grid_steps=1000,
        time_horizon=None,
        random_state=None,
    ):
        self.event_of_interest = event_of_interest
        self.objective = objective
        self.n_iter = n_iter
        self.n_repetitions_per_iter = n_repetitions_per_iter # TODO? data augmenting for early iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.show_progressbar = show_progressbar
        self.n_time_grid_steps = n_time_grid_steps
        self.time_horizon = time_horizon
        self.random_state = random_state

    def _build_base_estimator(self, monotonic_cst):

        if self.objective == "ibs":
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
        elif self.objective == "inll":
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
                "Parameter 'objective' must be either 'ibs' or 'inll', "
                f"got {self.objective}."
            )
        
    def fit(self, X, y, times=None):
        X = check_array(X)
        y = _check_y_survival(y)

        self.event_ids_ = np.unique(y["event"])

        # The time horizon is concatenated as an additional input feature
        # before the features of X and we constrain the prediction function
        # (that estimates the CIF) to monotically increase with the time
        # horizon feature.
        monotonic_cst = np.zeros(X.shape[1] + 1)
        monotonic_cst[0] = 1

        self.estimator_ = self._build_base_estimator(monotonic_cst)

        # Compute the time grid used at prediction time.
        any_event_mask = y["event"] > 0
        observed_times = y["duration"][any_event_mask]

        if times is None:
            if observed_times.shape[0] > self.n_time_grid_steps:
                self.time_grid_ = np.quantile(
                    observed_times,
                    np.linspace(0, 1, num=self.n_time_grid_steps)
                )
            else:
                self.time_grid_ = observed_times.copy()
                self.time_grid_.sort()
        else:
            self.time_grid_ = times
        
        ibs_training_sampler = BrierScoreSampler(
            y,
            event_of_interest=self.event_of_interest,
            random_state=self.random_state,
        )
        
        iterator = range(self.n_iter)
        if self.show_progressbar:
            iterator = tqdm(iterator)

        for idx_iter in iterator:
            (
                sampled_times,
                y_binary,
                sample_weight,
            ) = ibs_training_sampler.draw()
            Xt = np.hstack([sampled_times, X])
            self.estimator_.max_iter += 1
            self.estimator_.fit(Xt, y_binary, sample_weight=sample_weight)
            
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
        all_y_cif = []

        if times is None:
            times = self.time_grid_
    
        if self.show_progressbar:
            times = tqdm(times)

        for t in times:
            t = np.full((X.shape[0], 1), t)
            X_with_t = np.hstack([t, X])
            if self.objective == "ibs":
                y_cif = self.estimator_.predict(X_with_t)
            else:
                y_cif = self.estimator_.predict_proba(X_with_t)[:, 1] 
            all_y_cif.append(y_cif)
        
        cif = np.vstack(all_y_cif).T

        if self.objective == "ibs":
            cif = np.clip(cif, 0, 1)

        return cif
    
    def predict_survival_function(self, X, times=None):
        """Compute the event specific survival function.
        
        Warning: this metric only makes sense when y_train["event"] is binary
        (single event) or when setting event_of_interest='any'.
        """
        if (
            (self.event_ids_ > 0).sum() > 1
            and self.event_of_interest != "any"
        ):
            warnings.warn(
                f"Values returned by predict_survival_function only make "
                f"sense when the model is trained with a binary event "
                f"indicator or when setting event_of_interest='any'. "
                f"Instead this model was fit on data with event ids "
                f"{self.event_ids_.tolist()} and with "
                f"event_of_interest={self.event_of_interest}."
            )
        return 1 - self.predict_cumulative_incidence(X, times=times)
    
    def predict_quantile(self, X, quantile=0.5, times=None):
        """Estimate the conditional median (or other quantile) time to event
        
        Note: this can return np.inf values when the estimated CIF does not
        reach the `quantile` value at the maximum time horizon observed on
        the training set.
        """
        if times is None:
            times = self.time_grid_
        cif_curves = self.predict_cumulative_incidence(X, times=times)
        quantile_idx = np.apply_along_axis(
            lambda a: a.searchsorted(quantile, side='right'), 1, cif_curves
        )
        inf_mask = quantile_idx == cif_curves.shape[1]
        # Change quantile_idx to avoid out-of-bound index in the subsequent
        # line.
        quantile_idx[inf_mask] = cif_curves.shape[1] - 1
        results = times[quantile_idx]
        # Mark out-of-index results as np.inf
        results[inf_mask] = np.inf
        return results
