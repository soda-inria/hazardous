import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.utils import check_random_state

from hazardous.metrics._brier_score import integrated_brier_score_incidence
from hazardous.utils import check_y_survival
from models_sota.r_utils import r_dataframe, r_matrix, r_vector, np_matrix, parse_r_list

r_cmprsk = importr("cmprsk")


class FineGrayEstimator(BaseEstimator):
    """Fine and Gray competing risk estimator.

    This estimator is a rpy2 wrapper around the cmprsk R package.

    Parameters
    ----------
    max_samples : int, default=10_000,
        The maximum number of samples to use during fit.
        This is required since the time complexity of this operation is quadratic.

    random_state : default=None
        Used to subsample X during fit when X has more samples
        than max_fit_samples.
    """

    def __init__(
        self,
        max_samples=10_000,
        random_state=None,
    ):
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input covariates

        y : pandas.DataFrame of shape (n_samples, 2)
            The target, with columns 'event' and 'duration'.

        Returns
        -------
        self : fitted instance of FineGrayEstimator
        """
        X = self._check_input(X, y, reset=True)

        if X.shape[0] > self.max_samples:
            rng = check_random_state(self.random_state)
            sample_indices = rng.choice(
                np.arange(X.shape[0]),
                size=self.max_samples,
                replace=False,
            )
            X, y = X.iloc[sample_indices], y.iloc[sample_indices]

        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        event, duration = check_y_survival(y)
        self.times_ = np.unique(duration[event > 0])
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))

        event, duration = r_vector(event), r_vector(duration)

        X = r_dataframe(X)

        self.r_crr_results_ = []
        self.coefs_ = []
        for event_id in self.event_ids_[1:]:
            r_crr_result_ = r_cmprsk.crr(
                duration,
                event,
                X,
                failcode=int(event_id),
                cencode=0,
            )

            parsed = parse_r_list(r_crr_result_)
            coef_ = np.array(parsed["coef"])

            self.r_crr_results_.append(r_crr_result_)
            self.coefs_.append(coef_)

        self.y_train = y
        return self

    def predict_cumulative_incidence(self, X, times=None):
        """Predict the conditional cumulative incidence.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)

        times : ndarray of shape (n_times,), default=None
            The time steps to estimate the cumulative incidence at.
            * If set to None, the duration of the event of interest
              seen during fit 'times_' is used.
            * If not None, this performs a linear interpolation for each sample.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_times)
            The conditional cumulative cumulative incidence at times.
        """
        check_is_fitted(self, "r_crr_results_")

        X = self._check_input(X, y=None, reset=False)

        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)

        X = r_matrix(X)

        all_event_y_pred = []
        for event_id in self.event_ids_[1:]:
            # Interpolate each sample
            y_pred = r_cmprsk.predict_crr(
                self.r_crr_results_[event_id - 1],
                X,
            )
            y_pred = np_matrix(y_pred)
            times_event = np.unique(
                self.y_train["duration"][self.y_train["event"] == event_id]
            )
            y_pred = y_pred[:, 1:].T  # shape (n_samples, n_unique_times)

            y_pred_at_0 = np.zeros((y_pred.shape[0], 1))
            y_pred_t_max = y_pred[:, [-1]]
            y_pred = np.hstack([y_pred_at_0, y_pred, y_pred_t_max])

            times_event = np.hstack([[0], times_event, [np.inf]])

            if times is None:
                times = self.times_

            all_y_pred = []
            for idx in range(y_pred.shape[0]):
                y_pred_ = interp1d(
                    x=times_event,
                    y=y_pred[idx, :],
                    kind="linear",
                )(times)
                all_y_pred.append(y_pred_)

            y_pred = np.vstack(all_y_pred)
            all_event_y_pred.append(y_pred)

        surv_curve = 1 - np.sum(all_event_y_pred, axis=0)
        all_event_y_pred = [surv_curve] + all_event_y_pred
        all_event_y_pred = np.asarray(all_event_y_pred)

        return all_event_y_pred.swapaxes(0, 1)

    def _check_input(self, X, y=None, reset=True):
        if not hasattr(X, "__dataframe__"):
            X = pd.DataFrame(X)

        if y is not None and not hasattr(y, "__dataframe__"):
            raise TypeError(f"'y' must be a Pandas dataframe, got {type(y)}.")

        # Check no categories
        numeric_columns = X.select_dtypes("number").columns
        if numeric_columns.shape[0] != X.shape[1]:
            categorical_columns = set(X.columns).difference(list(numeric_columns))
            raise ValueError(
                f"Categorical columns {categorical_columns} need to be encoded."
            )

        # Check no constant columns
        if reset:
            self.stds = X.std(axis=0)
        X = X.loc[:, self.stds > 0]

        return X

    def score(self, X, y, shape_censoring=None, scale_censoring=None):
        """Return

        #TODO: implement time integrated NLL.
        """
        predicted_curves = self.predict_cumulative_incidence(X)
        ibs_events = []

        for idx, event_id in enumerate(self.event_ids_[1:]):
            predicted_curves_for_event = predicted_curves[idx]
            ibs_event = integrated_brier_score_incidence(
                y_train=self.y_train,
                y_test=y,
                y_pred=predicted_curves_for_event,
                times=self.times_,
                event_of_interest=event_id,
            )

            ibs_events.append(ibs_event)
        return -np.mean(ibs_events)
