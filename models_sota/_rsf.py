import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.utils import check_random_state
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
from rpy2.robjects import r

from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
)
from hazardous.utils import check_y_survival
from models_sota.r_utils import r_dataframe, parse_r_list

rfs = importr("randomForestSRC")
surv = importr("survival")
Surv = surv.Surv


class RSFEstimator(BaseEstimator):
    """RSF estimator for competing risks.

    This estimator is a rpy2 wrapper around the randomForestSRC R package.

    Parameters
    ----------
    event_of_interest : int, default=1,
        The event to perform Fine and Gray regression on.

    max_fit_samples : int, default=10_000,
        The maximum number of samples to use during fit.
        This is required since the time complexity of this operation is quadratic.

    random_state : default=None
        Used to subsample X during fit when X has more samples
        than max_fit_samples.
    """

    def __init__(
        self,
        max_fit_samples=None,
        random_state=0,
    ):
        self.max_fit_samples = max_fit_samples
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
        self : fitted instance of RSFEstimator
        """
        X = self._check_input(X, y, reset=True)

        self.max_fit_samples_ = self.max_fit_samples or X.shape[0]

        if X.shape[0] > self.max_fit_samples_:
            rng = check_random_state(self.random_state)
            sample_indices = rng.choice(
                np.arange(X.shape[0]),
                size=self.max_fit_samples,
                replace=False,
            )
            X, y = X.iloc[sample_indices], y.iloc[sample_indices]

        event, duration = check_y_survival(y)
        self.event_ids_ = np.array(sorted(list(set([0]) | set(event))))
        df = pd.concat([X, y], axis=1)
        cols = df.columns.to_list()
        # remove space in column names
        cols = [f"col{i}" for i in range(X.shape[1])] + y.columns.to_list()
        df.columns = cols
        names = " + ".join(cols)
        r_df = r_dataframe(df)
        rsf_object = rfs.rfsrc(
            Formula(f"Surv(duration, event) ~ {names}"),
            data=r_df,
            seed=self.random_state,
        )
        self.r_parsed = rsf_object
        self.parsed = parse_r_list(rsf_object)
        self.times_ = np.array(self.parsed["time.interest"])
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
        check_is_fitted(self, "parsed")
        X = self._check_input(X, y=None, predict_time=True, reset=False)
        df = X.copy()
        cols = [f"col{i}" for i in range(df.shape[1])]
        df.columns = cols
        r_df = r_dataframe(df)

        # predict
        object_pred_r = r.predict(self.r_parsed, r_df)

        object_pred = parse_r_list(object_pred_r)
        cif_pred = np.array(object_pred["cif"])

        # make the same format as SurvivalBoost,
        # where the first column is the survival function
        if len(self.event_ids_) > 2:
            survival_pred = 1 - cif_pred.sum(axis=2)
            survival_pred = survival_pred[:, :, None]
        else:
            survival_pred = np.array(object_pred["survival"])
            survival_pred = survival_pred[:, :, None]
            cif_pred = 1 - survival_pred.sum(axis=2)
            cif_pred = cif_pred[:, :, None]
        y_pred = np.concatenate((survival_pred, cif_pred), axis=2)
        y_pred = y_pred.swapaxes(1, 2)

        all_event_y_pred = []

        for event in range(y_pred.shape[1]):
            event_pred = y_pred[:, event, :]
            if event == 0:
                y_pred_at_0 = np.ones((event_pred.shape[0], 1))
            else:
                y_pred_at_0 = np.zeros((event_pred.shape[0], 1))

            y_pred_t_max = event_pred[:, [-1]]
            event_pred = np.hstack([y_pred_at_0, event_pred, y_pred_t_max])

            if times is None:
                times = self.times_

            times_event = np.hstack([[0], self.times_, [np.inf]])

            all_y_pred = []
            for idx in range(event_pred.shape[0]):
                y_pred_ = interp1d(
                    x=times_event,
                    y=event_pred[idx, :],
                    kind="linear",
                )(times)
                all_y_pred.append(y_pred_)

            event_pred = np.vstack(all_y_pred)
            all_event_y_pred.append(event_pred)

        all_event_y_pred = np.asarray(all_event_y_pred)
        return all_event_y_pred.swapaxes(0, 1)

    def _check_input(self, X, y, reset=True, predict_time=True):
        if not hasattr(X, "__dataframe__"):
            X = pd.DataFrame(X)
        if not predict_time and not hasattr(y, "__dataframe__"):
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
            self.stds_ = X.std(axis=0)
        X = X.loc[:, self.stds_ > 0]

        return X

    def score(self, X, y):
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
