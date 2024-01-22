import numpy as np
import pandas as pd
from rpy2 import rinterface, robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.utils import check_random_state

from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    integrated_brier_score_incidence_oracle,
)
from hazardous.utils import check_y_survival

r_cmprsk = importr("cmprsk")


def r_vector(array):
    """Convert a numpy vector to an R vector.

    Parameters
    ----------
    array : ndarray of shape (n_samples,)

    Returns
    -------
    array_out : rpy2.rinterface.SexpVector
        R vector of compatible data type
    """
    if array.ndim != 1:
        raise ValueError(f"array must be 1d, got {array.ndim}")

    dtype = array.dtype
    if np.issubdtype(dtype, np.integer):
        return rinterface.IntSexpVector(array)
    elif np.issubdtype(dtype, np.floating):
        return rinterface.FloatSexpVector(array)
    elif np.issubdtype(dtype, bool):
        return rinterface.BoolSexpVector(array)
    elif np.issubdtype(dtype, str):
        return rinterface.StrSexpVector(array)
    else:
        msg = f"Can't convert vectors with dtype {dtype} yet"
        raise NotImplementedError(msg)


def r_matrix(X):
    """Convert 2d array or pandas dataframe to an R matrix.

    Parameters
    ----------
    X : pd.DataFrame or ndarray of shape (n_samples, n_features)

    Returns
    -------
    X_out : robjects.r.matrix
        R matrix of compatible data type.
    """

    if X.ndim != 2:
        raise ValueError(f"X must be 2d, got {X.ndim}.")

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    n_samples = X.shape[0]

    X = r_vector(X.ravel())
    X = robjects.r.matrix(X, ncol=n_samples).transpose()

    return X


def np_matrix(r_dataframe):
    """Convert a R dataframe into a numpy 2d array."""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        return robjects.conversion.rpy2py(r_dataframe)


def r_dataframe(pd_dataframe):
    """Convert a Pandas dataframe into a R dataframe."""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        return robjects.conversion.py2rpy(pd_dataframe)


def parse_r_list(r_list):
    return dict(zip(r_list.names, np.array(r_list, dtype=object)))


class FineGrayEstimator(BaseEstimator):
    """Fine and Gray competing risk estimator.

    This estimator is a rpy2 wrapper around the cmprsk R package.

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
        max_fit_samples=10_000,
        random_state=None,
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
        self : fitted instance of FineGrayEstimator
        """
        X = self._check_input(X, y)

        if X.shape[0] > self.max_fit_samples:
            rng = check_random_state(self.random_state)
            sample_indices = rng.choice(
                np.arange(X.shape[0]),
                size=self.max_fit_samples,
                replace=False,
            )
            X, y = X.iloc[sample_indices], y.iloc[sample_indices]

        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        y = y.loc[X.index]
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
        return all_event_y_pred

    def _check_input(self, X, y):
        if not hasattr(X, "__dataframe__"):
            X = pd.DataFrame(X)

        if not hasattr(y, "__dataframe__"):
            raise TypeError(f"'y' must be a Pandas dataframe, got {type(y)}.")

        # Check no categories
        numeric_columns = X.select_dtypes("number").columns
        if numeric_columns.shape[0] != X.shape[1]:
            categorical_columns = set(X.columns).difference(list(numeric_columns))
            raise ValueError(
                f"Categorical columns {categorical_columns} need to be encoded."
            )

        # Check no constant columns
        stds = X.std(axis=0)
        if (stds == 0).any():
            constant_columns = stds[stds == 0].index
            raise ValueError(f"Constant columns {constant_columns} need jittering.")

        return X

    def score(self, X, y, shape_censoring=None, scale_censoring=None):
        """Return

        #TODO: implement time integrated NLL.
        """
        predicted_curves = self.predict_cumulative_incidence(X)
        ibs_events = []

        for idx, event_id in enumerate(self.event_ids_[1:]):
            predicted_curves_for_event = predicted_curves[idx]
            if scale_censoring is not None and shape_censoring is not None:
                ibs_event = integrated_brier_score_incidence_oracle(
                    y_train=self.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.times_,
                    shape_censoring=shape_censoring,
                    scale_censoring=scale_censoring,
                    event_of_interest=event_id,
                )

            else:
                ibs_event = integrated_brier_score_incidence(
                    y_train=self.y_train,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.times_,
                    event_of_interest=event_id,
                )

            ibs_events.append(ibs_event)
        return -np.mean(ibs_events)
