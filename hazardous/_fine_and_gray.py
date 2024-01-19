import numpy as np
import pandas as pd
from rpy2 import rinterface, robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, check_is_fitted

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
    with localconverter(robjects.default_converter + pandas2ri.converter):
        return robjects.conversion.rpy2py(r_dataframe)


def r_dataframe(pd_dataframe):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        return robjects.conversion.py2rpy(pd_dataframe)


def parse_r_list(r_list):
    return dict(zip(r_list.names, np.array(r_list, dtype=object)))


class FineGrayEstimator(BaseEstimator):
    def __init__(self, event_of_interest=1):
        self.event_of_interest = event_of_interest

    def fit(self, X, y):
        X = self._check_input(X)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        event, duration = check_y_survival(y)
        self.times_ = np.unique(duration[event == self.event_of_interest])
        event, duration = r_vector(event), r_vector(duration)
        X = r_dataframe(X)

        self.r_crr_result_ = r_cmprsk.crr(
            duration,
            event,
            X,
            failcode=self.event_of_interest,
            cencode=0,
        )
        parsed = parse_r_list(self.r_crr_result_)
        self.coef_ = np.array(parsed["coef"])

        return self

    def predict_cumulative_incidence(self, X, times=None):
        check_is_fitted(self, "r_crr_result_")

        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)

        X = r_matrix(X)

        y_pred = r_cmprsk.predict_crr(
            self.r_crr_result_,
            X,
        )
        y_pred = np_matrix(y_pred)

        unique_times = y_pred[:, 0]  # durations seen during fit
        y_pred = y_pred[:, 1:].T  # shape (n_samples, n_unique_times)

        # Interpolate each sample
        if times is not None:
            all_y_pred = []
            for idx in y_pred.shape[0]:
                y_pred_ = interp1d(
                    x=unique_times,
                    y=y_pred[idx, :],
                    kind="linear",
                )(times)
                all_y_pred.append(y_pred_)
            y_pred = np.vstack(all_y_pred)

        return y_pred

    def _check_input(self, X):
        if not hasattr(X, "__dataframe__"):
            X = pd.DataFrame(X)

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
