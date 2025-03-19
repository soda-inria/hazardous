import numpy as np
import pandas as pd
from rpy2 import rinterface, robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


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
