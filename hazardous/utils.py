import numpy as np
import pandas as pd


def _dict_to_recarray(y):
    y_rec = np.empty(
        shape=y.shape[0],
        dtype=[("event", np.int8), ("duration", np.float64)],
    )
    y_rec["event"] = y["event"]
    y_rec["duration"] = y["duration"]
    return y_rec


def _df_to_recarray(y):
    y_rec = np.empty(
        shape=y.shape[0],
        dtype=[("event", np.int8), ("duration", np.float64)],
    )
    y_rec["event"] = y["event"].values
    y_rec["duration"] = y["duration"].values
    return y_rec


def _check_y_survival(y):
    """Convert DataFrame and dictionnary to record array.
    """
    y_keys = ["event", "duration"]

    if (
        isinstance(y, np.ndarray)
        and sorted(y.dtype.names, reverse=True) == y_keys
    ):
        pass

    elif (
        isinstance(y, dict)
        and sorted(y, reverse=True) == y_keys
    ):
        y_rec = _dict_to_recarray(y)

    elif (
        isinstance(y, pd.DataFrame)
        and sorted(y.columns, reverse=True) == y_keys
    ):
        y_rec = _df_to_recarray(y)

    else:
        raise ValueError(
            "y must be a record array, a pandas DataFrame, or a dict "
            "whose dtypes, keys or columns are 'event' and 'duration'. "
            f"Got:\n{repr(y)}"
        )

    return y_rec
    
