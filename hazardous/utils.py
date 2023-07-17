import numpy as np
import pandas as pd
from sklearn.utils.validation import check_scalar


def _dict_to_pd(y):
    return pd.DataFrame(y)


def _dict_to_recarray(y, cast_event_to_bool=False):
    if cast_event_to_bool:
        event_dtype = np.bool_
    else:
        event_dtype = y["event"].dtype
    y_out = np.empty(
        shape=y["event"].shape[0],
        dtype=[("event", event_dtype), ("duration", y["duration"].dtype)],
    )
    y_out["event"] = y["event"]
    y_out["duration"] = y["duration"]
    return y_out


def check_y_survival(y):
    """Convert DataFrame and dictionnary to record array."""
    y_keys = ["event", "duration"]

    if (
        isinstance(y, np.ndarray)
        and sorted(y.dtype.names, reverse=True) == y_keys
        or isinstance(y, dict)
        and sorted(y, reverse=True) == y_keys
    ):
        event, duration = np.ravel(y["event"]), np.ravel(y["duration"])

    elif isinstance(y, pd.DataFrame) and sorted(y.columns, reverse=True) == y_keys:
        event, duration = y["event"].values, y["duration"].values

    else:
        raise ValueError(
            "y must be a record array, a pandas DataFrame, or a dict "
            "whose dtypes, keys or columns are 'event' and 'duration'. "
            f"Got:\n{repr(y)}"
        )

    return event, duration


def check_event_of_interest(k):
    """`event_of_interest` must be the string 'any' or a positive integer."""
    check_scalar(k, "event_of_interest", target_type=(str, int))
    not_str_any = isinstance(k, str) and k != "any"
    not_positive = isinstance(k, int) and k < 1
    if not_str_any or not_positive:
        raise ValueError(
            "event_of_interest must be a strictly positive integer or 'any', "
            f"got: event_of_interest={k}"
        )
    return
