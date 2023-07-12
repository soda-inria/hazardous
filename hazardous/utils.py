import warnings
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_scalar


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
    check_scalar(k, "event_of_interest", target_type=(str, Integral))
    not_str_any = isinstance(k, str) and k != "any"
    not_positive = isinstance(k, int) and k < 1
    if not_str_any or not_positive:
        raise ValueError(
            "event_of_interest must be a strictly positive integer or 'any', "
            f"got: event_of_interest={k}"
        )
    return


def check_y_mean_increasing(y_pred, times):
    """Check that the mean of y is increasing
    when sorting with times.

    This allow to warn users of spurious survival probability inputs,
    when the incidence probability is expected instead.

    Parameters
    ----------
    y_pred : np.ndarray of shape (n_samples, n_times)
        The incidence probability, expected to be monotonically increasing.

    times : np.ndarray of shape (n_times)
        The unsorted array of times used to estimate y_pred.
    """
    idx_time_sorted = np.argsort(times)
    y_mean = y_pred.mean(axis=0)[idx_time_sorted]
    is_y_mean_decreasing = y_mean[0] > y_mean[-1]
    if is_y_mean_decreasing:
        warnings.warn(
            "\n\nThe average shape of the y_pred curve is decreasing. "
            "However, the Cumulative Incidence Function should be "
            "monotonically increasing.\n"
            "The brier score for the kth cause of failure only makes "
            "sens with incidence probability, not survival probability.\n\n"
            "In the binary event settings: "
            "incidence probability = 1 - survival probability.\n"
        )
    return
