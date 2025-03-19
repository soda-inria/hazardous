import torch
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd


def pad_col_2d(input, val=0, where="end"):
    """Pad a 2d tensor column-wise.

    Parameters
    ----------
    input : torch.tensor of shape (n_samples, n_time_steps)
        Input to pad.

    val : int, default=0
        Padding value.

    where : {'start', 'end'}, default='end'
        * If set to start, the padding is added on the left.
        * If set to end, the padding is added to the right.
    """
    if input.ndim != 2:
        raise ValueError("Only works for `phi` tensor that is 2D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == "end":
        return torch.cat([input, pad], dim=1)
    elif where == "start":
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")


def pad_col_3d(input, val=0, where="end"):
    """Pad a 3d tensor second-axis-wise.

    Parameters
    ----------
    input : torch.tensor of shape (n_samples, n_time_steps, n_events)
        Input to pad on the second axis (axis=1).

    val : int, default=0
        Padding value.

    where : {'start', 'end'}, default='end'
        * If set to start, the padding is added on the left.
        * If set to end, the padding is added to the right.
    """
    if input.ndim != 3:
        raise ValueError("Only works for `phi` tensor that is 3D.")
    pad = torch.zeros_like(input[:, :, :1])
    if val != 0:
        pad = pad + val
    if where == "end":
        return torch.cat([input, pad], dim=2)
    elif where == "start":
        return torch.cat([pad, input], dim=2)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")


def check_y_survival(y):
    """Convert DataFrame and dictionnary to record array."""
    y_keys = ["event", "duration"]
    if (
        isinstance(y, np.ndarray)
        and sorted(y.dtype.names, reverse=True) == y_keys
        or isinstance(y, dict)
        and sorted(y, reverse=True) == y_keys
    ):
        return np.ravel(y["event"]), np.ravel(y["duration"])

    elif isinstance(y, pd.DataFrame) and sorted(y.columns, reverse=True) == y_keys:
        return y["event"].values, y["duration"].values

    else:
        raise ValueError(
            "y must be a record array, a pandas DataFrame, or a dict "
            "whose dtypes, keys or columns are 'event' and 'duration'. "
            f"Got:\n{repr(y)}"
        )


def get_n_events(event):
    """Fetch the number of distinct competing events.

    Parameters
    ----------
    event : pd.Series of shape (n_samples,)
        Binary or multiclass events.

    Returns
    -------
    n_events : int
        The number of events, without accounting for the censoring 0.
    """
    event_ids = event.unique()
    has_censoring = int(0 in event_ids)
    return len(event_ids) - has_censoring


class SurvStratifiedShuffleSplit(StratifiedShuffleSplit):
    def split(self, X, y, groups=None):
        if isinstance(y, np.ndarray):
            if len(np.unique(y[:, 0])) < 5:
                event = y[:, 0]
            else:
                event = y[:, 1]
        event = event.astype(int)
        return super().split(X, event, groups)


class SurvStratifiedSingleSplit(SurvStratifiedShuffleSplit):
    def split(self, X, y, groups=None):
        train, test = next(iter(super().split(X, y, groups)))
        yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1
