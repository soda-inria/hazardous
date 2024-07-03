from itertools import cycle

import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.datasets._base import Bunch
from sklearn.utils import check_random_state

DEFAULT_SHAPE_RANGES = (
    (0.4, 0.9),
    (1.0, 1.0),
    (1.2, 3),
)

DEFAULT_SCALE_RANGES = (
    (1, 20),
    (1, 10),
    (1.5, 5),
)


def _censor(y, relative_scale, random_state=None):
    """Censoring a population based on a relative scale.

    Individuals are censored by sampling a censoring time from
    a Weibull distribution with shape 1 and scale equal to
    the mean duration of the target event times the
    ``relative_scale``.

    Parameters
    ----------
    y: ndarray
        The target population.
    relative_scale: float
        Relative scale of the censoring. Setting it to 0 or None
        disables censoring, setting it to a small value (e.g. 0.5
        instead of 1.5) will result in a larger fraction of
        censored individuals.

    """

    if relative_scale == 0 or relative_scale is None:
        return y

    rng = check_random_state(random_state)
    scale = relative_scale * y["duration"].mean()
    censoring = weibull_min.rvs(1, scale=scale, size=y.shape[0], random_state=rng)
    y = y.copy()
    y["event"] = np.where(y["duration"] < censoring, y["event"], 0)
    y["duration"] = np.minimum(y["duration"], censoring)
    return y


def make_synthetic_competing_weibull(
    n_events=3,
    n_samples=3000,
    return_X_y=False,
    base_scale=1_000,
    feature_rounding=2,
    target_rounding=1,
    shape_ranges=DEFAULT_SHAPE_RANGES,
    scale_ranges=DEFAULT_SCALE_RANGES,
    censoring_relative_scale=1.5,
    random_state=None,
):
    """Generate a synthetic dataset with competing Weibull-distributed events.

    For each individual, we first sample one pair of shape and scale parameters
    for each event type uniformly from the given ranges. Each event type has a
    different range of shape and scale parameters.

    Then we sample event durations for each event type from the corresponding
    Weibull distribution parametrized by the sampled shape and scale
    parameters.

    The shape and scale parameters are returned as features. For each
    individual, the event type with the shortest duration is kept as the target
    event (competing risks setting) and its event identifier and duration are
    returned as the target dataframe.

    A fraction of the individuals are censored if ``censoring_relative_scale``
    is not None or 0.

    Parameters
    ----------
    n_events: int, default=3
        Number of events.
    n_samples: int, default=3000
        Number of individuals in the population.
    return_X_y: bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
    feature_rounding: int or None, default=2
        Round the feature values. If None, no rounding will be applied.
    target_rounding: int or None, default=1
        Round the target values. If None, no rounding will be applied.
    shape_ranges: tuple of shape (n_events, 2)
        The lower and upper boundary of the shape, `n_samples` shape
        values for `n_events` will be drawn from a uniform distribution.
    scale_ranges: tuple of shape (n_events, 2)
        The lower and upper boundary of the scale, `n_samples` scale
        values for `n_events` will be drawn from a uniform distribution.
    base_scale: int, default=1000
        Scaling parameter of the ``scale_range``.
    censoring_relative_scale: float, default=1.5
        Relative scale of the censoring level. Individuals are censored by
        sampling a censoring time from a Weibull distribution with shape 1
        and scale equal to the mean duration of the target event times
        the ``censoring_relative_scale``.
        Setting ``censoring_relative_scale`` to 0 or None disables censoring.
        Setting it to a small value (e.g. 0.5 instead of 1.5) will result in a
        larger fraction of censored individuals.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the uniform time sampler.

    Returns
    -------
    (data, target): tuple if ``return_X_y`` is True
        A tuple of two dataframes. The first containing a 2D array of shape
        (n_samples, n_features) with each row representing one sample
        and each column representing the events. The second dataframe
        of shape (n_samples, 2) containing the target samples.

    """
    rng = check_random_state(random_state)
    all_features = []
    event_durations = []

    for event_id, shape_range, scale_range in zip(
        range(1, n_events + 1), cycle(shape_ranges), cycle(scale_ranges)
    ):
        shape = rng.uniform(*shape_range, size=n_samples)
        scale = rng.uniform(*scale_range, size=n_samples) * base_scale
        all_features.append(pd.Series(shape, name=f"shape_{event_id}"))
        all_features.append(pd.Series(scale, name=f"scale_{event_id}"))
        durations = weibull_min.rvs(shape, scale=scale, random_state=rng)
        event_durations.append(durations)

    event_durations = np.asarray(event_durations)
    duration_argmin = np.argmin(event_durations, axis=0)
    X = pd.concat(all_features, axis=1)
    y = pd.DataFrame(
        dict(
            event=duration_argmin + 1,
            duration=event_durations[duration_argmin, np.arange(n_samples)],
        )
    )
    y = _censor(y, censoring_relative_scale, random_state=random_state)
    if feature_rounding is not None:
        X = X.round(feature_rounding)

    if target_rounding is not None:
        y = y.round(target_rounding)

    if return_X_y:
        return X, y

    frame = pd.concat([X, y], axis=1)
    return Bunch(data=frame[X.columns], target=frame[y.columns], frame=frame)
