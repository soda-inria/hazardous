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

    A fraction of the individuals are censored by sampling a censoring time
    from a Weibull distribution with shape 1 and scale equal to the mean
    duration of the target event times the ``censoring_relative_scale``.

    Setting ``censoring_relative_scale`` to 0 or None disables censoring.
    Setting it to a small value (e.g. 0.5 instead of 1.5) will result in a
    larger fraction of censored individuals.
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
