from itertools import cycle

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import weibull_min
from sklearn.datasets._base import Bunch
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.utils import check_random_state

DEFAULT_SHAPE_RANGES = (
    (0.7, 0.9),
    (1.0, 1.0),
    (2.0, 3.0),
)

DEFAULT_SCALE_RANGES = (
    (1, 20),
    (1, 10),
    (1.5, 5),
)


def _censor(
    y,
    independent=True,
    X=None,
    features_censoring_rate=0.2,
    censoring_relative_scale=1.5,
    random_state=0,
):
    rng = np.random.RandomState(random_state)
    if censoring_relative_scale == 0 or censoring_relative_scale is None:
        return y

    if independent:
        scale_censoring = censoring_relative_scale * y["duration"].mean()
        shape_censoring = 1
    else:
        features_impact_censoring = rng.randn(X.shape[1], 2)
        features_impact_censoring = np.where(
            rng.rand(
                features_impact_censoring.shape[0], features_impact_censoring.shape[1]
            )
            < features_censoring_rate,
            features_impact_censoring,
            0,
        )
        df_censoring_params = censoring_relative_scale * X @ features_impact_censoring
        df_censoring_params.columns = ["shape_0", "scale_0"]
        df_censoring_params = rescaling_params_to_respect_default_ranges(
            df_censoring_params,
            shape_default=(1.0, 1.0),
            scale_default=(1, 10 * censoring_relative_scale),
            event=0,
        )
        scale_censoring = df_censoring_params["scale_0"].values * y["duration"].mean()
        shape_censoring = df_censoring_params["shape_0"].values
    censoring = weibull_min.rvs(
        shape_censoring, scale=scale_censoring, size=y.shape[0], random_state=rng
    )
    y_censored = y.copy()
    y_censored["event"] = np.where(y["duration"] < censoring, y_censored["event"], 0)
    y_censored["duration"] = np.minimum(y_censored["duration"], censoring)
    return y_censored


def compute_shape_and_scale(
    df_features,
    features_rate=0.2,
    n_events=3,
    degree_interaction=2,
    shape_ranges=DEFAULT_SHAPE_RANGES,
    scale_ranges=DEFAULT_SCALE_RANGES,
    random_state=0,
):
    rng = np.random.RandomState(random_state)
    # Adding interactions between features
    df_poly_features = PolynomialFeatures(
        degree=degree_interaction, interaction_only=True, include_bias=False
    )
    df_poly_features.set_output(transform="pandas")
    df_trans = df_poly_features.fit_transform(df_features)

    # Create masked matrix with the interactions
    n_weibull_parameters = 2 * n_events
    w_star = rng.randn(df_trans.shape[1], n_weibull_parameters)
    # Set 1-feature_rate% of the w_star to 0
    w_star = np.where(
        rng.rand(w_star.shape[0], w_star.shape[1]) < features_rate, w_star, 0
    )
    # Computation of the true values of shape and scale
    shape_scale_star = df_trans.values @ w_star
    shape_scale_columns = [f"shape_{i}" for i in range(1, n_events + 1)] + [
        f"scale_{i}" for i in range(1, n_events + 1)
    ]
    df_shape_scale_star = pd.DataFrame(shape_scale_star, columns=shape_scale_columns)
    # Rescaling of these values to stay in the chosen range
    for event_id, shape_range, scale_range in zip(
        range(1, n_events + 1), cycle(shape_ranges), cycle(scale_ranges)
    ):
        df_shape_scale_star = rescaling_params_to_respect_default_ranges(
            df_shape_scale_star, shape_range, scale_range, event_id
        )
    return df_shape_scale_star


def rescaling_params_to_respect_default_ranges(
    df_shape_scale_star, shape_default, scale_default, event
):
    # Rescaling of these values to stay in the chosen range
    shape_min, shape_max = shape_default
    scale_min, scale_max = scale_default
    shape = df_shape_scale_star[f"shape_{event}"].copy()
    scale = df_shape_scale_star[f"scale_{event}"].copy()
    scaler = MinMaxScaler(feature_range=(-2, 2))
    df_shape_scale_star[f"shape_{event}"] = (
        shape_min
        + (shape_max - shape_min)
        * expit(scaler.fit_transform(shape.values.reshape(-1, 1)))
    ).flatten()
    df_shape_scale_star[f"scale_{event}"] = (
        scale_min
        + (scale_max - scale_min)
        * expit(scaler.fit_transform(scale.values.reshape(-1, 1))).flatten()
    )
    return df_shape_scale_star


def make_simple_features(
    n_events=3,
    n_samples=3_000,
    base_scale=1_000,
    shape_ranges=DEFAULT_SHAPE_RANGES,
    scale_ranges=DEFAULT_SCALE_RANGES,
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
    return X, event_durations, duration_argmin


def make_complex_features_with_sparse_matrix(
    n_events=3,
    n_samples=3_000,
    base_scale=1_000,
    shape_ranges=DEFAULT_SHAPE_RANGES,
    scale_ranges=DEFAULT_SCALE_RANGES,
    n_features=10,
    features_rate=0.3,
    degree_interaction=2,
    random_state=0,
):
    rng = np.random.RandomState(random_state)
    # Create features given to the model as X and then creating the interactions
    df_features = pd.DataFrame(rng.randn(n_samples, n_features))
    df_features.columns = [f"feature_{i}" for i in range(n_features)]

    df_shape_scale_star = compute_shape_and_scale(
        df_features,
        features_rate,
        n_events,
        degree_interaction,
        shape_ranges,
        scale_ranges,
        random_state,
    )
    # Throw durations from a weibull distribution with scale and shape as the parameters
    event_durations = []
    for event in range(1, n_events + 1):
        shape = df_shape_scale_star[f"shape_{event}"]
        scale = df_shape_scale_star[f"scale_{event}"]
        durations = weibull_min.rvs(shape, scale=scale * base_scale, random_state=rng)
        event_durations.append(durations)

    # Creating the target tabular
    event_durations = np.asarray(event_durations)
    duration_argmin = np.argmin(event_durations, axis=0)
    return df_features, event_durations, duration_argmin


def make_synthetic_competing_weibull(
    n_events,
    n_samples=3_000,
    base_scale=1_000,
    n_features=10,
    features_rate=0.3,
    degree_interaction=2,
    independent=True,
    features_censoring_rate=0.2,
    return_uncensored_data=False,
    return_X_y=True,
    feature_rounding=2,
    target_rounding=1,
    shape_ranges=DEFAULT_SHAPE_RANGES,
    scale_ranges=DEFAULT_SCALE_RANGES,
    censoring_relative_scale=1.5,
    random_state=0,
    complex_features=False,
):
    """
    Creating a synthetic dataset to make competing risks.
    Depending of the choice of the user, the function may genere a simple dataset
    (setting ``complex_features`` to False or either create complex features.)

    A fraction of the individuals are censored by sampling a censoring time
    from a Weibull distribution with shape 1 and scale equal to the mean
    duration of the target event times the ``censoring_relative_scale``.

    Setting ``censoring_relative_scale`` to 0 or None disables censoring.
    Setting it to a small value (e.g. 0.5 instead of 1.5) will result in a
    larger fraction of censored individuals.
    """
    if complex_features:
        X, event_durations, duration_argmin = make_complex_features_with_sparse_matrix(
            n_events,
            n_samples,
            base_scale,
            shape_ranges,
            scale_ranges,
            n_features,
            features_rate,
            degree_interaction,
            random_state,
        )
    else:
        X, event_durations, duration_argmin = make_simple_features(
            n_events, n_samples, base_scale, shape_ranges, scale_ranges, random_state
        )
    y = pd.DataFrame(
        dict(
            event=duration_argmin + 1,
            duration=event_durations[duration_argmin, np.arange(n_samples)],
        )
    )
    y_censored = _censor(
        y,
        censoring_relative_scale=censoring_relative_scale,
        independent=independent,
        X=X,
        features_censoring_rate=features_censoring_rate,
        random_state=random_state,
    )
    if feature_rounding is not None:
        X = X.round(feature_rounding)

    if target_rounding is not None:
        y_censored = y_censored.round(target_rounding)
        y = y.round(target_rounding)

    if return_X_y:
        if return_uncensored_data:
            return X, y_censored, y
        return X, y

    frame = pd.concat([X, y], axis=1)
    return Bunch(data=frame[X.columns], target=frame[y.columns], frame=frame)
