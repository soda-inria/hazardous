"""
==================================================
Synthetic DataSet
==================================================

Trying to complexy synthetic data
"""

## TODO
# Change the linear scaling by an expit rescaling
# Improve censoring method by using covariates

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from scipy.stats import weibull_min

rng = np.random.default_rng(0)
seed = 0
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

n_events = 3
n_weibull_parameters = 2 * n_events
n_samples = 3_000
base_scale = 1_000
n_features = 10
features_rate = 0.3
degree_interaction = 2
relative_scale = 1.5

assert len(DEFAULT_SCALE_RANGES) == len(DEFAULT_SHAPE_RANGES)
assert len(DEFAULT_SHAPE_RANGES) == n_events


def compute_shape_and_scale(
    df_features, n_weibull_parameters, features_rate, n_events, degree_interaction
):
    # Adding interactions between features
    df_poly_features = PolynomialFeatures(
        degree=degree_interaction, interaction_only=True, include_bias=False
    )
    df_poly_features.set_output(transform="pandas")
    df_trans = df_poly_features.fit_transform(df_features)

    # Create masked matrix with the interactions
    w_star = np.random.randn(df_trans.shape[1], n_weibull_parameters)
    # Set 1-feature_rate% of the w_star to 0
    w_star = np.where(
        np.random.rand(w_star.shape[0], w_star.shape[1]) < features_rate, w_star, 0
    )
    # Computation of the true values of shape and scale
    shape_scale_star = df_trans.values @ w_star
    shape_scale_columns = [f"shape_{i}" for i in range(n_events)] + [
        f"scale_{i}" for i in range(n_events)
    ]
    df_shape_scale_star = pd.DataFrame(shape_scale_star, columns=shape_scale_columns)
    # Rescaling of these values to stay in the chosen range
    df_shape_scale_star = rescaling_params_to_respect_default_ranges(
        df_shape_scale_star
    )
    return df_shape_scale_star


def rescaling_params_to_respect_default_ranges(df_shape_scale_star):
    # Rescaling of these values to stay in the chosen range
    for event, (scale_default, shape_default) in enumerate(
        zip(DEFAULT_SCALE_RANGES, DEFAULT_SHAPE_RANGES)
    ):
        shape_min, shape_max = shape_default
        scale_min, scale_max = scale_default
        shape = df_shape_scale_star[f"shape_{event}"].copy()
        scale = df_shape_scale_star[f"scale_{event}"].copy()
        shape = shape_min + (shape_max - shape_min) * (shape - shape.min()) / (
            shape.max() - shape.min()
        )
        scale = scale_min + (scale_max - scale_min) * (scale - scale.min()) / (
            scale.max() - scale.min()
        )
        df_shape_scale_star[f"shape_{event}"] = shape
        df_shape_scale_star[f"scale_{event}"] = scale

    return df_shape_scale_star


def censor_data(
    y, relative_scale, independant=True, X=None, features_censoring_rate=0.2
):
    if relative_scale == 0 or relative_scale is None:
        return y

    if independant:
        scale_censoring = relative_scale * y["duration"].mean()
    else:
        features_impact_censoring = np.abs(np.random.randn(X.shape[1], 1))
        features_impact_censoring = np.where(
            np.random.rand(
                features_impact_censoring.shape[0], features_impact_censoring.shape[1]
            )
            < features_censoring_rate,
            features_impact_censoring,
            0,
        )
        scale_censoring = (X @ features_impact_censoring).values.flatten()
        scale_censoring = (
            (
                1
                + (scale_censoring - scale_censoring.min())
                / (scale_censoring.max() - scale_censoring.min())
            )
            * relative_scale
            * y["duration"].mean()
        )

    censoring = weibull_min.rvs(
        1, scale=scale_censoring, size=y.shape[0], random_state=rng
    )
    y = y.copy()
    y["event"] = np.where(y["duration"] < censoring, y["event"], 0)
    y["duration"] = np.minimum(y["duration"], censoring)
    return y


def complex_data(
    n_events=n_events,
    n_weibull_parameters=2 * n_events,
    n_samples=3_000,
    base_scale=1_000,
    n_features=10,
    features_rate=0.3,
    degree_interaction=2,
    relative_scale=1.5,
    independant=True,
    features_censoring_rate=0.2,
    return_uncensored_data=False,
):
    # Create features given to the model as X and then creating the interactions
    df_features = pd.DataFrame(np.random.randn(n_samples, n_features))
    df_features.columns = [f"feature_{i}" for i in range(n_features)]

    df_shape_scale_star = compute_shape_and_scale(
        df_features, n_weibull_parameters, features_rate, n_events, degree_interaction
    )
    # Throw durations from a weibull distribution with scale and shape as the parameters
    event_durations = []
    for event in range(n_events):
        shape = df_shape_scale_star[f"shape_{event}"]
        scale = df_shape_scale_star[f"scale_{event}"]
        durations = weibull_min.rvs(shape, scale=scale * base_scale, random_state=rng)
        event_durations.append(durations)

    # Creating the target tabular
    event_durations = np.asarray(event_durations)
    duration_argmin = np.argmin(event_durations, axis=0)
    y = pd.DataFrame(
        dict(
            event=duration_argmin + 1,
            duration=event_durations[duration_argmin, np.arange(n_samples)],
        )
    )
    y_censored = censor_data(
        y,
        relative_scale,
        independant=independant,
        X=df_features,
        features_censoring_rate=features_censoring_rate,
    )
    if return_uncensored_data:
        return df_features, y_censored, y
    return df_features, y_censored
