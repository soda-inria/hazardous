from itertools import cycle

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import weibull_min
from sklearn.datasets._base import Bunch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.utils import check_random_state

DEFAULT_SHAPE_RANGES = (
    (1.0, 3.5),
    (1.0, 1.0),
    (3.0, 6.0),
)

DEFAULT_SCALE_RANGES = (
    (1, 7),
    (1, 15),
    (1.5, 5),
)


def _censor(
    y_uncensored,
    independent_censoring=True,
    X=None,
    features_censoring_rate=0.2,
    censoring_relative_scale=1.5,
    random_state=0,
):
    """Apply right censoring to y_uncensored by sampling from a Weibull distribution.

    Parameters
    ----------
    y_uncensored : pandas.DataFrame of shape (n_samples, 2)
        The input dataframe with columns 'event' and 'duration'.

    independent_censoring : bool, default=True
        Whether the censoring is independent of the covariates X or not.

        * If set to True, the scale and shape parameters are set using
          baseline scalars.
        * If set to False and X is defined, the scale and shape parameters
          are obtained using the compute_shape_and_scale function.

    X : pandas.DataFrame of shape (n_samples, n_features), default=None
        Only used when independent_censoring is set to True.
        Passed to compute_shape_and_scale as input data to generate the shape
        and scale parameters.

    features_censoring_rate : float, default=0.2
        Only used when independent_censoring is set to True.
        Passed to compute_shape_and_scale to define the "dropout" probability of
        the feature weight matrix.

    censoring_relative_scale : float, default=1.5
        The magnitude of censoring to apply.

        * If independent_censoring is set to True, censoring_relative_scale is passed
          to compute_shape_and_scale to define the upper bound of the scale.
          parameter.
        * Otherwise, the scale is a scalar, computed as
          censoring_relative_scale * mean duration.

    random_state : int or instance of RandomState, default=0

    Returns
    -------
    y_censored : pandas.DataFrame of shape (n_samples, 2) or Bunch.
        The input dataframe updated with right-censoring.

    shape_censoring : ndarray of shape (n_samples,)
        The Weibull shape parameter used to generate the censoring distribution.

    scale_censoring : ndarray of shape (n_samples,)
        The Weibull scale parameter used to generate the censoring distribution.
    """
    rng = check_random_state(random_state)
    if censoring_relative_scale is None:
        return Bunch(
            y_censored=y_uncensored,
            shape_censoring=None,
            scale_censoring=None,
        )

    mean_duration = y_uncensored["duration"].mean()

    if independent_censoring:
        scale_censoring = censoring_relative_scale * mean_duration
        shape_censoring = 3

    else:
        SS_star = compute_shape_and_scale(
            X,
            features_rate=features_censoring_rate,
            n_events=1,
            degree_interaction=2,
            shape_ranges=[(0.1, 20.0)],
            scale_ranges=[(0.5, 1.5)],
            random_state=random_state,
        )
        SS_star.columns = ["shape_0", "scale_0"]
        SS_star["scale_0"] *= mean_duration * censoring_relative_scale
        scale_censoring = SS_star["scale_0"]
        shape_censoring = SS_star["shape_0"]

    censoring = weibull_min.rvs(
        shape_censoring,
        scale=scale_censoring,
        size=y_uncensored.shape[0],
        random_state=rng,
    )
    y_censored = y_uncensored.copy()
    y_censored["event"] = np.where(
        y_censored["duration"] < censoring, y_censored["event"], 0
    )
    y_censored["duration"] = np.minimum(y_censored["duration"], censoring)

    return Bunch(
        y_censored=y_censored,
        shape_censoring=shape_censoring,
        scale_censoring=scale_censoring,
    )


def compute_shape_and_scale(
    X,
    features_rate=0.2,
    n_events=3,
    degree_interaction=2,
    shape_ranges=DEFAULT_SHAPE_RANGES,
    scale_ranges=DEFAULT_SCALE_RANGES,
    random_state=0,
):
    """Derive Weibull's shape and scale parameters from covariates X.

    The steps are:

    * Transformation of X by using spline then polynomial extensions,
      of shape (n_samples, n_features_transformed)
    * Generation of W_star by sampling from a standard distribution,
      of shape (n_features_transformed, n_weibull_params)
    * Sparsifying of W_star by uniformly setting features to 0.
    * Matrix multiplication between X_trans and W_star to obtain
      the scale and shape parameters SS_star:

    .. math::

        SS_star = X_trans . W_star

    * Standardization of SS_star and rescaling to the desired parameters intervals.

    Parameters
    ----------
    TODO

    Returns
    -------
    SS_star : pandas.DataFrame of shape (n_samples, n_weibull_params)
        The scale and shape matrix to sample individual duration from using a
        Weibull distribution.
    """

    rng = check_random_state(random_state)

    # Adding interactions between features
    preprocessor = make_pipeline(
        SplineTransformer(n_knots=3),
        PolynomialFeatures(
            degree=degree_interaction, interaction_only=True, include_bias=False
        ),
    )
    preprocessor.set_output(transform="pandas")
    X_trans = preprocessor.fit_transform(X)

    # Create masked matrix with the interactions
    n_weibull_parameters = 2 * n_events
    W_star = rng.randn(X_trans.shape[1], n_weibull_parameters)

    # 1 - feature_rate % of marginal features and interacted features
    # are set to 0
    W_star = _sparsify(W_star, X_trans.columns, features_rate, rng)

    # Computation of the true values of shape and scale
    SS_star = X_trans.values @ W_star
    column_names = [f"shape_{i}" for i in range(1, n_events + 1)] + [
        f"scale_{i}" for i in range(1, n_events + 1)
    ]
    SS_star = pd.DataFrame(SS_star, columns=column_names)

    # Rescaling of these values to stay in the chosen range
    return rescale_params(SS_star, n_events, shape_ranges, scale_ranges)


def _sparsify(W_star, column_names, features_rate, rng):
    """Apply uniform sparsity to W_star by setting 1 - feature_rate % of features to 0.

    The W_star matrix is split across lines corresponding to marginal and
    interaction features, to ensure that the same degree of sparsity is applied
    to both, i.e. we can't sparsify the marginal features only.

    Parameters
    ----------
    W_star : ndarray of shape (n_features, n_weibull_parameters)
        The weight matrix representing the linear combination between input
        features.

    column_names : list of str, of length (n_features)
        The column names of the covariates X, to be split between marginal and
        interaction.

    feature_rate : float,
        The probability threshold under which a feature is kept, or set to 0
        otherwise.

    rng : int or instance of RandomState

    Returns
    -------
    W_star : ndarray of shape (n_features, n_weibull_parameters)
        The sparsified input matrix.
    """

    # Feature names without " " are marginal, otherwise they are interactions.
    marginal_indices, interaction_indices = [], []
    for idx, col in enumerate(column_names):
        if len(col.split(" ")) == 1:
            marginal_indices.append(idx)
        else:
            interaction_indices.append(idx)

    sparse_mask = rng.rand(*W_star[marginal_indices, :].shape) < features_rate
    W_star[marginal_indices, :] = W_star[marginal_indices, :] * sparse_mask

    sparse_mask = rng.rand(*W_star[interaction_indices, :].shape) < features_rate
    W_star[interaction_indices, :] = W_star[interaction_indices, :] * sparse_mask

    return W_star


def rescale_params(SS_star, n_events, shape_ranges, scale_ranges):
    for event_id, shape_range, scale_range in zip(
        range(1, n_events + 1), cycle(shape_ranges), cycle(scale_ranges)
    ):
        shape_min, shape_max = shape_range
        scale_min, scale_max = scale_range
        SS_star[f"shape_{event_id}"] = _rescale_params(
            SS_star[[f"shape_{event_id}"]], shape_min, shape_max
        )
        SS_star[f"scale_{event_id}"] = _rescale_params(
            SS_star[[f"scale_{event_id}"]], scale_min, scale_max
        )
    return SS_star


def _rescale_params(column_param, param_min, param_max):
    # Rescaling of these values to stay in the chosen range
    # Expit is the logistic sigmoid function.
    scaler = StandardScaler()
    column_param = (
        param_min + (param_max - param_min) * expit(scaler.fit_transform(column_param))
    ).flatten()
    return column_param


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
    """Generate some covariates X and sample event-specific durations from a Weibull \
        distribution derived from X.

    Each event-specific Weibull distribution is defined with a shape and scale
    parameters. These parameters are derived from the input X, using a sparse weight
    matrix to linearly combine a polynomial expansion of splines of X.
    See the compute_shape_and_scale function for more details.

    For each event, durations are drawn from their corresponding Weibull distribution.
    The event of interest is the first hit, i.e. the event with the minimum duration.

    Parameters
    ----------
    TODO

    Returns
    -------
    X : pandas.DataFrame of shape (n_samples, n_features)
        Covariate matrix, generated from a standard distribution.

    event_durations : ndarray of shape (n_events, n_samples)
        The generated durations, for each events.

    duration_argmin : ndarray of shape (n_samples,)
        The index of the minimum duration, for each sample.
    """
    rng = check_random_state(random_state)
    # Create features given to the model as X and then creating the interactions
    columns = [f"feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=columns,
    )

    SS_star = compute_shape_and_scale(
        X,
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
        shape = SS_star[f"shape_{event}"]
        scale = SS_star[f"scale_{event}"]
        durations = weibull_min.rvs(shape, scale=scale * base_scale, random_state=rng)
        event_durations.append(durations)

    # Creating the target tabular
    event_durations = np.asarray(event_durations)
    duration_argmin = np.argmin(event_durations, axis=0)
    return X, event_durations, duration_argmin


def make_synthetic_competing_weibull(
    n_events=3,
    n_samples=3_000,
    base_scale=1_000,
    n_features=10,
    features_rate=0.3,
    degree_interaction=2,
    independent_censoring=True,
    features_censoring_rate=0.2,
    feature_rounding=2,
    target_rounding=4,
    shape_ranges=DEFAULT_SHAPE_RANGES,
    scale_ranges=DEFAULT_SCALE_RANGES,
    censoring_relative_scale=1.5,
    random_state=0,
    complex_features=False,
    return_X_y=False,
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
    We return all of the durations thrown and the features.
    """
    if complex_features:
        X, event_durations, duration_argmin = make_complex_features_with_sparse_matrix(
            n_events=n_events,
            n_samples=n_samples,
            base_scale=base_scale,
            shape_ranges=shape_ranges,
            scale_ranges=scale_ranges,
            n_features=n_features,
            features_rate=features_rate,
            degree_interaction=degree_interaction,
            random_state=random_state,
        )
    else:
        X, event_durations, duration_argmin = make_simple_features(
            n_events=n_events,
            n_samples=n_samples,
            base_scale=base_scale,
            shape_ranges=shape_ranges,
            scale_ranges=scale_ranges,
            random_state=random_state,
        )
    y_uncensored = pd.DataFrame(
        dict(
            event=duration_argmin + 1,
            duration=event_durations[duration_argmin, np.arange(n_samples)],
        )
    )
    censor_bunch = _censor(
        y_uncensored,
        independent_censoring=independent_censoring,
        X=X,
        features_censoring_rate=features_censoring_rate,
        censoring_relative_scale=censoring_relative_scale,
        random_state=random_state,
    )
    y_censored = censor_bunch.y_censored
    shape_censoring = censor_bunch.shape_censoring
    scale_censoring = censor_bunch.scale_censoring

    if feature_rounding is not None:
        X = X.round(feature_rounding)

    if target_rounding is not None:
        y_censored["duration"] = y_censored["duration"].round(target_rounding)
        y_uncensored = y_uncensored.round(target_rounding)

    if return_X_y:
        return X, y_censored

    return Bunch(
        X=X,
        y=y_censored,
        y_uncensored=y_uncensored,
        shape_censoring=shape_censoring,
        scale_censoring=scale_censoring,
    )
