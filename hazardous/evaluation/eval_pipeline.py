# %%
import itertools

import psutil
from sklearn.model_selection import GridSearchCV, train_test_split

import hazardous.data._competing_weibull as competing_w
from hazardous._comp_risks_loss import MultinomialBinaryLoss
from hazardous._gb_multi_incidence import GBMultiIncidence

SEED_RANGE = range(1)
SAMPLES_SIZE = [3_000, 10_000]
CENSORING_RATE = [0, 50]
COMPLEX_FEATURES = [False, True]
NUMBER_EVENTS = [1, 3]
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


def get_memory_info():
    process = psutil.Process()
    return process.memory_info().rss / (2**20)


def eval_model(
    estimator,
    param_grid,
    n_events=3,
    n_samples=3_000,
    censoring_relative_scale=1.5,
    independent_censoring=True,
    features_censoring_rate=0.2,
    complex_features=False,
    seed=0,
):
    bunch = competing_w.make_synthetic_competing_weibull(
        n_events=n_events,
        n_samples=n_samples,
        censoring_relative_scale=censoring_relative_scale,
        complex_features=complex_features,
        independent_censoring=independent_censoring,
        features_censoring_rate=features_censoring_rate,
        feature_rounding=None,
        target_rounding=None,
        base_scale=1_000,
        n_features=10,
        features_rate=0.3,
        degree_interaction=2,
        random_state=seed,
    )

    # params of the evaluation
    params_evals = dict(
        n_events=n_events,
        n_samples=n_samples,
        censoring_relative_scale=censoring_relative_scale,
        complex_features=complex_features,
        independent_censoring=independent_censoring,
        features_censoring_rate=features_censoring_rate,
        seed=seed,
    )

    X = bunch["data"]
    y_censored = bunch["target"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_censored, random_state=seed
    )
    # fit
    clf = GridSearchCV(
        estimator,
        param_grid,
        cv=5,
        scoring=MultinomialBinaryLoss(),
        refit=True,
    )
    clf.fit(X_train, y_train)

    # params of the best estimator
    time_to_predict = clf.cv_results_["mean_fit_time"].mean()
    best_params = clf.best_params_

    # store results
    results = dict(
        bunch=bunch,
        time_to_predict=time_to_predict,
        best_params=best_params,
        params_evals=params_evals,
    )
    return results


# if __name__ == "__main__":
seed = 0

estimator = GBMultiIncidence(loss="inll", random_state=seed, show_progressbar=False)
param_grid = dict(
    n_iter=[50, 100, 200],
    learning_rate=[0.01, 0.03, 0.05],
    max_depth=[3, 5],
    max_leaf_nodes=[10, 30, 50],
    min_samples_leaf=[10, 50],
)

NUMBER_EVENTS = [1, 3]
SAMPLES_SIZE = [3_000, 10_000, 100_000]
CENSORING_RATE = [0, 0.8, 1.5]
COMPLEX_FEATURES = [False, True]
INDEPENDANT_CENSORING = [True, False]
# %%
for (
    n_events,
    n_samples,
    censoring_relative_scale,
    complex_features,
    independant_censoring,
) in itertools.product(
    NUMBER_EVENTS,
    SAMPLES_SIZE,
    CENSORING_RATE,
    COMPLEX_FEATURES,
    INDEPENDANT_CENSORING,
):
    results = eval_model(
        estimator,
        param_grid,
        n_events=n_events,
        n_samples=n_samples,
        censoring_relative_scale=censoring_relative_scale,
        independent_censoring=independant_censoring,
        features_censoring_rate=0.2,
        complex_features=complex_features,
        seed=seed,
    )
    name_file = (
        f"results_{n_events}-{n_samples}-{censoring_relative_scale}"
        + f"-{complex_features}-{independant_censoring}.parquet"
    )
    results.to_parquet(name_file)
