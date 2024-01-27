import json
from pathlib import Path
from itertools import product
from datetime import datetime
import pandas as pd

from joblib import delayed, Parallel, dump
from sklearn.model_selection import GridSearchCV, train_test_split

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.data._seer import (
    load_seer,
    CATEGORICAL_COLUMN_NAMES,
    NUMERIC_COLUMN_NAMES,
)
from hazardous._gb_multi_incidence import GBMultiIncidence
from hazardous.survtrace._encoder import SurvFeatureEncoder
from hazardous.utils import CumulativeIncidencePipeline


# Enable oracle scoring for GridSearchCV
# GBMI.set_score_request(scale=True, shape=True)

gbmi_10 = CumulativeIncidencePipeline(
    [
        ("surv_feature_encoder", SurvFeatureEncoder()),
        ("gb_multi_incidence", GBMultiIncidence(n_iter=10, show_progressbar=True)),
    ]
)

gbmi_20 = CumulativeIncidencePipeline(
    [
        ("surv_feature_encoder", SurvFeatureEncoder()),
        ("gb_multi_incidence", GBMultiIncidence(n_iter=20, show_progressbar=True)),
    ]
)

ESTIMATOR_GRID = {
    "gbmi_10": {
        "estimator": gbmi_10,
        "param_grid": {
            "gb_multi_incidence__learning_rate": [0.01],
            # "n_iter": [50, 100, 200],
            # "max_depth": [3, 5],
            # "max_leaf_nodes": [10, 30, 50],
            # "min_samples_leaf": [10, 50],
        },
    },
    "gbmi_20": {
        "estimator": gbmi_20,
        "param_grid": {
            "gb_multi_incidence__learning_rate": [0.01],
        },
    },
}

# Parameters of the make_synthetic_competing_weibull function.
DATASET_GRID = {
    "n_events": [3],
    "n_samples": [1_000, 5_000, 10_000, 20_000, 50_000],
    "censoring_relative_scale": [0.8, 1.5, 2.5],
    "complex_features": [True],
    "independent_censoring": [True, False],
}

PATH_DAILY_SESSION = Path(datetime.now().strftime("%Y-%m-%d"))

SEER_PATH = "../hazardous/data/seer_cancer_cardio_raw_data.txt"
SEED = 0


def run_all_synthetic_datasets():
    grid_dataset_params = list(product(*DATASET_GRID.values()))

    parallel = Parallel(n_jobs=-1)
    parallel(
        delayed(run_synthetic_dataset)(dataset_params)
        for dataset_params in grid_dataset_params
    )


def run_synthetic_dataset(dataset_params):
    dataset_params = dict(zip(DATASET_GRID.keys(), dataset_params))
    data_bunch = make_synthetic_competing_weibull(**dataset_params)
    for estimator_name in ESTIMATOR_GRID:
        run_estimator(
            estimator_name,
            data_bunch,
            dataset_name="weibull",
            dataser_params=dataset_params,
        )


def run_seer():
    data_bunch = load_seer(
        SEER_PATH,
        survtrace_preprocessing=True,
        return_X_y=False,
    )
    X, y = data_bunch.X, data_bunch.y
    column_names = CATEGORICAL_COLUMN_NAMES + NUMERIC_COLUMN_NAMES
    data_bunch.X = data_bunch.X[column_names]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=SEED)
    data_bunch.X, data_bunch.y = X_train, y_train

    parallel = Parallel(n_jobs=-1)
    parallel(
        delayed(run_estimator)(
            estimator_name,
            data_bunch,
            dataset_name="seer",
            dataset_params={},
        )
        for estimator_name in ESTIMATOR_GRID
    )


def run_estimator(estimator_name, data_bunch, dataset_name, dataset_params):
    """Find the best hyper-parameters for a given model and a given dataset."""

    print(f"{estimator_name}\n{dataset_params}")
    X, y = data_bunch.X, data_bunch.y
    # scale_censoring = data_bunch.scale_censoring
    # shape_censoring = data_bunch.shape_censoring
    estimator = ESTIMATOR_GRID[estimator_name]["estimator"]
    param_grid = ESTIMATOR_GRID[estimator_name]["param_grid"]

    hp_search = GridSearchCV(estimator, param_grid, cv=2, return_train_score=True)
    hp_search.fit(
        X,
        y,
        # scale_censoring=scale_censoring,
        # shape_censoring=shape_censoring,
    )

    best_params = hp_search.best_params_
    best_estimator = hp_search.best_estimator_

    cols = [
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]
    best_results = pd.DataFrame(hp_search.cv_results_)[cols]
    best_results = best_results.iloc[hp_search.best_index_].to_dict()
    best_results["estimator_name"] = estimator_name

    best_estimator.fit(X, y)

    # hack for benchmarks
    best_estimator.y_train = y

    str_params = [str(v) for v in dataset_params.values()]
    str_params = "_".join([estimator_name, *str_params])
    path_profile = PATH_DAILY_SESSION / dataset_name / str_params
    path_profile.mkdir(parents=True, exist_ok=True)

    json.dump(best_params, open(path_profile / "best_params.json", "w"))
    dump(best_estimator, path_profile / "best_estimator.joblib")
    json.dump(best_results, open(path_profile / "cv_results.json", "w"))
    json.dump(dataset_params, open(path_profile / "dataset_params.json", "w"))


if __name__ == "__main__":
    run_seer()
