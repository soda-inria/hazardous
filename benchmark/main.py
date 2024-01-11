import json
import pickle
from pathlib import Path
from itertools import product
from tqdm import tqdm
from datetime import datetime

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous._gb_multi_incidence import GBMultiIncidence
from hazardous._gradient_boosting_incidence import GradientBoostingIncidence

from memory_monitor import MemoryMonitor

GBMI = GBMultiIncidence(show_progressbar=False, n_iter=20)
# GBMI.set_score_request(scale=True, shape=True)

ESTIMATOR_GRID = {
    "gbmi": {
        "estimator": GBMI,
        "param_grid": {
            "learning_rate": [0.01],
            # "n_iter": [50, 100, 200],
            # "max_depth": [3, 5],
            # "max_leaf_nodes": [10, 30, 50],
            # "min_samples_leaf": [10, 50],
        },
    },
    "gbi": {
        "estimator": GradientBoostingIncidence(n_iter=20, show_progressbar=False),
        "param_grid": {
            "learning_rate": [0.01],
        },
    },
}

# Parameters of the make_synthetic_competing_weibull function.
DATASET_GRID = {
    "n_events": [3],
    "n_samples": [1_000, 5_000, 10_000],
    "censoring_relative_scale": [0.8, 1.5, 3],
    "complex_features": [True],
    "independent_censoring": [True, False],
}

PATH_DAILY_SESSION = Path(datetime.now().strftime("%Y-%m-%d"))


def run_all():
    for dataset_param in tqdm(list(product(*DATASET_GRID.values()))):
        dataset_param = dict(zip(DATASET_GRID.keys(), dataset_param))
        print(dataset_param)
        for estimator_name in ESTIMATOR_GRID:
            profile_estimator(estimator_name, dataset_param, seeds=range(1))


def profile_estimator(estimator_name, dataset_params, seeds):
    """Find the best hyper-parameters for a given model and a given dataset."""
    bunch = make_synthetic_competing_weibull(**dataset_params)
    X, y = bunch.X, bunch.y
    # scale_censoring, shape_censoring = bunch.scale_censoring, bunch.shape_censoring

    estimator = ESTIMATOR_GRID[estimator_name]["estimator"]
    param_grid = ESTIMATOR_GRID[estimator_name]["param_grid"]

    hp_search = GridSearchCV(estimator, param_grid, cv=2)
    hp_search.fit(
        X,
        y,
        # scale_censoring=scale_censoring,
        # shape_censoring=shape_censoring,
    )

    best_params = hp_search.best_params_
    best_estimator = hp_search.best_estimator_

    results = cross_val_best_estimator(best_estimator, dataset_params, seeds)
    results["estimator_name"] = estimator_name

    str_params = [str(v) for v in dataset_params.values()]
    str_params = "_".join([estimator_name, *str_params])
    path_profile = PATH_DAILY_SESSION / str_params
    path_profile.mkdir(parents=True, exist_ok=True)

    json.dump(best_params, open(path_profile / "best_params.json", "w"))
    pickle.dump(best_estimator, open(path_profile / "best_estimator.pkl", "wb"))
    json.dump(results, open(path_profile / "result.json", "w"))
    json.dump(dataset_params, open(path_profile / "dataset_params.json", "w"))


def cross_val_best_estimator(best_estimator, dataset_params, seeds):
    peak_memory, fit_time, score_time = [], [], []
    test_score, train_score = [], []
    for seed in seeds:
        monitor = MemoryMonitor()

        dataset_params["random_state"] = seed
        X, y = make_synthetic_competing_weibull(return_X_y=True, **dataset_params)

        results = cross_validate(best_estimator, X, y, return_train_score=True)

        fit_time.append(results["fit_time"])
        score_time.append(results["score_time"])
        train_score.append(results["train_score"])
        test_score.append(results["test_score"])

        monitor.join()
        peak_memory.append(max(monitor.memory_buffer) / 1e6)

    return dict(
        memory_peak_mean=np.mean(peak_memory),
        memory_peak_std=np.std(peak_memory),
        fit_time_mean=np.mean(fit_time),
        fit_time_std=np.std(fit_time),
        test_time_mean=np.mean(score_time),
        test_time_std=np.std(score_time),
        train_score_mean=np.mean(train_score),
        train_score_std=np.std(train_score),
        test_score_mean=np.mean(test_score),
        test_score_std=np.std(test_score),
    )


if __name__ == "__main__":
    run_all()
