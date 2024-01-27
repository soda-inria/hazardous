# %%
import json
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
from joblib import load
from sklearn.model_selection import train_test_split

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.data._seer import load_seer

from main import SEER_PATH, SEED

sns.set_style(
    style="white",
)
sns.set_context("paper")
sns.set_palette("colorblind")


def aggregate_result(path_session_dataset, estimator_names):
    data = []

    if not Path(path_session_dataset).exists():
        raise ValueError(f"{path_session_dataset} doesn't exist.")

    for path_profile in Path(path_session_dataset).glob("*"):
        results = json.load(open(path_profile / "cv_results.json"))
        estimator_name = results["estimator_name"]
        if estimator_name in estimator_names:
            dataset_params = json.load(open(path_profile / "dataset_params.json"))
            estimator = load(path_profile / "best_estimator.joblib")
            estimator = {"estimator": estimator}
            data.append({**dataset_params, **results, **estimator})

    return pd.DataFrame(data)


def load_dataset(dataset_name, data_params, random_state=None):
    if dataset_name == "weibull":
        return make_synthetic_competing_weibull(
            return_X_y=False,
            random_state=random_state,
            **data_params,
        )

    elif dataset_name == "seer":
        bunch = load_seer(
            input_path=SEER_PATH,
            survtrace_preprocessing=True,
            return_X_y=False,
        )
        _, X_test, _, y_test = train_test_split(
            bunch.X, bunch.y, test_size=0.3, random_state=SEED
        )
        bunch.X, bunch.y = X_test, y_test
        return bunch
    else:
        raise ValueError(f"Got {dataset_name} instead of ('seer', 'weibull').")


def make_time_grid(duration, n_steps=100):
    t_min, t_max = duration.min(), duration.max()
    return np.linspace(t_min, t_max, n_steps)


def make_query(data_params):
    query = []
    for k, v in data_params.items():
        if isinstance(v, str):
            v = f"'{v}'"
        query.append(f"({k} == {v})")
    return " & ".join(query)


def get_estimator(df, estimator_name):
    df_est = df.query("estimator_name == @estimator_name")
    if df_est.shape[0] != 1:
        raise ValueError(f"selection should be a single row, got {df_est}.")
    row = df_est.iloc[0]

    return row["estimator"]


def get_kind(data_params):
    if "independent_censoring" in data_params:
        return "independent" if data_params["independent_censoring"] else "dependent"
    return ""
