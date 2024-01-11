# %%
import json
import pandas as pd
from copy import deepcopy
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

from main import DATASET_GRID

sns.set_style("darkgrid")


def aggregate_result(path_session):
    data = []
    for path_profile in Path(path_session).glob("*"):
        results = json.load(open(path_profile / "result.json"))
        dataset_params = json.load(open(path_profile / "dataset_params.json"))
        data.append({**dataset_params, **results})
    return pd.DataFrame(data)


def _make_query(x_col, query_params):
    _check_query_params(x_col, query_params)
    query = []
    for k, v in query_params.items():
        query.append(f"({k} == {v})")
    return " & ".join(query)


def _check_query_params(x_col, query_params):
    dataset_grid = deepcopy(DATASET_GRID)
    dataset_grid.pop(x_col)
    xor = set(query_params) ^ set(dataset_grid)

    # check keys
    if len(xor) > 0:
        raise ValueError(
            "'query_params' must have the same keys than 'DATASET_GRID'"
            f"but {xor} differ."
        )

    # check values exist in DATASET_GRID
    for k, v in query_params.items():
        if v not in dataset_grid[k]:
            raise ValueError(f"Options for {k} are {dataset_grid[k]}, got {v}.")


class ResultDisplayer:
    def __init__(self, path_session, estimator_names):
        if not Path(path_session).exists():
            raise FileNotFoundError(f"{path_session} doesn't exist.")
        self.path_session = Path(path_session)
        self.df = aggregate_result(self.path_session)
        self.estimator_names = [
            name
            for name in self.df["estimator_name"].unique()
            if name in estimator_names
        ]

    def plot_memory_time(self, x_col, query_params):
        self._plot_lines(
            x_col=x_col,
            y_col="memory_peak",
            query_params=query_params,
            ylabel="RAM (MiB)",
            title="RAM peak",
        )
        self._plot_lines(
            x_col=x_col,
            y_col="fit_time",
            query_params=query_params,
            ylabel="time (s)",
            title="Time to fit",
        )
        self._plot_lines(
            x_col=x_col,
            y_col="test_time",
            query_params=query_params,
            ylabel="time (s)",
            title="Time to predict",
        )

    def plot_score(self, x_col, query_params):
        self._plot_lines(
            x_col=x_col,
            y_col="train_score",
            query_params=query_params,
            ylabel="NIBS",
            title="Train score",
        )
        self._plot_lines(
            x_col=x_col,
            y_col="test_score",
            query_params=query_params,
            ylabel="NIBS",
            title="Test score",
        )

    def _plot_lines(self, x_col, y_col, query_params, ylabel, title):
        query = _make_query(x_col, query_params)
        df = self.df.query(query)

        fig, ax = plt.subplots()
        for estimator_name in self.estimator_names:
            df_est = df.query("estimator_name == @estimator_name").sort_values(x_col)
            mean = df_est[f"{y_col}_mean"]
            std = df_est[f"{y_col}_std"]
            ax.plot(df_est[x_col], mean, label=estimator_name, marker="o")
            ax.fill_between(df_est[x_col], mean - std, mean + std, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(ylabel)
        ax.legend()

    # def plot_IPSR(self, x_col, **query_params):

    #     query = _make_query(x_col, query_params)
    #     df = self.df.query(query)

    #     fig, ax = plt.subplot()
    #     for estimator_name in self.estimator_names:
    #         df_est = df.query("estimator_name == @estimator_name")

    #         ax.plot(df_est[x_col], mean, label=estimator_name)
    #         ax.fill_between(df_est[x_col], mean - std, mean + std)
    #     ax.legend();


# %%


def show_results():
    path_session = "2024-01-11"
    estimator_names = ["gbi", "gbmi"]
    displayer = ResultDisplayer(path_session, estimator_names)

    query_params = {
        "n_events": 3,
        "censoring_relative_scale": 1.5,
        "complex_features": True,
        "independent_censoring": True,
    }
    displayer.plot_memory_time(
        x_col="n_samples",
        query_params=query_params,
    )

    query_params = {
        "n_events": 3,
        "n_samples": 10_000,
        "complex_features": True,
        "independent_censoring": True,
    }
    displayer.plot_score(
        x_col="censoring_relative_scale",
        query_params=query_params,
    )


show_results()

# %%
