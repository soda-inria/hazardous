# %%
import warnings
import json
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from IPython.display import display

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import load
from lifelines import AalenJohansenFitter

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    brier_score_incidence,
    brier_score_incidence_oracle,
)
from hazardous.metrics._concordance import concordance_index_ipcw

from main import DATASET_GRID

sns.set_style("whitegrid")


def aggregate_result(path_session):
    data = []
    for path_profile in Path(path_session).glob("*"):
        results = json.load(open(path_profile / "cv_results.json"))
        dataset_params = json.load(open(path_profile / "dataset_params.json"))
        estimator = load(path_profile / "best_estimator.joblib")
        estimator = {"estimator": estimator}
        data.append({**dataset_params, **results, **estimator})
    return pd.DataFrame(data)


def make_time_grid(duration, n_steps=100):
    t_min, t_max = duration.min(), duration.max()
    return np.linspace(t_min, t_max, n_steps)


def _make_query(data_params, x_col=None):
    data_params = deepcopy(data_params)
    _check_data_params(data_params, x_col)
    query = []
    for k, v in data_params.items():
        query.append(f"({k} == {v})")
    return " & ".join(query)


def _check_data_params(data_params, x_col=None):
    data_grid = deepcopy(DATASET_GRID)
    if x_col is not None:
        data_params.pop(x_col, None)
        data_grid.pop(x_col)
    xor = set(data_params) ^ set(data_grid)

    # check keys
    if len(xor) > 0:
        raise ValueError(
            "'data_params' must have the same keys than 'DATASET_GRID' "
            f"but {xor} differ."
        )

    # check values exist in DATASET_GRID
    for k, v in data_params.items():
        if v not in data_grid[k]:
            raise ValueError(f"Options for {k} are {data_grid[k]}, got {v}.")


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

    def plot_memory_time(self, data_params, x_col):
        query = _make_query(data_params, x_col)
        df = self.df.query(query)

        fig, ax = plt.subplots()

        sns.barplot(
            df,
            x=x_col,
            y="peak_memory",
            hue="estimator_name",
            ax=ax,
        )
        ax.set(
            title="RAM peak",
            ylabel="RAM (GiB)",
        )

        fig, axes = plt.subplots(ncols=2, sharey=True)

        sns.barplot(df, x=x_col, y="mean_fit_time", hue="estimator_name", ax=axes[0])
        axes[0].set(
            title="Time to fit",
            ylabel="time(s)",
        )

        sns.barplot(df, x=x_col, y="mean_score_time", hue="estimator_name", ax=axes[1])
        axes[1].set(
            title="Time to test",
            ylabel=None,
        )

    def plot_IPSR(self, data_params):
        x_cols = ["n_samples", "censoring_relative_scale"]
        fig, axes = plt.subplots(ncols=2)

        for ax, x_col in zip(axes, x_cols):
            query = _make_query(data_params, x_col=x_col)
            df = self.df.query(query)

            for estimator_name in tqdm(self.estimator_names):
                df_est = df.query("estimator_name == @estimator_name")
                self._plot_IPSR_estimator(
                    df_est,
                    x_col,
                    data_params,
                    ax=ax,
                )
            ax.set(
                title="IBS",
                xlabel=x_col,
                ylabel=None,
            )
            ax.legend()

    @staticmethod
    def _plot_IPSR_estimator(df, x_col, data_params, ax):
        x_col_params, all_ipsr = [], []
        for x_col_param, df_group in df.groupby(x_col):
            data_params[x_col] = x_col_param
            X, y = make_synthetic_competing_weibull(**data_params, return_X_y=True)

            time_grid = make_time_grid(y["duration"])

            if df_group.shape[0] != 1:
                raise ValueError(f"selection should be a single row, got {df_group}.")

            row = df_group.iloc[0]
            estimator = row["estimator"]
            y_train = estimator.y_train  # hack for benchmarks

            y_pred = estimator.predict_cumulative_incidence(
                X,
                times=time_grid,
            )
            event_specific_ipsr = []
            for idx in range(data_params["n_events"]):
                event_specific_ipsr.append(
                    integrated_brier_score_incidence(
                        y_train=y_train,
                        y_test=y,
                        # TODO: remove when removing GBI.
                        y_pred=y_pred[idx + 1] if y_pred.ndim == 3 else y_pred,
                        times=time_grid,
                        event_of_interest=idx + 1,
                    )
                )
            x_col_params.append(x_col_param)
            all_ipsr.append(np.mean(event_specific_ipsr))

        ax.plot(
            x_col_params,
            all_ipsr,
            label=row["estimator_name"],
            marker="o",
        )

    def plot_PSR(self, data_params):
        query = _make_query(data_params, x_col=None)
        df = self.df.query(query)

        bunch = make_synthetic_competing_weibull(**data_params, return_X_y=False)
        (
            X,
            y,
            scale_censoring,
            shape_censoring,
        ) = (
            bunch.X,
            bunch.y,
            bunch.scale_censoring,
            bunch.shape_censoring,
        )
        time_grid = make_time_grid(y["duration"])

        n_events = data_params["n_events"]
        fig, axes = plt.subplots(figsize=(12, 4), ncols=n_events, sharey=True)

        censoring_fraction = (y["event"] == 0).mean()
        kind = "independent" if data_params["independent_censoring"] else "dependent"
        fig.suptitle(
            f"Time-varying Brier score ({censoring_fraction:.1%} {kind} censoring)"
        )

        for estimator_id, estimator_name in enumerate(self.estimator_names, 1):
            df_est = df.query("estimator_name == @estimator_name")

            if df_est.shape[0] != 1:
                raise ValueError(f"selection should be a single row, got {df_est}.")

            row = df_est.iloc[0]
            estimator = row["estimator"]
            y_pred = estimator.predict_cumulative_incidence(X, times=time_grid)

            for event_id, ax in enumerate(axes, 1):
                debiased_bs_scores = brier_score_incidence_oracle(
                    y_train=y,
                    y_test=y,
                    y_pred=y_pred[event_id] if y_pred.ndim == 3 else y_pred,  # TODO
                    times=time_grid,
                    shape_censoring=shape_censoring,
                    scale_censoring=scale_censoring,
                    event_of_interest=event_id,
                )

                bs_scores = brier_score_incidence(
                    y_train=y,
                    y_test=y,
                    y_pred=y_pred[event_id] if y_pred.ndim == 3 else y_pred,  # TODO
                    times=time_grid,
                    event_of_interest=event_id,
                )

                (line,) = ax.plot(
                    time_grid,
                    bs_scores,
                    label=f"{estimator_name} (estimated PSR)",
                )
                ax.plot(
                    time_grid,
                    debiased_bs_scores,
                    label=f"{estimator_name} (oracle PSR)",
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.5,
                )
                ax.set_title(f"event {event_id}")
        axes[0].legend()

    def plot_marginal_incidence(self, data_params):
        query = _make_query(data_params, x_col=None)
        df = self.df.query(query)

        bunch = make_synthetic_competing_weibull(**data_params, return_X_y=False)
        X, y, y_uncensored = bunch.X, bunch.y, bunch.y_uncensored
        time_grid = make_time_grid(y["duration"])

        n_events = data_params["n_events"]
        fig, axes = plt.subplots(figsize=(12, 4), ncols=n_events, sharey=True)

        censoring_fraction = (y["event"] == 0).mean()
        kind = "independent" if data_params["independent_censoring"] else "dependent"
        fig.suptitle(
            "Cause-specific cumulative incidence functions"
            f" ({censoring_fraction:.1%} {kind} censoring)"
        )

        for estimator_id, estimator_name in enumerate(self.estimator_names, 1):
            df_est = df.query("estimator_name == @estimator_name")

            if df_est.shape[0] != 1:
                raise ValueError(f"selection should be a single row, got {df_est}.")

            row = df_est.iloc[0]
            estimator = row["estimator"]
            y_pred = estimator.predict_cumulative_incidence(X, times=time_grid)

            for event_id, ax in enumerate(axes, 1):
                ax.plot(
                    time_grid,
                    (y_pred[event_id] if y_pred.ndim == 3 else y_pred).mean(axis=0),
                    label=estimator_name,
                )
                ax.set(title=f"event {event_id}")
                ax.legend()

        for event_id, ax in enumerate(axes, 1):
            with warnings.catch_warnings(record=True):
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                aj = AalenJohansenFitter(calculate_variance=False).fit(
                    durations=y["duration"],
                    event_observed=y["event"],
                    event_of_interest=event_id,
                )
                aj.plot(label="AalenJohansen", ax=ax, color="k")

                aj.fit(
                    durations=y_uncensored["duration"],
                    event_observed=y_uncensored["event"],
                    event_of_interest=event_id,
                )
                aj.plot(
                    label="AalenJohansen uncensored",
                    ax=ax,
                    color="k",
                    linestyle="--",
                )

        for ax in [axes[1], axes[2]]:
            ax.legend().remove()

    def plot_individual_incidence(self, data_params, sample_ids=2):
        if isinstance(sample_ids, int):
            sample_ids = list(range(sample_ids))

        query = _make_query(data_params, x_col=None)
        df = self.df.query(query)

        bunch = make_synthetic_competing_weibull(
            **data_params,
            return_X_y=False,
            random_state=None,
        )
        X, y = bunch.X, bunch.y
        time_grid = make_time_grid(y["duration"])

        n_events = data_params["n_events"]
        fig, axes = plt.subplots(
            figsize=(10, 8),
            nrows=len(sample_ids),
            ncols=n_events,
            sharey=True,
        )
        fig.suptitle(f"Probability of incidence for {len(sample_ids)} samples")

        for estimator_name in self.estimator_names:
            df_est = df.query("estimator_name == @estimator_name")
            if df_est.shape[0] != 1:
                raise ValueError(f"selection should be a single row, got {df_est}.")

            row = df_est.iloc[0]
            estimator = row["estimator"]

            y_pred = estimator.predict_cumulative_incidence(
                X.loc[sample_ids], times=time_grid
            )
            y_test = y.loc[sample_ids]

            for row_idx in range(len(sample_ids)):
                y_sample = y_test.iloc[row_idx]
                sample_event, sample_duration = y_sample["event"], y_sample["duration"]
                for col_idx in range(n_events):
                    event_id = col_idx + 1
                    ax = axes[row_idx, col_idx]
                    ax.plot(
                        time_grid,
                        (
                            y_pred[event_id][row_idx]
                            if y_pred.ndim == 3
                            else y_pred[row_idx]
                        ),  # TODO
                        label=estimator_name,
                    )

                    if sample_event == event_id:
                        ax.axvline(x=sample_duration, color="r", linestyle="--")

                    if row_idx == 0:
                        ax.set_title(f"event {event_id}")
                        if col_idx == 0:
                            ax.legend()
        plt.tight_layout()

    def print_table_metrics(self, data_params):
        query = _make_query(data_params, x_col=None)
        df = self.df.query(query)

        bunch = make_synthetic_competing_weibull(**data_params, return_X_y=False)
        X, y = bunch.X, bunch.y
        time_grid = make_time_grid(y["duration"])

        results = []

        for estimator_name in self.estimator_names:
            df_est = df.query("estimator_name == @estimator_name")
            if df_est.shape[0] != 1:
                raise ValueError(f"selection should be a single row, got {df_est}.")

            row = df_est.iloc[0]
            estimator = row["estimator"]

            y_train = estimator.y_train
            y_pred = estimator.predict_cumulative_incidence(X, times=time_grid)

            truncation_quantiles = [0.25, 0.5, 0.75]
            truncation_times = np.quantile(time_grid, truncation_quantiles)
            truncation_indices = np.searchsorted(time_grid, truncation_times)

            for event_id in range(1, data_params["n_events"] + 1):
                ibs = integrated_brier_score_incidence(
                    y_train=y_train,
                    y_test=y,
                    y_pred=y_pred[event_id] if y_pred.ndim == 3 else y_pred,  # TODO
                    times=time_grid,
                    event_of_interest=event_id,
                )

                for truncation_idx, truncation_time, truncation_q in zip(
                    truncation_indices, truncation_times, truncation_quantiles
                ):
                    y_pred_at_t = (
                        y_pred[event_id][:, truncation_idx]
                        if y_pred.ndim == 3
                        else y_pred[:, truncation_idx]  # TODO
                    )
                    ct_index, _, _, _, _ = concordance_index_ipcw(
                        y_train,
                        y,
                        y_pred_at_t,
                        tau=truncation_time,
                    )
                    results.append(
                        dict(
                            estimator_name=estimator_name,
                            ibs=ibs,
                            event=event_id,
                            truncation_q=truncation_q,
                            ct_index=ct_index,
                        )
                    )
        results = pd.DataFrame(results)
        results_ibs = (
            results[["estimator_name", "event", "ibs"]]
            .drop_duplicates()
            .pivot(
                index="estimator_name",
                columns="event",
                values="ibs",
            )
        )
        _ = results.pop("ibs")
        results_ct_index = results.sort_values(["truncation_q", "event"]).pivot(
            index="estimator_name",
            columns=["truncation_q", "event"],
            values="ct_index",
        )

        print("IBS")
        display(results_ibs)
        print("\n")
        print("Ct-index")
        display(results_ct_index)


# %%

path_session = "2024-01-14"
estimator_names = ["gbmi_10", "gbmi_20"]
displayer = ResultDisplayer(path_session, estimator_names)

data_params = {
    "n_events": 3,
    "complex_features": True,
    "censoring_relative_scale": 1.5,
    "independent_censoring": True,
}
displayer.plot_memory_time(
    data_params=data_params,
    x_col="n_samples",
)

# %%

displayer.plot_IPSR(
    data_params=data_params,
)

# %%

data_params.update(
    {
        "n_samples": 10_000,
        "censoring_relative_scale": 1.5,
        "independent_censoring": True,
    }
)
displayer.plot_PSR(
    data_params=data_params,
)

# %%

data_params["independent_censoring"] = False
displayer.plot_PSR(
    data_params=data_params,
)

# %%
data_params["independent_censoring"] = True
displayer.plot_marginal_incidence(data_params)

# %%

data_params["independent_censoring"] = False
displayer.plot_marginal_incidence(data_params)

# %%

data_params.update(
    {
        "n_samples": 10_000,
        "censoring_relative_scale": 1.5,
        "independent_censoring": True,
    }
)
displayer.plot_individual_incidence(data_params, sample_ids=2)
# %%

data_params.update(
    {
        "n_samples": 10_000,
        "censoring_relative_scale": 1.5,
        "independent_censoring": True,
    }
)
displayer.print_table_metrics(data_params=data_params)
# %%
