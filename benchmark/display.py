# %%
from abc import ABC, abstractmethod
import warnings
import json
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from IPython.display import display
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from joblib import load
from lifelines import AalenJohansenFitter
from sklearn.model_selection import train_test_split

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.data._seer import load_seer
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence,
    brier_score_incidence,
    brier_score_incidence_oracle,
)
from hazardous.metrics._concordance import concordance_index_ipcw

from main import DATASET_GRID, SEER_PATH, SEED

sns.set_style(
    style="white",
)
sns.set_context("paper")
sns.set_palette("colorblind")


def aggregate_result(path_session_dataset, estimator_names):
    data = []
    for path_profile in Path(path_session_dataset).glob("*"):
        results = json.load(open(path_profile / "cv_results.json"))
        estimator_name = results["estimator_name"]
        if estimator_name in estimator_names:
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
    data_params = _check_data_params(data_params, x_col)
    query = []
    for k, v in data_params.items():
        if isinstance(v, str):
            v = f"'{v}'"
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

    return data_params


def _get_estimator(df, estimator_name):
    df_est = df.query("estimator_name == @estimator_name")
    if df_est.shape[0] != 1:
        raise ValueError(f"selection should be a single row, got {df_est}.")
    row = df_est.iloc[0]

    return row["estimator"]


def _get_kind(data_params):
    if "independent_censoring" in data_params:
        return "independent" if data_params["independent_censoring"] else "dependent"
    return ""


class BaseDisplayer(ABC):
    @abstractmethod
    def __init__(self, path_session, estimator_names, dataset_name):
        path_session_dataset = Path(path_session) / dataset_name
        if not path_session_dataset.exists():
            raise FileNotFoundError(f"{path_session_dataset} doesn't exist.")
        self.path_session = Path(path_session)
        self.estimator_names = estimator_names
        self.dataset_name = dataset_name
        self.df = aggregate_result(self.path_session / dataset_name, estimator_names)
        path_profile = Path(path_session) / f"{dataset_name}_plots/"
        path_profile.mkdir(parents=True, exist_ok=True)
        self.path_profile = path_profile

    def plot_PSR(self, data_params):
        df = self.load_cv_results(data_params)

        bunch = self.load_dataset(data_params, return_X_y=False)
        X, y = bunch.X, bunch.y
        time_grid = make_time_grid(y["duration"])

        n_events = y["event"].nunique() - 1
        fig, axes = plt.subplots(figsize=(12, 4), ncols=n_events, sharey=True)

        kind = _get_kind(data_params)
        censoring_fraction = (y["event"] == 0).mean()
        fig.suptitle(
            f"Time-varying Brier score ({censoring_fraction:.1%} {kind} censoring)"
        )
        sns.despine(fig=fig)

        for estimator_name in self.estimator_names:
            estimator = _get_estimator(df, estimator_name)
            y_pred = self.get_predictions(X, time_grid, estimator, estimator_name)

            for event_id, ax in enumerate(axes, 1):
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

                if hasattr(bunch, "scale_censoring") and hasattr(
                    bunch, "shape_censoring"
                ):
                    debiased_bs_scores = brier_score_incidence_oracle(
                        y_train=y,
                        y_test=y,
                        y_pred=y_pred[event_id] if y_pred.ndim == 3 else y_pred,  # TODO
                        times=time_grid,
                        shape_censoring=bunch.shape_censoring,
                        scale_censoring=bunch.scale_censoring,
                        event_of_interest=event_id,
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
        plt.savefig(self.path_profile / "PSR.pdf", format="pdf")

    def plot_marginal_incidence(self, data_params):
        df = self.load_cv_results(data_params)
        bunch = self.load_dataset(data_params, return_X_y=False)
        X, y = bunch.X, bunch.y
        time_grid = make_time_grid(y["duration"])

        n_events = y["event"].nunique() - 1
        fig, axes = plt.subplots(figsize=(12, 4), ncols=n_events, sharey=True)

        censoring_fraction = (y["event"] == 0).mean()
        kind = _get_kind(data_params)
        fig.suptitle(
            "Cause-specific cumulative incidence functions"
            f" ({censoring_fraction:.1%} {kind} censoring)"
        )

        for estimator_name in self.estimator_names:
            estimator = _get_estimator(df, estimator_name)
            y_pred = self.get_predictions(X, time_grid, estimator, estimator_name)

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

                if hasattr(bunch, "y_uncensored"):
                    y_uncensored = bunch.y_uncensored
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
        plt.savefig(self.path_profile / "marginal_incidence.pdf", format="pdf")

    def plot_individual_incidence(self, data_params, sample_ids=2):
        if isinstance(sample_ids, int):
            sample_ids = list(range(sample_ids))

        df = self.load_cv_results(data_params)
        bunch = self.load_dataset(data_params, return_X_y=False)
        X, y = bunch.X, bunch.y
        time_grid = make_time_grid(y["duration"])

        n_events = y["event"].nunique() - 1
        fig, axes = plt.subplots(
            figsize=(10, 8),
            nrows=len(sample_ids),
            ncols=n_events,
            sharey=True,
        )
        fig.suptitle(f"Probability of incidence for {len(sample_ids)} samples")

        for estimator_name in self.estimator_names:
            estimator = _get_estimator(df, estimator_name)
            y_pred = self.get_predictions(
                X.iloc[sample_ids], time_grid, estimator, estimator_name
            )
            y_test = y.iloc[sample_ids]

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
        plt.savefig(self.path_profile / "individual_incidence.pdf", format="pdf")

    def print_table_metrics(self, data_params):
        df = self.load_cv_results(data_params)
        bunch = self.load_dataset(data_params, return_X_y=False)
        X, y = bunch.X, bunch.y
        time_grid = make_time_grid(y["duration"])
        n_events = y["event"].nunique() - 1

        results = []

        for estimator_name in self.estimator_names:
            estimator = _get_estimator(df, estimator_name)
            y_train = estimator.y_train
            y_pred = self.get_predictions(X, time_grid, estimator, estimator_name)

            truncation_quantiles = [0.25, 0.5, 0.75]
            truncation_times = np.quantile(time_grid, truncation_quantiles)
            truncation_indices = np.searchsorted(time_grid, truncation_times)

            for event_id in range(1, n_events + 1):
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
                    n_subsamples = 10_000
                    ct_index, _, _, _, _ = concordance_index_ipcw(
                        y_train.head(n_subsamples),
                        y.head(n_subsamples),
                        y_pred_at_t[:n_subsamples],
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

    def plot_performance_time(self, data_params, x_col="n_samples"):
        # Plot performance vs time
        df = self.load_cv_results(data_params, x_col)
        fig, ax = plt.subplots(figsize=(8, 4))
        fit_time = {
            "mean_fit_time": [],
            "std_fit_time": [],
            "estimator_name": [],
            "mean_ipsr": [],
        }
        for estimator_name in tqdm(self.estimator_names):
            for x_col_param, df_group in df.groupby(x_col):
                data_params[x_col] = x_col_param

                X, y = self.load_dataset(data_params, return_X_y=True)
                time_grid = make_time_grid(y["duration"])

                estimator = _get_estimator(df_group, estimator_name)

                y_train = estimator.y_train  # hack for benchmarks
                y_pred = self.get_predictions(X, time_grid, estimator, estimator_name)
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
                fit_time["mean_fit_time"].append(df_group["mean_fit_time"].values[0])
                fit_time["estimator_name"].append(estimator_name + f" {x_col_param} TS")
                fit_time["mean_ipsr"].append(np.mean(event_specific_ipsr))
                fit_time["std_fit_time"].append(df_group["std_fit_time"].values[0])
        fit_time = pd.DataFrame(fit_time)
        sns.scatterplot(
            fit_time,
            x="mean_fit_time",
            y="mean_ipsr",
            hue="estimator_name",
            ax=ax,
        )
        ax.set(
            xlabel="time(s) to fit",
            ylabel="IPSR",
        )
        plt.savefig(self.path_profile / "performance_vs_time.pdf", format="pdf")


class WeibullDisplayer(BaseDisplayer):
    def __init__(self, path_session, estimator_names):
        super().__init__(
            path_session,
            estimator_names,
            dataset_name="weibull",
        )

    def plot_memory_time(self, data_params, x_col="n_samples"):
        df = self.load_cv_results(data_params, x_col)

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
        plt.savefig(self.path_profile / "memory_time.pdf", format="pdf")

    def plot_IPSR(self, data_params):
        x_cols = ["n_samples", "censoring_relative_scale"]
        fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)
        sns.despine(fig=fig)
        for x_col, ax in zip(x_cols, axes):
            self._plot_IPSR(data_params, x_col, ax)
        plt.savefig(self.path_profile / "IPSR.pdf", format="pdf")

    def _plot_IPSR(self, data_params, x_col, ax):
        df = self.load_cv_results(data_params, x_col)

        for estimator_name in tqdm(self.estimator_names):
            x_col_params, all_ipsr = [], []
            for x_col_param, df_group in df.groupby(x_col):
                data_params[x_col] = x_col_param

                X, y = self.load_dataset(data_params, return_X_y=True)
                time_grid = make_time_grid(y["duration"])

                estimator = _get_estimator(df_group, estimator_name)

                y_train = estimator.y_train  # hack for benchmarks
                y_pred = self.get_predictions(X, time_grid, estimator, estimator_name)
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
                label=estimator_name,
                marker="o",
            )
            ax.set(
                title="IBS",
                xlabel=x_col,
                ylabel=None,
            )
            ax.legend()

    def load_cv_results(self, data_params, x_col=None):
        query = _make_query(data_params, x_col)
        return self.df.query(query)

    def load_dataset(self, data_params, return_X_y=False, use_cache=True):
        del use_cache
        return make_synthetic_competing_weibull(
            **data_params, return_X_y=return_X_y, random_state=1345
        )

    def get_predictions(self, X, times, estimator, estimator_name):
        """TODO: implement cache if some estimators take long to predict."""
        return estimator.predict_cumulative_incidence(X, times)


class SEERDisplayer(BaseDisplayer):
    def __init__(self, path_session, estimator_names):
        super().__init__(
            path_session,
            estimator_names,
            dataset_name="seer",
        )
        self.estimator_cumulative_incidence = dict()

    def plot_memory_time(self, data_params, x_col=None):
        del x_col
        df = self.load_cv_results(data_params)

        fig, ax = plt.subplots()

        sns.barplot(
            df,
            x="estimator_name",
            y="peak_memory",
            ax=ax,
        )
        ax.set(
            title="RAM peak",
            ylabel="RAM (GiB)",
        )

        fig, axes = plt.subplots(ncols=2, sharey=True)

        sns.barplot(df, x="estimator_name", y="mean_fit_time", ax=axes[0])
        axes[0].set(
            title="Time to fit",
            ylabel="time(s)",
        )

        sns.barplot(df, x="estimator_name", y="mean_score_time", ax=axes[1])
        axes[1].set(
            title="Time to test",
            ylabel=None,
        )
        plt.savefig(self.path_profile / "memory_time.pdf", format="pdf")

    def load_cv_results(self, data_params, x_col=None):
        del data_params, x_col
        return self.df

    def load_dataset(self, data_params, return_X_y=False, use_cache=True):
        del data_params
        if use_cache:
            path_cache = Path("cache_seer_surv_preprocessing.pkl")
            if path_cache.exists():
                bunch = pickle.load(open(path_cache, "rb"))
            else:
                bunch = load_seer(
                    SEER_PATH, survtrace_preprocessing=True, return_X_y=False
                )
                pickle.dump(bunch, open(path_cache, "wb"))
        else:
            bunch = load_seer(SEER_PATH, survtrace_preprocessing=True, return_X_y=False)

        _, X_test, _, y_test = train_test_split(
            bunch.X,
            bunch.y,
            test_size=0.3,
            random_state=SEED,
        )
        bunch.X, bunch.y = X_test, y_test

        if return_X_y:
            return bunch.X, bunch.y

        return bunch

    def get_predictions(self, X, times, estimator, estimator_name):
        if estimator_name in self.estimator_cumulative_incidence:
            y_pred = self.estimator_cumulative_incidence[estimator_name]
        else:
            y_pred = estimator.predict_cumulative_incidence(X, times)
            self.estimator_cumulative_incidence[estimator_name] = y_pred
        return y_pred


# %%

path_session = "2024-01-20"
estimator_names = ["gbmi_competing_loss"]
displayer = SEERDisplayer(path_session, estimator_names)

data_params = {}
displayer.plot_memory_time(data_params)

# %%

displayer.plot_PSR(data_params)

# %%

displayer.plot_marginal_incidence(data_params)

# %%

displayer.plot_individual_incidence(data_params, sample_ids=2)

# %%

displayer.print_table_metrics(data_params=data_params)

# %%

path_session = "2024-01-20"
estimator_names = ["gbmi_competing_loss"]
displayer = WeibullDisplayer(path_session, estimator_names)

data_params = {
    "censoring_relative_scale": 1.5,
    "n_events": 3,
    "complex_features": True,
    "independent_censoring": True,
}
displayer.plot_memory_time(data_params)

# %%
displayer.plot_performance_time(data_params)
# %%

displayer.plot_IPSR(data_params)

# %%
data_params
# %%
data_params.update(
    {
        "censoring_relative_scale": 1.5,
        "independent_censoring": True,
    }
)
displayer.plot_PSR(
    data_params=data_params,
)

# %%
data_params.update(
    {
        "censoring_relative_scale": 1.5,
        "independent_censoring": False,
    }
)
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
data_params["independent_censoring"] = True
displayer.plot_individual_incidence(data_params, sample_ids=2)
# %%

displayer.print_table_metrics(data_params=data_params)
# %%
