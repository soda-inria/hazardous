# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.utils import Bunch
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from skrub import TableVectorizer

from pycox.datasets import metabric, support

from hazardous.data import load_seer
from hazardous.utils import check_y_survival, make_time_grid

from hazardous._xbgse import XGBSE
from hazardous import SurvivalBoost
from hazardous import metrics
from hazardous._km_sampler import _KaplanMeierSampler, _AalenJohansenSampler


TARGET_COLS = ["event", "duration"]


def load_metabric_():
    df = metabric.read_df()
    return Bunch(X=df.drop(columns=TARGET_COLS), y=df[TARGET_COLS])


def load_support_():
    df = support.read_df()
    return Bunch(X=df.drop(columns=TARGET_COLS), y=df[TARGET_COLS])


def load_seer_():
    X, y = load_seer(
        "hazardous/data/seer_cancer_raw_data.txt",
        survtrace_preprocessing=True,
        return_X_y=True,
    )
    return Bunch(X=X, y=y)


class Scorer:
    def __init__(
        self,
        vectorizer=None,
        c_index_quantiles=None,
        acc_in_time_quantiles=None,
    ):
        if vectorizer is None:
            vectorizer = TableVectorizer(
                high_cardinality=OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
                cardinality_threshold=1,
            )
        self.vectorizer = vectorizer

        if c_index_quantiles is None:
            c_index_quantiles = (0.25, 0.5, 0.75)
        self.c_index_quantiles = c_index_quantiles

        if acc_in_time_quantiles is None:
            acc_in_time_quantiles = (0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875)
        self.acc_in_time_quantiles = acc_in_time_quantiles

        self.results = defaultdict(list)
        self.dataset_marginal_est = dict()

    def compute_scores(
        self,
        model_name,
        dataset_name,
        model,
        X,
        y,
        hp_params,
        seed,
        time_grid=None,
    ):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y["event"], test_size=0.2, random_state=seed
        )

        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)

        model.fit(X_train, y_train)

        if time_grid is None:
            if hasattr(model, "time_grid_"):
                time_grid = model.time_grid_
            else:
                raise ValueError(
                    "No time_grid passed and the model doesn't have a time_grid_ "
                    "attribute."
                )

        y_pred = model.predict_cumulative_incidence(X_test, times=time_grid)

        if hp_params:
            str_config = "_".join([f"{k}={v}" for k, v in hp_params.items()])
            model_id = "__".join([model_name, str_config])
        else:
            model_id = model_name

        ibs = self._compute_ibs(y_train, y_test, y_pred, time_grid)
        c_index = self._compute_c_index(y_train, y_test, y_pred, time_grid)
        brier_scores = self._compute_brier_scores(y_train, y_test, y_pred, time_grid)
        acc_in_time = self._compute_acc_in_time(y_test, y_pred, time_grid)

        if dataset_name not in self.dataset_marginal_est:
            self.dataset_marginal_est[dataset_name] = self._compute_km_or_aj(
                y, n_events=(y_pred.shape[1] - 1)
            )

        self.results["model_id"].append(model_id)
        self.results["model_name"].append(model_name)
        self.results["dataset_name"].append(dataset_name)
        self.results["seed"].append(seed)
        self.results["hp_params"].append(hp_params)
        self.results["y_pred"].append(y_pred)
        self.results["time_grid"].append(time_grid)

        self.results["ibs"].append(ibs)
        self.results["c_index"].append(c_index)
        self.results["brier_scores"].append(brier_scores)
        self.results["acc_in_time"].append(acc_in_time)

        print(f"{ibs=}")

    def remove(self, index):
        if index >= len(self):
            raise ValueError(f"{index=} is larger than the number of runs {len(self)}")
        for key in self.results.keys():
            self.results[key].pop(index)

    def plot_ibs(self, dataset_name):
        if (df := self._get_results_dataset(dataset_name)) is None:
            return

        agg_metrics = []
        for model_id, df_model in df.groupby("model_id"):
            ibs = f"{df_model['ibs'].mean():.5f} ± {df_model['ibs'].std():.5f}"

            agg_metrics.append(
                {
                    "dataset_name": dataset_name,
                    "model_id": model_id,
                    "ibs": ibs,
                }
            )

        return pd.DataFrame(agg_metrics)

    def plot_c_index(self, dataset_name):
        if (df := self._get_results_dataset(dataset_name)) is None:
            return

        agg_metrics = []
        for model_id, df_model in df.groupby("model_id"):
            n_events = len(df_model["c_index"].values[0])

            for event_idx in range(n_events):
                agg_event = []
                for seed_row in df_model["c_index"].to_list():
                    seed_event_row = seed_row[event_idx]
                    agg_event.append(seed_event_row)

                mean_list = np.mean(agg_event, axis=0).round(5).tolist()
                std_list = np.std(agg_event, axis=0).round(5).tolist()
                zip_iter = zip(mean_list, std_list, self.c_index_quantiles)

                for mean_q, std_q, q in zip_iter:
                    agg_metrics.append(
                        dict(
                            dataset_name=dataset_name,
                            model_id=model_id,
                            event_idx=event_idx + 1,
                            q=q,
                            c_index=f"{mean_q} ± {std_q}",
                        )
                    )
        df = pd.DataFrame(agg_metrics).pivot(
            index="model_id", columns=["q", "event_idx"], values=["c_index"]
        )
        return df

    def plot_brier_scores(self, dataset_name):
        if (df := self._get_results_dataset(dataset_name)) is None:
            return

        n_events = len(df["brier_scores"].values[0])

        fig, axes = plt.subplots(ncols=n_events)
        axes = np.atleast_1d(axes)

        for model_id, df_model in df.groupby("model_id"):
            time_grid = df_model["time_grid"].values[0]

            for event_idx in range(n_events):
                agg_event = []
                for seed_row in df_model["brier_scores"].to_list():
                    seed_event_row = seed_row[event_idx]
                    agg_event.append(seed_event_row)

                mean_arr = np.mean(agg_event, axis=0).round(5)
                std_arr = np.std(agg_event, axis=0).round(5)

                axes[event_idx].plot(time_grid, mean_arr, label=model_id)
                axes[event_idx].fill_between(
                    time_grid,
                    y1=mean_arr - std_arr,
                    y2=mean_arr + std_arr,
                    alpha=0.3,
                )

        for event_idx, ax in enumerate(axes, 1):
            ax.set_title(f"Event {event_idx}")
        axes[-1].legend()

        sns.despine()
        plt.show()

    def plot_acc_in_time(self, dataset_name):
        if (df := self._get_results_dataset(dataset_name)) is None:
            return

        results = []
        for model_id, df_model in df.groupby("model_id"):
            agg_acc_in_time = []
            for row_dict in df_model["acc_in_time"]:
                seed_acc_in_time = row_dict["acc_in_time"]
                seed_taus = row_dict["taus"]
                agg_acc_in_time.append(seed_acc_in_time)

            mean_arr = np.mean(agg_acc_in_time, axis=0).round(5)
            std_arr = np.std(agg_acc_in_time, axis=0).round(5)

            for q, tau, mean_acc, std_acc in zip(
                self.acc_in_time_quantiles, seed_taus, mean_arr, std_arr
            ):
                results.append(
                    dict(
                        model_id=model_id,
                        q=q,
                        tau=tau,
                        mean_acc=mean_acc,
                        std_acc=std_acc,
                    )
                )

        df = pd.DataFrame(results)

        fig, ax = plt.subplots()

        sns.lineplot(
            df,
            x="tau",
            y="mean_acc",
            hue="model_id",
            palette="colorblind",
            legend=False,
            ax=ax,
        )
        sns.scatterplot(
            df,
            x="tau",
            y="mean_acc",
            hue="model_id",
            s=50,
            zorder=100,
            style="model_id",
            palette="colorblind",
            ax=ax,
        )

        ax.legend()
        ax.grid()
        ax.set_xlabel("Time quantiles", fontsize=10)
        ax.set_ylabel("Accuracy in time", fontsize=10)
        sns.despine()
        plt.show()

    def plot_target_distribution(self, y):
        check_y_survival(y)
        sns.histplot(
            y,
            x="duration",
            hue="event",
            multiple="stack",
            palette="colorblind",
        )
        plt.show()

    def plot_km_calibration(self, dataset_name):
        if (df := self._get_results_dataset(dataset_name)) is None:
            return

        # If a single model for this dataset has a competing risk setting
        # we compare the Aalen-Johansen estimator to all models.
        # Otherwise, we use the Kaplan-Meier estimator for comparison.

        n_events = max([row.shape[1] - 1 for row in df["y_pred"].values])
        fig, axes = plt.subplots(ncols=n_events)
        axes = np.atleast_1d(axes)

        for model_id, df_model in df.groupby("model_id"):
            time_grid = df_model["time_grid"].values[0]

            if n_events == 1:
                y_pred_agg_event = []
                for row_seed in df_model["y_pred"].values:
                    y_pred_agg_event.append(row_seed[:, 0, :].mean(axis=0))

                self._plot_mean_std(y_pred_agg_event, time_grid, model_id, axes[0])
                axes[0].set_title("Survival probability")

            else:
                for event_idx in range(n_events):
                    y_pred_agg_event = []
                    for row_seed in df_model["y_pred"].values:
                        y_pred_agg_event.append(
                            row_seed[:, event_idx + 1, :].mean(axis=0)
                        )

                    self._plot_mean_std(
                        y_pred_agg_event, time_grid, model_id, axes[event_idx]
                    )
                    axes[event_idx].set_title(f"Event {event_idx+1}")

        if n_events == 1:
            y_km = self.dataset_marginal_est[dataset_name](time_grid)
            axes[0].plot(
                time_grid,
                y_km,
                linestyle="--",
                label="Kaplan Meier",
            )
        else:
            for event_idx in range(n_events):
                aj_est = self.dataset_marginal_est[dataset_name][event_idx + 1]
                y_aj = aj_est(time_grid)
                axes[event_idx].plot(
                    time_grid,
                    y_aj,
                    linestyle="--",
                    label="Aalen-Johansen",
                )

        axes[-1].legend()
        sns.move_legend(axes[-1], "lower left", bbox_to_anchor=(1, 0))
        sns.despine()
        plt.show()

    @staticmethod
    def _plot_mean_std(list_y_pred, time_grid, model_id, ax):
        mean_pred = np.mean(list_y_pred, axis=0)
        std_pred = np.std(list_y_pred, axis=0)

        ax.plot(time_grid, mean_pred, label=model_id)
        ax.fill_between(
            time_grid, y1=mean_pred - std_pred, y2=mean_pred + std_pred, alpha=0.3
        )
        ax.grid()

    def _compute_ibs(self, y_train, y_test, y_pred, time_grid):
        ibs_events = []
        n_events = y_pred.shape[1]
        for event_idx in range(n_events):
            y_pred_event = y_pred[:, event_idx]
            if event_idx == 0:
                ibs_event = metrics.integrated_brier_score_survival(
                    y_train=y_train,
                    y_test=y_test,
                    y_pred=y_pred_event,
                    times=time_grid,
                )
            else:
                ibs_event = metrics.integrated_brier_score_incidence(
                    y_train=y_train,
                    y_test=y_test,
                    y_pred=y_pred_event,
                    times=time_grid,
                    event_of_interest=event_idx,
                )
            ibs_events.append(ibs_event)
        return round(np.mean(ibs_events), 5)

    def _compute_brier_scores(self, y_train, y_test, y_pred, time_grid):
        brier_scores = []
        n_events = y_pred.shape[1]
        for event_idx in range(1, n_events):
            y_pred_event = y_pred[:, event_idx]
            brier_scores_event = metrics.brier_score_incidence(
                y_train=y_train,
                y_test=y_test,
                y_pred=y_pred_event,
                times=time_grid,
                event_of_interest=event_idx,
            )
            brier_scores.append(brier_scores_event)
        return brier_scores

    def _compute_c_index(self, y_train, y_test, y_pred, time_grid):
        c_index = []
        n_events = y_pred.shape[1]
        taus = np.quantile(time_grid, self.c_index_quantiles)
        for event_idx in range(1, n_events):
            c_index.append(
                metrics.concordance_index_incidence(
                    y_test=y_test,
                    y_pred=y_pred[:, event_idx],
                    y_train=y_train,
                    time_grid=time_grid,
                    event_of_interest=event_idx,
                    taus=taus,
                ).tolist()
            )
        return c_index

    def _compute_acc_in_time(self, y_test, y_pred, time_grid):
        acc_in_time, taus = metrics.accuracy_in_time(
            y_test=y_test,
            y_pred=y_pred,
            time_grid=time_grid,
            quantiles=self.acc_in_time_quantiles,
        )
        return dict(acc_in_time=acc_in_time, taus=taus)

    def _compute_km_or_aj(self, y, n_events):
        if n_events == 1:
            km_sampler = _KaplanMeierSampler().fit(y)
            return km_sampler.survival_func_
        else:
            aj_sampler = _AalenJohansenSampler().fit(y)
            return aj_sampler.incidence_func_

    def _get_results_dataset(self, dataset_name):
        df = pd.DataFrame(self.results)
        if "dataset_name" not in df.columns:
            print("scorer is empty")
            return None

        df_ = df.loc[df["dataset_name"] == dataset_name]
        if df_.empty:
            dataset_names = df["dataset_name"].unique().tolist()
            print(f"{dataset_name=} not in scorer. Current datasets: {dataset_names}")
            return None

        return df_

    def __repr__(self):
        return pd.DataFrame(
            dict(
                dataset=self.results["dataset_name"],
                model=self.results["model_name"],
                seed=self.results["seed"],
                ibs=self.results["ibs"],
                hp_params=self.results["hp_params"],
            )
        ).to_markdown()

    def __len__(self):
        return len(self.results["model_id"])


# %%

bunch = load_metabric_()
hp_params = {"time_sampler": "uniform"}

time_grid = make_time_grid(bunch.y["event"], bunch.y["duration"], n_time_grid_steps=20)

scorer = Scorer()
scorer.compute_scores(
    model_name="survival_boost",
    dataset_name="metabric",
    model=SurvivalBoost(**hp_params),
    hp_params=hp_params,
    X=bunch.X,
    y=bunch.y,
    seed=0,
    time_grid=time_grid,
)

print(scorer)

# %%

hp_params = {"time_sampler": "kaplan-meier"}

scorer.compute_scores(
    model_name="survival_boost",
    dataset_name="metabric",
    model=SurvivalBoost(**hp_params),
    hp_params=hp_params,
    X=bunch.X,
    y=bunch.y,
    seed=0,
    time_grid=time_grid,
)

print(scorer)

# %%

scorer.compute_scores(
    model_name="XGBSE",
    dataset_name="metabric",
    model=XGBSE(),
    hp_params={},
    X=bunch.X,
    y=bunch.y,
    seed=0,
    time_grid=time_grid,
)

print(scorer)

# %%

scorer.plot_target_distribution(bunch.y)

# %%
scorer.plot_km_calibration("metabric")

# %%
scorer.plot_acc_in_time("metabric")

# %%

scorer.plot_c_index("metabric")

# %%
scorer.plot_brier_scores("metabric")

# %%

seeds = range(3)
hp_params_grid = {"time_sampler": ["uniform", "kaplan-meier"]}
bunch = load_seer(
    "../hazardous/data/seer_cancer_cardio_raw_data.txt",
    survtrace_preprocessing=True,
)

# ParameterGrid allows to iterate from a grid.
for hp_params in ParameterGrid(hp_params_grid):
    for seed in seeds:
        model = SurvivalBoost(random_state=seed, **hp_params)
        scorer.compute_scores(
            model_name="survival_boost",
            dataset_name="seer",
            model=model,
            X=bunch.data.head(10_000),
            y=bunch.target.head(10_000),
            hp_params=hp_params,
            seed=seed,
        )

print(scorer)

# %%

scorer.plot_target_distribution(y=bunch.target.head(10_000))

# %%

scorer.plot_ibs("seer")

# %%

scorer.plot_c_index("seer")

# %%

scorer.plot_km_calibration("seer")

# %%

scorer.plot_brier_scores("seer")

# %%

scorer.plot_ibs("seer")

# %%
