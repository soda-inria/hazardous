# %%
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.utils import Bunch
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from skrub import TableVectorizer

from pycox.datasets import metabric, support

from hazardous.data import load_seer
from hazardous._xbgse import XGBSE
from hazardous import SurvivalBoost
from hazardous import metrics


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

        str_config = "_".join([f"{k}={v}" for k, v in hp_params.items()])
        model_id = f"{model_name}__{str_config}"

        self.results["model_id"].append(model_id)
        self.results["model_name"].append(model_name)
        self.results["dataset_name"].append(dataset_name)
        self.results["seed"].append(seed)
        self.results["hp_params"].append(hp_params)
        self.results["y_pred"].append(y_pred)
        self.results["time_grid"].append(time_grid)

        self.results["ibs"].append(
            self.compute_multi_ibs(y_train, y_test, y_pred, time_grid)
        )
        self.results["brier_scores"].append(
            self.compute_multi_brier_scores(y_train, y_test, y_pred, time_grid)
        )
        self.results["c_index"].append(
            self.compute_multi_c_index(y_train, y_test, y_pred, time_grid)
        )
        self.results["acc_in_time"].append(
            self.compute_acc_in_time(y_test, y_pred, time_grid)
        )

        print(f"ibs: {self.results['ibs'][-1]}")

    def compute_multi_ibs(self, y_train, y_test, y_pred, time_grid):
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

    def compute_multi_brier_scores(self, y_train, y_test, y_pred, time_grid):
        brier_scores = []
        n_events = y_pred.shape[1]
        for event_idx in range(n_events):
            y_pred_event = y_pred[:, event_idx]
            if event_idx == 0:
                brier_scores_event = metrics.brier_score_survival(
                    y_train=y_train,
                    y_test=y_test,
                    y_pred=y_pred_event,
                    times=time_grid,
                )
            else:
                brier_scores_event = metrics.brier_score_incidence(
                    y_train=y_train,
                    y_test=y_test,
                    y_pred=y_pred_event,
                    times=time_grid,
                )
            brier_scores.append(brier_scores_event)
        return brier_scores

    def compute_multi_c_index(self, y_train, y_test, y_pred, time_grid):
        c_index = []
        n_events = y_pred.shape[1]
        taus = np.quantile(time_grid, self.c_index_quantiles)
        for event_idx in range(n_events):
            c_index.append(
                metrics.concordance_index_incidence(
                    y_test=y_test,
                    y_pred=y_pred[:, event_idx],
                    y_train=y_train,
                    time_grid=time_grid,
                    event_of_interest=event_idx,
                    taus=taus,
                )
            )
        return c_index

    def compute_acc_in_time(self, y_test, y_pred, time_grid):
        acc_in_time, taus = metrics.accuracy_in_time(
            y_test=y_test,
            y_pred=y_pred,
            time_grid=time_grid,
            quantiles=self.acc_in_time_quantiles,
        )
        return dict(acc_in_time=acc_in_time, taus=taus)

    def plot_metrics(self, dataset_name):
        df = pd.DataFrame(self.results)
        df = df.loc[df["dataset_name"] == dataset_name]

        # agg_results = []
        for model_id in df["model_id"].unique():
            # df_model = df.loc[df["model_id"] == model_id]
            pass
            # ibs = f"{df_model['ibs'].mean():.5f} ± {df_model['ibs'].std():.5f}"

    def plot_brier_scores(self, dataset_name):
        pass

    def plot_acc_in_time(self, dataset_name):
        pass

    def plot_target_distribution(self, dataset_name):
        pass

    def plot_km_calibration(self, dataset_name):
        pass

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


datasets = {
    # "seer": load_seer_,
    # "support": load_support_,
    "metabric": load_metabric_,
}

model_hp_params = defaultdict(dict)
model_hp_params["survival_boost"] = {"time_sampler": ["uniform", "kaplan-meier"]}

models = {
    "survival_boost": SurvivalBoost,
    "xgbse": XGBSE,
}

seeds = range(1)

scorer = Scorer()
for dataset_name, dataset_func in datasets.items():
    bunch = dataset_func()
    for model_name, model_cls in models.items():
        hp_params_grid = model_hp_params[model_name]

        # ParameterGrid allows to iterate from a grid.
        for hp_params in ParameterGrid(hp_params_grid):
            for seed in seeds:
                model = model_cls(random_state=seed, **hp_params)
                scorer.compute_scores(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    model=model,
                    X=bunch.X,
                    y=bunch.y,
                    hp_params_set=hp_params,
                    seed=seed,
                )

print(scorer)
# %%
scorer.plot_target_distribution("metabric")

# %%

scorer.plot_km_calibration("metabric")

# %%
scorer.plot_metrics("metabric")


# %%
