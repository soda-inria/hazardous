# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous import SurvivalBoost
from hazardous.data._seer import load_seer, FeatureEncoder

from models_sota._deephit import DeepHitEstimator
from models_sota._aalen_johansen import AalenJohansenEstimator
from models_sota._finegray import FineGrayEstimator
from models_sota._rsf import RSFEstimator
from models_sota.survtrace._model import SurvTRACE

PATH_PREDICTIONS = Path("preds/")
DATASET_NAME = "seer"

if DATASET_NAME == "competing_weibull":
    n_samples = 10000
    n_events = 3

if DATASET_NAME == "seer":
    n_samples = None


def init_survivalboost(
    random_state=None,
    **model_params,
):
    return SurvivalBoost(random_state=random_state).set_params(**model_params)


def init_deephit(
    num_nodes_shared=[64, 64],
    num_nodes_indiv=[32],
    verbose=True,
    num_durations=10,
    batch_norm=True,
    dropout=None,
    random_state=None,
    **model_params,
):
    return DeepHitEstimator(
        num_nodes_shared=num_nodes_shared,
        num_nodes_indiv=num_nodes_indiv,
        verbose=verbose,
        num_durations=num_durations,
        batch_norm=batch_norm,
        dropout=dropout,
        **model_params,
    )


def init_aalen_johansen(calculate_variance=False, random_state=None):
    return AalenJohansenEstimator(seed=random_state)


def init_fine_and_gray(random_state=None, **model_params):
    return FineGrayEstimator(random_state=random_state)


def init_random_survival_forest(random_state=None, **model_params):
    return RSFEstimator(random_state=random_state)


def init_survtrace(random_state=None, **model_params):
    return SurvTRACE(random_state=random_state, max_epochs=100)


INIT_MODEL_FUNCS = {
    "SurvivalBoost": init_survivalboost,
    "DeepHit": init_deephit,
    "FineGray": init_fine_and_gray,
    "AalenJohansen": init_aalen_johansen,
    "RSF": init_random_survival_forest,
    "SurvTRACE": init_survtrace,
}

models = INIT_MODEL_FUNCS.keys()


def compute_ft(model, X_conf, y_conf):
    f_t = [
        model.predict_cumulative_incidence(
            X_conf.iloc[i : i + 1], times=np.array([y_conf["duration"].iloc[i]])
        )
        for i in range(len(X_conf))
    ]
    f_t = np.asarray(f_t).reshape(len(X_conf), n_events + 1, 1)
    return f_t


for seed in range(5):
    if DATASET_NAME == "competing_weibull":
        X, y = make_synthetic_competing_weibull(
            n_samples=n_samples,
            return_X_y=True,
            n_events=n_events,
            censoring_relative_scale=1.5,
            random_state=seed,
        )
    if DATASET_NAME == "seer":
        X, y = load_seer(
            input_path="../hazardous/data/seer_cancer_cardio_raw_data.txt",
            return_X_y=True,
        )
        if n_samples is not None and n_samples < len(X):
            X, _, y, _ = train_test_split(X, y, random_state=seed, train_size=n_samples)
        X = FeatureEncoder().fit_transform(X)
    import ipdb

    ipdb.set_trace()
    X_train_, X_test, y_train_, y_test = train_test_split(
        X, y, random_state=seed, test_size=0.3
    )
    X_train, X_conf, y_train, y_conf = train_test_split(
        X_train_, y_train_, random_state=seed, test_size=0.5
    )
    times = np.quantile(y_train_["duration"], np.linspace(0, 1, 100))

    for model_name in list(models):
        model = INIT_MODEL_FUNCS[model_name](random_state=seed)
        model.fit(X_train.astype("float64"), y_train)
        predictions_model = {}
        prediction_whole_train = model.predict_cumulative_incidence(
            X_train_, times=times
        )
        prediction_conf = model.predict_cumulative_incidence(X_conf, times=times)
        prediction_test = model.predict_cumulative_incidence(X_test, times=times)
        prediction_infty_conf = model.predict_cumulative_incidence(
            X_conf, times=np.array([y_conf["duration"].max()])
        )
        prediction_duration_conf = compute_ft(model, X_conf, y_conf)

        predictions_model["predictions_whole_train"] = prediction_whole_train
        predictions_model["predictions_conf"] = prediction_conf
        predictions_model["predictions_test"] = prediction_test
        predictions_model["prediction_infty_conf"] = prediction_infty_conf
        predictions_model["prediction_duration_conf"] = prediction_duration_conf

        # save predictions in a json
        path_dir = PATH_PREDICTIONS / DATASET_NAME / model_name / f"seed_{seed}"
        path_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(times).to_parquet(path_dir / "times.parquet")

        for pred in predictions_model:
            path_dir_pred = path_dir / f"{pred}"
            path_dir_pred.mkdir(parents=True, exist_ok=True)
            for event_id in range(n_events + 1):
                path_file = path_dir_pred / f"event_{event_id}.parquet"
                df_pred = pd.DataFrame(predictions_model[pred][:, event_id, :])
                df_pred.to_parquet(path_file)
    print(f"Wrote {path_dir}")

# %%
