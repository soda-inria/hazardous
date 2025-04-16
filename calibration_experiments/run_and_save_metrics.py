# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

from hazardous.data._competing_weibull import load_synthetic
from hazardous.data._seer import load_seer, FeatureEncoder
from hazardous.data._metabric import load_metabric

from hazardous.calibration._aj_calibration import (
    aj_calibration,
    recalibrate_incidence_functions_predictions,
)
from hazardous.calibration._d_calibration import d_calibration
from hazardous.metrics import (
    integrated_brier_score_incidence,
    concordance_index_incidence,
    accuracy_in_time,
)

from hazardous import SurvivalBoost
from models_sota._deephit import DeepHitEstimator
from models_sota._aalen_johansen import AalenJohansenEstimator
from models_sota._finegray import FineGrayEstimator
from models_sota._rsf import RSFEstimator
from models_sota.survtrace._model import SurvTRACE

PATH_PREDICTIONS = Path("preds/")
DATASET_NAME = "metabric"

if DATASET_NAME == "competing_weibull":
    n_samples = None

if DATASET_NAME == "seer10k":
    n_samples = 10000

if DATASET_NAME == "seer100k":
    n_samples = 100000

if DATASET_NAME == "seer":
    n_samples = None

if DATASET_NAME == "metabric":
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


def compute_ft(model, X, y):
    f_t = [
        model.predict_cumulative_incidence(
            X.iloc[i : i + 1], times=np.array([y["duration"].iloc[i]])
        )
        for i in range(len(X))
    ]
    f_t = np.asarray(f_t).reshape(len(X), n_events + 1, 1)
    return f_t


def load_dataset(dataset_name, n_samples=None, seed=None):
    if DATASET_NAME == "competing_weibull":
        X, y = load_synthetic(
            input_path="../hazardous/data/competing_synthetic.csv", return_X_y=True
        )
    if DATASET_NAME.find("seer") != -1:
        X, y = load_seer(
            input_path="../hazardous/data/seer_cancer_cardio_raw_data.txt",
            return_X_y=True,
            survtrace_preprocessing=True,
        )
        X = FeatureEncoder().fit_transform(X)

    if DATASET_NAME.find("metabric") != -1:
        X, y = load_metabric(return_X_y=True)
        X = FeatureEncoder().fit_transform(X)

    X_train_, X_test, y_train_, y_test = train_test_split(
        X, y, random_state=seed, test_size=0.3
    )

    if n_samples is not None and n_samples < len(X_train_):
        X_train_, _, y_train_, _ = train_test_split(
            X_train_, y_train_, random_state=seed, train_size=n_samples
        )

    return X_train_, X_test, y_train_, y_test


if __name__ == "__main__":
    for seed in range(5):
        X_train_, X_test, y_train_, y_test = load_dataset(
            DATASET_NAME, n_samples=n_samples, seed=seed
        )
        n_events = y_train_["event"].max()
        X_train, X_conf, y_train, y_conf = train_test_split(
            X_train_, y_train_, random_state=seed, test_size=0.5
        )
        times = np.quantile(y_train_["duration"], np.linspace(0, 1, 100))
        for recalibration in [False, True]:
            for model_name in list(models):
                metrics_model = {}

                model = INIT_MODEL_FUNCS[model_name](random_state=seed)
                model.fit(X_train.astype("float64"), y_train)

                # Need to save this for visualization
                prediction_whole_train = model.predict_cumulative_incidence(
                    X_train, times=times
                ).mean(axis=0)

                prediction_test = model.predict_cumulative_incidence(
                    X_test, times=times
                )

                prediction_infty_test = model.predict_cumulative_incidence(
                    X_test, times=np.array([y_train["duration"].max()])
                )
                if recalibration is False:
                    prediction_duration_test = compute_ft(model, X_test, y_test)
                    s_t = prediction_duration_test[:, 0, :]

                if recalibration:
                    prediction_conf = model.predict_cumulative_incidence(
                        X_conf, times=times
                    )
                    final_prediction = recalibrate_incidence_functions_predictions(
                        prediction_test,
                        prediction_conf,
                        times,
                        y_conf,
                    )

                    prediction_infty_conf = model.predict_cumulative_incidence(
                        X_conf, times=np.array([y_train["duration"].max()])
                    )
                    prediction_infty_test = recalibrate_incidence_functions_predictions(
                        prediction_infty_test,
                        prediction_infty_conf,
                        np.array([y_train["duration"].max()]),
                        y_conf,
                    )
                    prediction_duration_test = model.predict_cumulative_incidence(
                        X_test, times=y_test.duration.values
                    )
                    prediction_duration_conf = model.predict_cumulative_incidence(
                        X_conf, times=y_test.duration.values
                    )
                    prediction_duration_test = (
                        recalibrate_incidence_functions_predictions(
                            prediction_duration_test,
                            prediction_duration_conf,
                            y_test.duration.values,
                            y_conf,
                        )
                    )
                    prediction_duration_test = np.array(
                        [
                            prediction_duration_test[i, :, i]
                            for i in range(len(prediction_duration_test))
                        ]
                    )
                    prediction_duration_test = prediction_duration_test.reshape(
                        len(X_test), n_events + 1, 1
                    )
                    s_t = prediction_duration_test[:, 0, :]

                else:
                    final_prediction = prediction_test

                ajk_cal, _ = aj_calibration(
                    y_test, times, final_prediction, return_diff_at_t=True
                )
                metrics_model["ajk_cal"] = [*ajk_cal.values()]

                metrics_model["d_cal"] = {}
                metrics_model["ibs"] = {}
                metrics_model["c_index"] = {}
                for event_id in range(1, n_events + 1):
                    predictions_event = final_prediction[:, event_id, :]
                    c_index = concordance_index_incidence(
                        y_test=y_test,
                        y_pred=predictions_event,
                        y_train=y_train,
                        time_grid=times,
                        event_of_interest=event_id,
                        ipcw_estimator="km",
                    )

                    ibs = integrated_brier_score_incidence(
                        y_test=y_test,
                        y_pred=predictions_event,
                        y_train=y_train,
                        times=times,
                        event_of_interest=event_id,
                    )

                    fk_infty = prediction_infty_test[:, event_id, :]
                    dk_cal = d_calibration(
                        fk=prediction_duration_test[:, event_id, :],
                        fk_infty=fk_infty,
                        s_t=s_t,
                        y_conf=y_test,
                        event_of_interest=event_id,
                    )

                    metrics_model["c_index"][event_id] = c_index[0]
                    metrics_model["ibs"][event_id] = ibs
                    metrics_model["d_cal"][event_id] = dk_cal.values.flatten().tolist()

                metrics_model["accuracy_in_time"] = (
                    accuracy_in_time(
                        y_test,
                        prediction_test,
                        time_grid=times,
                    )[0]
                    .flatten()
                    .tolist()
                )

                path_dir = (
                    PATH_PREDICTIONS
                    / DATASET_NAME
                    / model_name
                    / f"recalibration_{recalibration}"
                    / f"seed_{seed}"
                )
                path_dir.mkdir(parents=True, exist_ok=True)

                path_dir_pred = path_dir / "vizualization_mean_predictions.csv"
                prediction_whole_train = pd.DataFrame(prediction_whole_train)
                prediction_whole_train.to_parquet(path_dir_pred)
                # save metrics in a json
                path_file_agg = path_dir / "metrics.json"
                json.dump(metrics_model, open(path_file_agg, "w"))

    print(f"Wrote {path_dir}")

# %%
