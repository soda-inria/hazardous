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
)
from hazardous.calibration._d_calibration import d_calibration
from hazardous.metrics import (
    integrated_brier_score_incidence,
    concordance_index_incidence,
    accuracy_in_time,
)
from hazardous.recalibration_posthoc.ts_recalibration import (
    RecalibrationTS,
)
from hazardous.recalibration_posthoc.aj_recalibration import (
    RecalibrationAJ,
)

from models_sota._deephit import DeepHitEstimator
from models_sota._aalen_johansen import AalenJohansenEstimator
from models_sota._finegray import FineGrayEstimator
from models_sota._rsf import RSFEstimator
from models_sota.survtrace._model import SurvTRACE
from hazardous import SurvivalBoost

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


def init_survivalboost(random_state=None, **model_params):
    return SurvivalBoost(random_state=random_state, n_iter=50, show_progressbar=False)


INIT_MODEL_FUNCS = {
    "DeepHit": init_deephit,
    "FineGray": init_fine_and_gray,
    "AalenJohansen": init_aalen_johansen,
    "RSF": init_random_survival_forest,
    "SurvTRACE": init_survtrace,
    #    "SurvivalBoost": init_survivalboost,
}

models = INIT_MODEL_FUNCS.keys()


def compute_ft(model, X, y, model_name):
    # import ipdb; ipdb.set_trace()
    if model_name == "AalenJohansen":
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        f_t = [
            model.predict_cumulative_incidence(
                X_array[i : i + 1], times=np.array([y["duration"].iloc[i]])
            )
            for i in range(X.shape[0])
        ]
    else:
        f_t = [
            model.predict_cumulative_incidence(
                X.iloc[i : i + 1], times=np.array([y["duration"].iloc[i]])
            )
            for i in range(X.shape[0])
        ]
    f_t = np.asarray(f_t).reshape(len(X), n_events + 1, 1)
    return f_t


def load_dataset(dataset_name, n_samples=None, seed=None):
    if dataset_name == "competing_weibull":
        X, y = load_synthetic(
            input_path="../hazardous/data/competing_synthetic.csv", return_X_y=True
        )
    if dataset_name.find("seer") != -1:
        X, y = load_seer(
            input_path="../hazardous/data/seer_cancer_cardio_raw_data.txt",
            return_X_y=True,
            survtrace_preprocessing=True,
        )
        X = FeatureEncoder().fit_transform(X)

    if dataset_name.find("metabric") != -1:
        X, y = load_metabric(return_X_y=True)
        X = FeatureEncoder().fit_transform(X)

    X_train_, X_test, y_train_, y_test = train_test_split(
        X, y, random_state=seed, test_size=0.3
    )

    if n_samples is not None and n_samples < len(X_train_):
        X_train_, _, y_train_, _ = train_test_split(
            X_train_, y_train_, random_state=seed, train_size=n_samples
        )
    if len(X_test) > 10000:
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, random_state=seed, train_size=5000
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
        for model_name in list(models):
            model = INIT_MODEL_FUNCS[model_name](random_state=seed)
            model.fit(X_train.astype("float64"), y_train)
            for recalibration in ["False", "aj_recalibration", "ts_recalibration"]:  #
                if recalibration == "False":
                    model_recal = model
                    prediction_test = model.predict_cumulative_incidence(
                        X_test, times=times
                    )
                    prediction_duration_test = compute_ft(
                        model, X_test, y_test, model_name
                    )
                elif recalibration == "aj_recalibration":
                    model_recal = RecalibrationAJ(model, seed=seed)
                    model_recal = model_recal.fit(X_conf, y_conf, times=times)
                    prediction_test = model_recal.predict_cumulative_incidence(X_test)
                    prediction_duration_test = model_recal.compute_ft(X_test, y_test)
                elif recalibration == "ts_recalibration":
                    model_recal = RecalibrationTS(model, seed=seed)
                    model_recal = model_recal.fit(X_conf, y_conf, times=times)
                    prediction_test = model_recal.predict_cumulative_incidence(X_test)
                    prediction_duration_test = model_recal.compute_ft(X_test, y_test)
                    # import ipdb; ipdb.set_trace()
                print(
                    f"Running model {model_name} with seed {seed}, recalibration"
                    f" {recalibration}"
                )
                metrics_model = {}
                metrics_model["model_name"] = model_name
                metrics_model["recalibration"] = recalibration
                metrics_model["seed"] = seed

                prediction_infty_test = model_recal.predict_cumulative_incidence(
                    X_test, times=np.array([y_train["duration"].max()])
                )

                s_t = prediction_duration_test[:, 0, :]
                ajk_cal = aj_calibration(
                    y_test, times, prediction_test, return_diff_at_t=False
                )
                metrics_model["ajk_cal"] = [*ajk_cal.values()]

                metrics_model["d_cal"] = {}
                metrics_model["ibs"] = {}
                metrics_model["c_index"] = {}
                for event_id in range(1, n_events + 1):
                    predictions_event = prediction_test[:, event_id, :]
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
                    / recalibration
                    / f"seed_{seed}"
                )
                path_dir.mkdir(parents=True, exist_ok=True)

                path_dir_pred = path_dir / "vizualization_mean_predictions.csv"
                prediction_visu = pd.DataFrame(prediction_test.mean(axis=0))
                prediction_visu.to_parquet(path_dir_pred)
                # save metrics in a json
                path_file_agg = path_dir / "metrics.json"
                json.dump(metrics_model, open(path_file_agg, "w"))

    print(f"Wrote {path_dir}")

# %%
