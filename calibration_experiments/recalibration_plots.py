# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
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

from .load_preds import load_predictions

PATH_PREDICTIONS = Path("preds/")
DATASET_NAME = "competing_weibull"

if DATASET_NAME == "competing_weibull":
    n_samples = 10000
    n_events = 3

    X, y = make_synthetic_competing_weibull(
        n_samples=n_samples,
        return_X_y=True,
        n_events=n_events,
        censoring_relative_scale=1.5,
        random_state=0,
    )

PATH_PREDICTIONS = Path("preds/")
DATASET_NAME = "competing_weibull"

preds = load_predictions(PATH_PREDICTIONS, DATASET_NAME)
# %%
n_events = preds["AalenJohansenEstimator"]["predictions_whole_train"].shape[1] - 1

# %%
fig, ax = plt.subplots(ncols=n_events + 1, figsize=(15, 5))

for model in preds.keys():
    predictions = preds[model]["predictions_whole_train"]
    times = preds[model]["times"]
    for event_id in range(n_events + 1):
        inc_prob_sb = predictions[:, event_id, :].mean(axis=0)
        ax[event_id].plot(times, inc_prob_sb, label=model)
        ax[event_id].set_title("Incidence function for event {}".format(event_id))

plt.legend()
# plt.savefig("uncalibrated_models.pdf", format="pdf")
plt.show()

# %%
X_train_, X_test, y_train_, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)
X_train, X_conf, y_train, y_conf = train_test_split(
    X_train_, y_train_, random_state=0, test_size=0.5
)

for model in preds.keys():
    f = preds[model]["prediction_duration_conf"]
    f_infty = preds[model]["prediction_infty_conf"]
    times = preds[model]["times"]
    s_t = f[:, 0, :]
    for event_id in range(1, 4):
        fk_t = f[:, event_id, :]
        fk_infty = f_infty[:, event_id, :]

        final_binning = d_calibration(
            fk_t, fk_infty, s_t, y_conf, event_of_interest=event_id
        )
        final_binning.plot(label=model)
        plt.plot(
            range(1, 101),
            np.linspace(0, 1, 100),
            linestyle="--",
            color="black",
            label="Perfect calibration",
        )
        plt.title(f"{model} D-calibration with censoring for event {event_id}")
        plt.legend()
        plt.show()
# %%

for model in preds.keys():
    f_conf = preds[model]["predictions_conf"]
    times = preds[model]["times"]
    cal, diff = aj_calibration(y_conf, times, f_conf, return_diff_at_t=True)
    print(f"{model} AJ calibration: {cal}")
# %%
c_indexes = {}
ibss = {}
accuracy_in_times = {}
for model in preds.keys():
    predictions_test = preds[model]["predictions_test"]
    predictions_conf = preds[model]["predictions_conf"]
    predictions_recalibrated = recalibrate_incidence_functions_predictions(
        predictions_test,
        predictions_conf,
        times,
        y_conf,
    )

    c_indexes[model] = []
    ibss[model] = []
    c_indexes[model + "_recalibrated"] = []
    ibss[model + "_recalibrated"] = []

    for event_id in range(1, n_events + 1):
        c_index = concordance_index_incidence(
            y_test,
            predictions_test[:, event_id, :],
            y_train=y_train,
            time_grid=times,
            event_of_interest=event_id,
            ipcw_estimator="km",
        )
        c_indexes[model].append(c_index)

        ibs = integrated_brier_score_incidence(
            y_test=y_test,
            y_pred=predictions_test[:, event_id, :],
            y_train=y_train,
            times=times,
            event_of_interest=event_id,
        )
        ibss[model].append(ibs)

        c_index = concordance_index_incidence(
            y_test,
            predictions_recalibrated[:, event_id, :],
            y_train=y_train,
            time_grid=times,
            event_of_interest=event_id,
            ipcw_estimator="km",
        )
        c_indexes[model + "_recalibrated"].append(c_index)

        ibs = integrated_brier_score_incidence(
            y_test=y_test,
            y_pred=predictions_recalibrated[:, event_id, :],
            y_train=y_train,
            times=times,
            event_of_interest=event_id,
        )
        ibss[model + "_recalibrated"].append(ibs)

    accuracy_in_times[model] = accuracy_in_time(
        y_test,
        predictions_test,
        time_grid=times,
    )
    accuracy_in_times[model + "_recalibrated"] = accuracy_in_time(
        y_test,
        predictions_recalibrated,
        time_grid=times,
    )


# %%
pd.DataFrame(c_indexes)
# %%
pd.DataFrame(ibss)
# %%
pd.DataFrame(accuracy_in_times)

# %%
