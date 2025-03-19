# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.calibration._aj_calibration import (
    aj_calibration,
    recalibrate_incidence_functions,
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

# %%
# KM-calibration without censoring
n_samples = 8000
n_events = 3

X, y = make_synthetic_competing_weibull(
    n_samples=n_samples,
    return_X_y=True,
    n_events=n_events,
    censoring_relative_scale=1.5,
    random_state=0,
)
times = np.linspace(0, y["duration"].max(), 100)
sns.histplot(data=y, x="duration", hue="event", bins=50, kde=True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
X_train, X_conf, y_train, y_conf = train_test_split(
    X_train, y_train, random_state=0, test_size=0.5
)
# %%
aalen = AalenJohansenEstimator()
aalen.fit(X_train, y_train)
aalen_incidence_probs = aalen.predict_cumulative_incidence(X_test, times=times)

deephit = DeepHitEstimator()
deephit.fit(X_train, y_train)
deephit_inc_probs = deephit.predict_cumulative_incidence(X, times=times)

survivalboost = SurvivalBoost(n_iter=30, show_progressbar=False)
survivalboost.fit(X_train, y_train)
survivalboost_inc_probs = survivalboost.predict_cumulative_incidence(X, times=times)

# %%
finegray = FineGrayEstimator()
finegray.fit(X_train, y_train)
finegray_inc_probs = finegray.predict_cumulative_incidence(X, times=times)

rsf = RSFEstimator()
rsf.fit(X_train, y_train)
rsf_inc_probs = rsf.predict_cumulative_incidence(X, times=times)


# %%
fig, ax = plt.subplots(ncols=n_events + 1, figsize=(15, 5))

for event_id in range(n_events + 1):
    inc_prob_aj = aalen_incidence_probs[:, event_id, :].mean(axis=0)
    ax[event_id].plot(times, inc_prob_aj, label="AJ")

    inc_prob_deephit = deephit_inc_probs[:, event_id, :].mean(axis=0)
    ax[event_id].plot(times, inc_prob_deephit, label="DeepHit")

    inc_prob_finegray = finegray_inc_probs[:, event_id, :].mean(axis=0)
    ax[event_id].plot(times, inc_prob_finegray, label="Fine & Gray")

    inc_prob_rsf = rsf_inc_probs[:, event_id, :].mean(axis=0)
    ax[event_id].plot(times, inc_prob_rsf, label="RSF")

    inc_prob_sb = survivalboost_inc_probs[:, event_id, :].mean(axis=0)
    ax[event_id].plot(times, inc_prob_sb, label="SurvivalBoost")

    ax[event_id].set_title("Incidence function for event {}".format(event_id))

plt.legend()
plt.savefig("uncalibrated_models.pdf", format="pdf")
plt.show()


# %%
def compute_ft_finfty(model, X_conf, y_conf):
    f_t = [
        model.predict_cumulative_incidence(
            X_conf.iloc[i : i + 1], times=np.array([y_conf["duration"].iloc[i]])
        )
        for i in range(len(X_train))
    ]
    f_t = np.asarray(f_t).reshape(len(X_conf), n_events + 1, 1)
    f_infty = model.predict_cumulative_incidence(
        X_conf, times=np.array([y_conf["duration"].max()])
    )
    return f_t, f_infty


for model in [aalen, deephit, survivalboost]:  # ,  finegray, rsf]:
    f_t, f_infty = compute_ft_finfty(model, X_conf, y_conf)
    s_t = f_t[:, 0, :]
    for event_id in range(1, 4):
        fk_t = f_t[:, event_id, :]
        fk_infty = f_infty[:, event_id, :]

        final_binning = d_calibration(
            fk_t, fk_infty, s_t, y_conf, event_of_interest=event_id
        )
        final_binning.plot(label=model.__class__.__name__)
        plt.plot(
            range(1, 101),
            np.linspace(0, 1, 100),
            linestyle="--",
            color="black",
            label="Perfect calibration",
        )
        plt.title(
            f"{model.__class__.__name__} D-calibration with censoring for event"
            f" {event_id}"
        )
        plt.legend()
        plt.show()
# %%
times = np.quantile(y_conf["duration"], np.linspace(0, 1, 100))

for model in [aalen]:  # , deephit, finegray, rsf, survivalboost]:
    f_k_conf = model.predict_cumulative_incidence(X_conf, times=times)
    cal, diff = aj_calibration(y_conf, times, f_k_conf, return_diff_at_t=True)
    print(f"{model.__class__.__name__} AJ calibration: {cal}")
# %%
c_indexes = {}
ibss = {}
accuracy_in_times = {}
for model in [aalen, deephit, survivalboost]:
    predictions = model.predict_cumulative_incidence(X_test, times=times)
    predictions_recalibrated = recalibrate_incidence_functions(
        model, X_conf, y_conf, X_test
    )

    c_indexes[model.__class__.__name__] = []
    ibss[model.__class__.__name__] = []
    c_indexes[model.__class__.__name__ + "_recalibrated"] = []
    ibss[model.__class__.__name__ + "_recalibrated"] = []

    for event_id in range(1, n_events + 1):
        c_index = concordance_index_incidence(
            y_test,
            predictions[:, event_id, :],
            y_train=y_train,
            time_grid=times,
            event_of_interest=event_id,
            ipcw_estimator="km",
        )
        c_indexes[model.__class__.__name__].append(c_index)

        ibs = integrated_brier_score_incidence(
            y_test=y_test,
            y_pred=predictions[:, event_id, :],
            y_train=y_train,
            times=times,
            event_of_interest=event_id,
        )
        ibss[model.__class__.__name__].append(ibs)

        c_index = concordance_index_incidence(
            y_test,
            predictions_recalibrated[:, event_id, :],
            y_train=y_train,
            time_grid=times,
            event_of_interest=event_id,
            ipcw_estimator="km",
        )
        c_indexes[model.__class__.__name__ + "_recalibrated"].append(c_index)

        ibs = integrated_brier_score_incidence(
            y_test=y_test,
            y_pred=predictions_recalibrated[:, event_id, :],
            y_train=y_train,
            times=times,
            event_of_interest=event_id,
        )
        ibss[model.__class__.__name__ + "_recalibrated"].append(ibs)

    accuracy_in_times[model.__class__.__name__] = accuracy_in_time(
        y_test,
        predictions,
        time_grid=times,
    )
    accuracy_in_times[model.__class__.__name__ + "_recalibrated"] = accuracy_in_time(
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
