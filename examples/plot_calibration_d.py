# %%
# D-calibration tests
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from hazardous._km_sampler import _KaplanMeierSampler, _AalenJohansenSampler
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.calibration._d_calibration import d_calibration

# %%
n_samples = 3000
X, y = make_synthetic_competing_weibull(
    n_samples=n_samples, return_X_y=True, n_events=3, censoring_relative_scale=0
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sns.histplot(data=y, x="duration", hue="event", bins=50, kde=True)
plt.show()

# %%
# Kaplan-Meier D-calibration without censoring

kaplan_sampler = _KaplanMeierSampler()
y_any_event = y.copy()
y_any_event["event"] = y_any_event["event"] > 0
kaplan_sampler.fit(y_any_event)

surv_func = kaplan_sampler.survival_func_
fk_t = 1 - surv_func(y_any_event["duration"])
fk_infty = (1 - surv_func(y_any_event["duration"].max())) * np.ones(n_samples)
s_t = surv_func(y_any_event["duration"])

final_binning = d_calibration(
    fk_t, fk_infty, s_t, y_any_event["event"].astype(int).values
)
sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
plt.title("Kaplan-Meier D-calibration without censoring")
plt.show()
# %%
# Aalen-Johansen D-calibration without censoring
# incidence functions
aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y)

surv_func = aalen_sampler.survival_func_
s_t = surv_func(y["duration"])
events = y["event"].astype(int).values

for i in range(1, 4):
    inc_func = aalen_sampler.incidence_func_[i]

    fk_t = inc_func(y["duration"])
    fk_infty = (inc_func(y["duration"].max())) * np.ones(n_samples)
    s_t = surv_func(y["duration"])

    final_binning = d_calibration(fk_t, fk_infty, s_t, events, event_of_interest=i)
    sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
    plt.title(f"Aalen-Johansen D-calibration without censoring for event {i}")
    plt.show()

# %%
# Adding censoring to the synthetic data

n_samples = 3000
n_events = 3
X, y = make_synthetic_competing_weibull(
    n_samples=n_samples,
    return_X_y=True,
    n_events=n_events,
    censoring_relative_scale=1.5,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Aalen-Johansen D-calibration with censoring
# incidence functions
aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y)

surv_func = aalen_sampler.survival_func_
s_t = surv_func(y["duration"])
events = y["event"].astype(int).values

for i in range(1, 4):
    inc_func = aalen_sampler.incidence_func_[i]

    fk_t = inc_func(y["duration"])
    fk_infty = (inc_func(y["duration"].max())) * np.ones(n_samples)
    s_t = surv_func(y["duration"])

    final_binning = d_calibration(fk_t, fk_infty, s_t, events, event_of_interest=i)
    sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
    plt.title(f"Aalen-Johansen D-calibration with censoring for event {i}")
    plt.show()

# %%
# Testing for other models

from hazardous import SurvivalBoost

survboost = SurvivalBoost(n_iter=30, show_progressbar=False)
survboost.fit(X_train, y_train)

survboost_probs = survboost.predict_cumulative_incidence(X_test)

fig, ax = plt.subplots(ncols=n_events + 1, figsize=(15, 5))

for event_id in range(n_events + 1):
    inc_func = survboost_probs[:, event_id, :]
    ax[event_id].plot(survboost.time_grid_, inc_func.mean(axis=0), label="SB")
    ax[event_id].set_title("Incidence function for event {}".format(event_id))

plt.legend()
plt.show()
# %%
events = y_train["event"].astype(int).values

f_t = [
    survboost.predict_cumulative_incidence(
        X_train.iloc[i : i + 1].values, times=np.array([y_train["duration"].iloc[i]])
    )
    for i in range(len(X_train))
]
f_t = np.asarray(f_t).reshape(len(X_train), n_events + 1, 1)
f_infty = survboost.predict_cumulative_incidence(
    X_train, times=np.array([y_train["duration"].max()])
)

s_t = f_t[:, 0, :]

# %%
for event_id in range(1, 4):
    fk_t = f_t[:, event_id, :]
    fk_infty = f_infty[:, event_id, :]

    final_binning = d_calibration(
        fk_t, fk_infty, s_t, events, event_of_interest=event_id
    )
    sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
    plt.title(f"SurvivalBoost D-calibration with censoring for event {event_id}")
    plt.show()

# %%
