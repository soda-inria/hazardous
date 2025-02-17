# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from hazardous._km_sampler import _KaplanMeierSampler, _AalenJohansenSampler
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

# %%


def d_calibration(inc_prob_at_t, c, n_buckets, inc_event=1):
    inc_event = max(inc_prob_at_t)
    buckets = np.linspace(0, inc_event, n_buckets + 1)
    event_bins = np.digitize(inc_prob_at_t, buckets, right=True)
    event_bins = np.clip(event_bins, 1, n_buckets)
    event_binning = pd.DataFrame(
        np.unique(event_bins, return_counts=True), index=["buckets", "count_event"]
    ).T

    if c is None:
        return event_binning.set_index("buckets") / len(inc_prob_at_t)

    df = pd.DataFrame(c, columns=["c"])
    for buck in range(1, n_buckets + 1):
        li = buckets[buck - 1]
        li1 = buckets[buck]
        df[f"{buck}"] = 0.0
        df.loc[df["c"] <= li, f"{buck}"] = li1 - li
        df.loc[((df["c"] > li) & (df["c"] <= li1)), f"{buck}"] = li1 - df["c"]
        df[f"{buck}"] /= 1 - c
        df[f"{buck}"] *= inc_event

    event_binning["censored_count"] = df.iloc[:, 1:].sum(axis=0).values

    event_binning.set_index("buckets", inplace=True)
    final_binning = event_binning[["count_event", "censored_count"]].sum(axis=1)
    return pd.DataFrame(final_binning, columns=["count_event"]) / (
        len(inc_prob_at_t) + len(c)
    )


# %%
X, y = make_synthetic_competing_weibull(
    n_samples=20000, return_X_y=True, n_events=3, censoring_relative_scale=0
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
inc_prob_at_t = 1 - surv_func(y_any_event[y_any_event["event"]]["duration"])

final_binning = d_calibration(inc_prob_at_t, None, 20)
sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
plt.title("Kaplan-Meier D-calibration without censoring")
plt.show()
# %%
# Aalen-Johansen D-calibration without censoring
# survival function

aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y)

surv_func = aalen_sampler.survival_func_
inc_prob_at_t = 1 - surv_func(y[y["event"] > 0]["duration"])

final_binning = d_calibration(inc_prob_at_t, None, 20)
sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
plt.title("Aalen-Johansen survival function D-calibration without censoring")
plt.show()
# %%
# Aalen-Johansen D-calibration without censoring
# incidence functions
aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y)

for i in range(1, 4):
    inc_func = aalen_sampler.incidence_func_[i]
    inc_prob_at_t = inc_func(y[y["event"] == i]["duration"])
    inc_event = len(y[y["event"] == i]) / y[y["event"] > 0].shape[0]
    final_binning = d_calibration(inc_prob_at_t, None, 10, inc_event)
    sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
    plt.title(f"Aalen-Johansen D-calibration without censoring for event {i}")
    plt.show()

# %%
# Adding censoring to the synthetic data

X, y = make_synthetic_competing_weibull(
    n_samples=50000, return_X_y=True, n_events=3, censoring_relative_scale=1.5
)


# %%
# Kaplan-Meier D-calibration with censoring

kaplan_sampler = _KaplanMeierSampler()
y_any_event = y.copy()
y_any_event["event"] = y_any_event["event"] > 0
kaplan_sampler.fit(y_any_event)

surv_func = kaplan_sampler.survival_func_
inc_prob_at_t = 1 - surv_func(y_any_event[y_any_event["event"]]["duration"])
inc_censor_at_t = 1 - surv_func(y_any_event[~y_any_event["event"]]["duration"])
final_binning = d_calibration(inc_prob_at_t, inc_censor_at_t, 10)
sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
plt.title("Kaplan-Meier D-calibration with censoring")
plt.show()
# %%
# Aalen-Johansen D-calibration with censoring
# incidence functions
aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y)

for i in range(1, 4):
    inc_func = aalen_sampler.incidence_func_[i]
    inc_prob_at_t = inc_func(y[y["event"] == i]["duration"])
    inc_event = len(y[y["event"] == i])
    p_censor = len(y[y["event"] == 0])
    inc_event = inc_event / (y.shape[0] - p_censor)
    c = inc_func(y_any_event[~y_any_event["event"]]["duration"])
    final_binning = d_calibration(inc_prob_at_t, c, 10, inc_event)
    sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
    plt.title(f"Aalen-Johansen D-calibration with censoring for event {i}")
    plt.show()
# %%
X, y = make_synthetic_competing_weibull(
    n_samples=10000, return_X_y=True, n_events=3, censoring_relative_scale=1.5
)

aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y)
# %%
i = 1
inc_func = aalen_sampler.incidence_func_[i]
inc_prob_at_t = inc_func(y[y["event"] == i]["duration"])
inc_event = len(y[y["event"] == i]) / y[y["event"] > 0].shape[0]
c = inc_func(y_any_event[~y_any_event["event"]]["duration"])
final_binning, event_binning, df = d_calibration(inc_prob_at_t, c, 10, inc_event)
# %%
event_binning
# %%
final_binning
# %%
