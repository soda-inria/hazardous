# %%
# D-calibration tests
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from hazardous._km_sampler import _KaplanMeierSampler, _AalenJohansenSampler
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.calibration._d_calibration import d_calibration

# %%
n_samples = 3000
X, y = make_synthetic_competing_weibull(
    n_samples=n_samples, return_X_y=True, n_events=3, censoring_relative_scale=0
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
t_maxs = y_any_event["duration"].max() * np.ones(n_samples)
inc_prob_at_infty = 1 - surv_func(t_maxs)

final_binning = d_calibration(inc_prob_at_t, inc_prob_at_infty, 20)
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
t_maxs = y_any_event["duration"].max() * np.ones(n_samples)
inc_prob_at_infty = 1 - surv_func(t_maxs)

final_binning = d_calibration(inc_prob_at_t, inc_prob_at_infty, 20)
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
    y_event_i = y[y["event"] == i]
    inc_prob_at_t = inc_func(y_event_i["duration"])
    t_maxs = y_event_i["duration"].max() * np.ones(len(y_event_i))
    inc_prob_at_infty = inc_func(t_maxs)
    final_binning = d_calibration(inc_prob_at_t, inc_prob_at_infty, 10)
    sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
    plt.title(f"Aalen-Johansen D-calibration without censoring for event {i}")
    plt.show()

# %%
# Adding censoring to the synthetic data

X, y = make_synthetic_competing_weibull(
    n_samples=5000, return_X_y=True, n_events=3, censoring_relative_scale=1.5
)


# %%
# Kaplan-Meier D-calibration with censoring

kaplan_sampler = _KaplanMeierSampler()
y_any_event = y.copy()
y_any_event["event"] = y_any_event["event"] > 0
kaplan_sampler.fit(y_any_event)

surv_func = kaplan_sampler.survival_func_
inc_prob_t = 1 - surv_func(y_any_event[y_any_event["event"]]["duration"])
t_maxs = y_any_event["duration"].max() * np.ones(len(y[y["event"] > 0]))
inc_prob_infty = 1 - surv_func(t_maxs)

inc_prob_t_censor = 1 - surv_func(y_any_event[~y_any_event["event"]]["duration"])
t_maxs = y_any_event["duration"].max() * np.ones(
    len(y_any_event[~y_any_event["event"]])
)
inc_prob_infty_censor = 1 - surv_func(t_maxs)

final_binning = d_calibration(
    inc_prob_t, inc_prob_infty, 20, inc_prob_t_censor, inc_prob_infty_censor
)
sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
plt.title("Kaplan-Meier D-calibration with censoring")
plt.show()


# %%
# Aalen-Johansen D-calibration with censoring
# survival function

aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y)

surv_func = aalen_sampler.survival_func_
inc_prob_at_t = 1 - surv_func(y[y["event"] > 0]["duration"])
t_maxs = y_any_event["duration"].max() * np.ones(len(y[y["event"] > 0]))
inc_prob_at_infty = 1 - surv_func(t_maxs)

inc_prob_t_censor = 1 - surv_func(y_any_event[~y_any_event["event"]]["duration"])
t_maxs = y_any_event["duration"].max() * np.ones(
    len(y_any_event[~y_any_event["event"]])
)
inc_prob_infty_censor = 1 - surv_func(t_maxs)

final_binning = d_calibration(
    inc_prob_at_t, inc_prob_at_infty, 20, inc_prob_t_censor, inc_prob_infty_censor
)
sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
plt.title("Aalen-Johansen survival function D-calibration without censoring")
plt.show()
# %%
# Aalen-Johansen D-calibration with censoring
# incidence functions
aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y)

for i in range(1, 4):
    inc_func = aalen_sampler.incidence_func_[i]
    y_event_i = y[y["event"] == i]

    inc_prob_at_t = inc_func(y_event_i["duration"])
    t_maxs = y["duration"].max() * np.ones(len(y_event_i))
    inc_prob_at_infty = inc_func(t_maxs)

    y_censor = y[y["event"] == 0]
    inc_prob_t_censor = inc_func(y_censor["duration"])
    t_maxs = y["duration"].max() * np.ones(len(y_censor))
    inc_prob_infty_censor = inc_func(t_maxs)

    final_binning = d_calibration(
        inc_prob_at_t, inc_prob_at_infty, 10, inc_prob_t_censor, inc_prob_infty_censor
    )
    sns.barplot(data=final_binning.reset_index(), x="buckets", y="count_event")
    plt.title(f"Aalen-Johansen D-calibration without censoring for event {i}")
    plt.show()
