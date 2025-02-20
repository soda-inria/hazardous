# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from hazardous._km_sampler import _KaplanMeierSampler, _AalenJohansenSampler
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from lifelines import CoxPHFitter
from scipy.interpolate import interp1d

# %%
# KM-calibration without censoring


def km_cal(y, times, surv_prob_at_conf, return_diff_at_t=False):
    """
    Args:
        y (n_samples, 2): samples to fit the KM estimator
        times (array(n_times, )): array of times t at which to calculate the calibration
        surv_prob_at_conf (array(n_conf, n_times)): survival predictions at time t for
        D_{conf}

    Returns:
    """
    kaplan_sampler = _KaplanMeierSampler()
    kaplan_sampler.fit(y)
    surv_func = kaplan_sampler.survival_func_

    t_max = max(times)

    # global surv prob from KM
    surv_probs_KM = surv_func(times)
    # global surv prob from estimator
    surv_probs = surv_prob_at_conf.mean(axis=0)

    # Calculate calibration by integrating over times and
    # taking the difference between the survival probabilities
    # at time t and the survival probabilities at time t from KM
    diff_at_t = surv_probs - surv_probs_KM

    KM_cal = np.trapz(diff_at_t**2, times) / t_max
    if return_diff_at_t:
        return KM_cal, diff_at_t
    return KM_cal


def recalibrate_survival_function(
    X, y, X_conf, times, estimator=None, surv_probs=None, surv_probs_conf=None
):
    """
    Args:
        X (n_conf, n_features): samples to recalibrate the estimator
        y (n_conf, 2): target
        estimator (BaseEstimator): trained estimator
        times (n_times): times at which to calculate the calibration
            and recalibrate the survival function

    Returns:
        estimator_calibrated: _description_
    """

    if estimator is None and (surv_probs is None or surv_probs_conf is None):
        raise ValueError(
            "Either estimator or (surv_probs and surv_probs_conf) must be provided"
        )

    # Calculate the survival probabilities to compute the calibration
    if surv_probs is None:
        if not hasattr(estimator, "predict_survival_function"):
            raise ValueError("Estimator must have a predict_survival_function method")

        surv_probs = estimator.predict_survival_function(X, times)
        surv_probs_conf = estimator.predict_survival_function(X_conf, times)

    # Calculate the calibration
    diff_at_t = km_cal(y, times, surv_probs_conf, return_diff_at_t=True)[1]
    surv_probs_calibrated = surv_probs - diff_at_t

    # Recalibrate the survival function
    return interp1d(
        x=times,
        y=surv_probs_calibrated,
        kind="previous",
        bounds_error=False,
        fill_value="extrapolate",
    )


# %%
n_samples = 10000
X, y = make_synthetic_competing_weibull(
    n_samples=n_samples,
    return_X_y=True,
    n_events=1,
    censoring_relative_scale=0.5,
    random_state=0,
)
sns.histplot(data=y, x="duration", hue="event", bins=50, kde=True)
plt.show()
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.5)
X_train, X_conf, y_train, y_conf = train_test_split(
    X_train, y_train, random_state=0, test_size=0.5
)


kaplan_sampler = _KaplanMeierSampler()
kaplan_sampler.fit(y_train)

surv_func = kaplan_sampler.survival_func_
# Test if KM is KM-calibrated
times = kaplan_sampler.unique_times_
km_km = km_cal(
    y_conf,
    times,
    np.repeat(surv_func(times), len(y), axis=0).reshape(len(times), len(y)).T,
)

plt.plot(times, surv_func(times), label="KM")
plt.legend()
plt.title(km_km)
plt.show()

# %%
# Test if Cox is KM calibrated

cox = CoxPHFitter()
cox.fit(y_train, duration_col="duration", event_col="event")
cox_surv_probs = cox.predict_survival_function(X_test).T
times_cox = cox_surv_probs.columns
km_cox = km_cal(y_conf, cox_surv_probs.columns, cox_surv_probs.values)

plt.plot(times, surv_func(times), label="KM, km_cal = {}".format(km_km))
plt.plot(cox_surv_probs.mean(axis=0), label="Cox, km_cal = {}".format(km_cox))
plt.legend()
plt.title("KM vs Cox: marginal survival function")
plt.show()


# %%

cox_probs_recalibrated = recalibrate_survival_function(
    X_train,
    y_train,
    X_conf,
    times_cox,
    surv_probs=cox_surv_probs,
    surv_probs_conf=cox.predict_survival_function(X_test).T,
)
km_cox_recal = km_cal(y_train, times, cox_probs_recalibrated(times))

# %%
plt.plot(times, surv_func(times), label="KM, km_cal = {}".format(km_km))
plt.plot(cox_surv_probs.mean(axis=0), label="Cox, km_cal = {}".format(km_cox))
plt.plot(
    times,
    cox_probs_recalibrated(times).mean(axis=0),
    label="Cox recalibrated, km_cal = {}".format(km_cox_recal),
)
plt.legend()
plt.title("KM vs Cox: marginal survival function")
plt.show()

# %%
# Test if DeepHit is calibrated (hopefully not)
from pycox.models import DeepHit


deephit = DeepHit()
deephit.fit(X_train, y_train, batch_size=256, epochs=10, verbose=True)


# %%


# %%
def d_calibration(
    inc_prob_t,
    inc_prob_infty,
    n_buckets,
    inc_prob_t_censor=None,
    inc_prob_infty_censor=None,
):
    buckets = np.linspace(0, 1, n_buckets + 1)
    # import ipdb; ipdb.set_trace()
    event_bins = np.digitize(inc_prob_t / inc_prob_infty, buckets, right=True)
    event_bins = np.clip(event_bins, 1, n_buckets)
    event_binning = pd.DataFrame(
        np.unique(event_bins, return_counts=True), index=["buckets", "count_event"]
    ).T
    if inc_prob_t_censor is None:
        return event_binning.set_index("buckets") / len(inc_prob_t)

    df = pd.DataFrame(inc_prob_t_censor / inc_prob_infty_censor, columns=["c"])
    for buck in range(1, n_buckets + 1):
        li = buckets[buck - 1]
        li1 = buckets[buck]
        df[f"{buck}"] = 0.0
        df.loc[df["c"] <= li, f"{buck}"] = li1 - li
        df.loc[((df["c"] > li) & (df["c"] <= li1)), f"{buck}"] = li1 - df["c"]
        df[f"{buck}"] /= 1 - df["c"]

    event_binning["censored_count"] = df.iloc[:, 1:].sum(axis=0).values

    event_binning.set_index("buckets", inplace=True)
    final_binning = event_binning[["count_event", "censored_count"]].sum(axis=1)
    return pd.DataFrame(final_binning, columns=["count_event"]) / (
        len(inc_prob_at_t) + len(inc_prob_t_censor)
    )


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

# %%
