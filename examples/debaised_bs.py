# Here, we create a synthetic dataset, either with a censoring independant
# of the covariates of dependant with a rate given by the user.
# We train a GBI to obtain some prediction. Givent a time grid, we compute
# the true Brier score (with the real distribution of the censoring) and the BS
# with an estimate of the proba of censoring (Kapplan Meier).
#

# %%
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.base import clone

from hazardous.data._competing_weibull import (
    make_complex_features_with_sparse_matrix,
)


event_of_interest = 1
seed = 0
n_samples = 30_000

X, event_durations, duration_argmin = make_complex_features_with_sparse_matrix(
    n_events=3,
    n_samples=n_samples,
    base_scale=1_000,
    n_features=10,
    features_rate=0.3,
    degree_interaction=2,
    random_state=seed,
)

y_uncensored = pd.DataFrame(
    dict(
        event=duration_argmin + 1,
        duration=event_durations[duration_argmin, np.arange(n_samples)],
    )
)

# %%
from hazardous.data._competing_weibull import _censor


y_censored_indep, shape_censoring_indep, scale_censoring_indep = _censor(
    y_uncensored,
    independent_censoring=True,
    X=X,
    features_censoring_rate=0.5,
    censoring_relative_scale=0.5,
    random_state=seed,
)

ax = sns.histplot(
    y_censored_indep,
    x="duration",
    hue="event",
    palette="magma",
    multiple="stack",
)
ax.set(title="Duration distributions when censoring is independent of X")


# %%
from hazardous.data._competing_weibull import _censor


y_censored_dep, shape_censoring_dep, scale_censoring_dep = _censor(
    y_uncensored,
    independent_censoring=False,
    X=X,
    features_censoring_rate=0.5,
    censoring_relative_scale=0.5,
    random_state=seed,
)

ax = sns.histplot(
    y_censored_dep,
    x="duration",
    hue="event",
    palette="magma",
    multiple="stack",
)
ax.set(title="Duration distributions when censoring is dependent of X")


# %%
from hazardous import GradientBoostingIncidence


gbi_indep = GradientBoostingIncidence(
    learning_rate=0.1,
    n_iter=20,
    max_leaf_nodes=15,
    hard_zero_fraction=0.1,
    min_samples_leaf=5,
    loss="ibs",
    show_progressbar=False,
    random_state=seed,
    event_of_interest=event_of_interest,
)
gbi_indep.fit(X, y_censored_indep)

gbi_dep = clone(gbi_indep)
gbi_dep.fit(X, y_censored_dep)

# %%
from matplotlib import pyplot as plt

from hazardous.metrics._brier_score import (
    brier_score_incidence,
    brier_score_true_probas_incidence,
)


time_grid = gbi_indep.time_grid_
y_pred_indep = gbi_indep.predict_cumulative_incidence(X, times=time_grid)

scores_indep = brier_score_true_probas_incidence(
    y_train=y_censored_indep,
    y_test=y_censored_indep,
    y_pred=y_pred_indep,
    times=time_grid,
    shape_censoring=shape_censoring_indep,
    scale_censoring=scale_censoring_indep,
    event_of_interest=event_of_interest,
)

y_pred_dep = gbi_dep.predict_cumulative_incidence(X, times=time_grid)

scores_dep = brier_score_true_probas_incidence(
    y_train=y_censored_dep,
    y_test=y_censored_dep,
    y_pred=y_pred_dep,
    times=time_grid,
    shape_censoring=shape_censoring_dep,
    scale_censoring=scale_censoring_dep,
    event_of_interest=event_of_interest,
)

bs_scores = brier_score_incidence(
    y_train=y_censored_indep,
    y_test=y_censored_indep,
    y_pred=y_pred_indep,
    times=time_grid,
    event_of_interest=event_of_interest,
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_grid, scores_indep, label="from true distrib")
ax.plot(time_grid, bs_scores, label="from estimate distrib with km")

ax.set(
    title="Time-varying Brier score, Independent censoring",
)
ax.legend()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_grid, scores_dep, label="from true distrib")
ax.plot(time_grid, bs_scores, label="from estimate distrib with km")

ax.set(
    title="Time-varying Brier score, Dependent censoring",
)
ax.legend()

# %%

from lifelines import AalenJohansenFitter


aj = AalenJohansenFitter(calculate_variance=False, seed=seed)
n_events = y_uncensored["event"].max()
t_max = y_uncensored["duration"].max()

coarse_timegrid = np.linspace(0, t_max, num=100)
censoring_fraction = (y_censored_indep["event"] == 0).mean()


# Compute the estimate of the CIFs on a coarse grid.
_, axes = plt.subplots(figsize=(12, 4), ncols=n_events, sharey=True)

plt.suptitle(
    "Cause-specific cumulative incidence functions, Independent"
    f" ({censoring_fraction:.1%} censoring)"
)

for event_id, ax in enumerate(axes, 1):
    aj.fit(y_uncensored["duration"], y_uncensored["event"], event_of_interest=event_id)
    aj.plot(label="Aalen-Johansen_uncensored", ax=ax)

    aj.fit(
        y_censored_indep["duration"],
        y_censored_indep["event"],
        event_of_interest=event_id,
    )
    aj.plot(label="Aalen-Johansen_censored", ax=ax)

    ax.set_xlim(0, 8_000)
    ax.set_ylim(0, 0.5)
# %%

aj = AalenJohansenFitter(calculate_variance=False, seed=seed)
n_events = y_uncensored["event"].max()
t_max = y_uncensored["duration"].max()

coarse_timegrid = np.linspace(0, t_max, num=100)
censoring_fraction = (y_censored_dep["event"] == 0).mean()


# Compute the estimate of the CIFs on a coarse grid.
_, axes = plt.subplots(figsize=(12, 4), ncols=n_events, sharey=True)

plt.suptitle(
    "Cause-specific cumulative incidence functions, Dependent"
    f" ({censoring_fraction:.1%} censoring)"
)

for event_id, ax in enumerate(axes, 1):
    aj.fit(y_uncensored["duration"], y_uncensored["event"], event_of_interest=event_id)
    aj.plot(label="Aalen-Johansen_uncensored", ax=ax)

    aj.fit(
        y_censored_dep["duration"], y_censored_dep["event"], event_of_interest=event_id
    )
    aj.plot(label="Aalen-Johansen_censored", ax=ax)

    ax.set_xlim(0, 8_000)
    ax.set_ylim(0, 0.5)

# %%

from hazardous._ipcw import IPCWEstimator, IPCWSampler

ipcw_est = IPCWEstimator().fit(y_censored_indep)
ipcw_y_est = ipcw_est.compute_ipcw_at(time_grid)

ipcw_sample_indep = IPCWSampler(
    shape=shape_censoring_indep,
    scale=scale_censoring_indep,
).fit(y_censored_indep)
ipcw_y_sample_indep = ipcw_sample_indep.compute_ipcw_at(time_grid)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time_grid, ipcw_y_est, label="est")
ax.plot(time_grid, ipcw_y_sample_indep, label="sample indep")
# ax.plot(time_grid, ipcw_y_sample_dep, label="sample dep")
plt.legend()


# %%


# ipcw_sample_dep = IPCWSampler(
#     shape=shape_censoring_dep,
#     scale=scale_censoring_dep,
# ).fit(y_censored_dep)
# ipcw_y_sample_dep = ipcw_sample_dep.compute_ipcw_at(time_grid_rescale)

# %%
