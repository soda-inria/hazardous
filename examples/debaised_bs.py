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
    features_censoring_rate=0.1,
    censoring_relative_scale=0.6,
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
from matplotlib import pyplot as plt

y_censored_dep, shape_censoring_dep, scale_censoring_dep = _censor(
    y_uncensored,
    independent_censoring=False,
    X=X,
    features_censoring_rate=0.7,
    censoring_relative_scale=1,
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
plt.show()

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

bs_scores_indep = brier_score_incidence(
    y_train=y_censored_indep,
    y_test=y_censored_indep,
    y_pred=y_pred_indep,
    times=time_grid,
    event_of_interest=event_of_interest,
)

bs_scores_dep = brier_score_incidence(
    y_train=y_censored_dep,
    y_test=y_censored_dep,
    y_pred=y_pred_dep,
    times=time_grid,
    event_of_interest=event_of_interest,
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_grid, scores_indep, label="BS with the true distribution of censoring")
ax.plot(
    time_grid,
    bs_scores_indep,
    label="BS with the estimate distribution of censoring with KM",
)

ax.set(
    title="Time-varying Brier score, Independent censoring",
)
ax.legend()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_grid, scores_dep, label="BS with the true distribution of censoring")
ax.plot(
    time_grid,
    bs_scores_dep,
    label="BS with the estimate distribution of censoring with KM",
)

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


# Compute the estimate of the CIFs on a coarse grid.
for y, censor_dep in zip(
    [y_censored_indep, y_censored_dep], ["Inpedendant", "Dependant"]
):
    censoring_fraction = (y["event"] == 0).mean()
    _, axes = plt.subplots(figsize=(12, 4), ncols=n_events, sharey=True)

    plt.suptitle(
        f"Cause-specific cumulative incidence functions, {censor_dep}"
        f" ({censoring_fraction:.1%} censoring)"
    )

    for event_id, ax in enumerate(axes, 1):
        aj.fit(
            y_uncensored["duration"], y_uncensored["event"], event_of_interest=event_id
        )
        aj.plot(label="AJ with uncensored data", ax=ax)

        aj.fit(
            y["duration"],
            y["event"],
            event_of_interest=event_id,
        )
        aj.plot(label="AJ with censored data", ax=ax)

        ax.set_xlim(0, 8_000)
        ax.set_ylim(0, 0.5)

# %%

from hazardous._ipcw import IPCWEstimator, IPCWSampler

for y, (shape, scale), censor_dep in zip(
    [y_censored_indep, y_censored_dep],
    [
        (shape_censoring_indep, scale_censoring_indep),
        (shape_censoring_dep, scale_censoring_dep),
    ],
    ["Independent", "Dependent"],
):
    ipcw_est = IPCWEstimator().fit(y)
    ipcw_y_est = ipcw_est.compute_ipcw_at(time_grid)

    ipcw_true_distrib = IPCWSampler(
        shape=shape,
        scale=scale,
    ).fit(y)

    if censor_dep == "Independent":
        ipcw_y_true_distrib = ipcw_true_distrib.compute_ipcw_at(time_grid)
    else:
        ipcw_y_true_distribs = []
        for t in time_grid:
            ipcw_y_true_distrib = ipcw_true_distrib.compute_ipcw_at(
                times=np.full(shape=n_samples, fill_value=t)
            )
            ipcw_y_true_distribs.append(ipcw_y_true_distrib)
        fig, ax = plt.subplots(figsize=(8, 4))
        ipcw_y_true_distribs = np.array(ipcw_y_true_distribs)
        for i in range(ipcw_y_true_distribs.shape[1]):
            ax.plot(time_grid, ipcw_y_true_distribs[:, i])
        ipcw_y_true_distrib = ipcw_y_true_distribs.mean(axis=1)
        ax.set_title(f"Weights for each sample, {censor_dep} censoring")
        ax.plot()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_grid, ipcw_y_est, label="Estimated weights (KM)")
    ax.plot(time_grid, ipcw_y_true_distrib, label="True weights (mean)")
    plt.legend()
    ax.set_title(f"Weights, {censor_dep} censoring")
    ax.plot()


# %%


# ipcw_sample_dep = IPCWSampler(
#     shape=shape_censoring_dep,
#     scale=scale_censoring_dep,
# ).fit(y_censored_dep)
# ipcw_y_sample_dep = ipcw_sample_dep.compute_ipcw_at(time_grid_rescale)

# %%
