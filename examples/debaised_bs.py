# Here, we create a synthetic dataset, either with a censoring independant
# of the covariates of dependant with a rate given by the user.
# We train a GBI to obtain some prediction. Givent a time grid, we compute
# the true Brier score (with the real distribution of the censoring) and the BS
# with an estimate of the proba of censoring (Kapplan Meier).
#

# %%
import numpy as np
import pandas as pd

from hazardous.data._competing_weibull import (
    make_complex_features_with_sparse_matrix,
)


event_of_interest = 1
seed = 0
n_samples = 3_000

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
import seaborn as sns
from matplotlib import pyplot as plt

from hazardous.data._competing_weibull import _censor


def plot_events(y, kind):
    censoring_rate = int((y["event"] == 0).mean() * 100)
    ax = sns.histplot(
        y,
        x="duration",
        hue="event",
        palette="magma",
        multiple="stack",
    )
    title = (
        f"Duration distributions when censoring is {kind} of X, {censoring_rate=!r}%"
    )
    ax.set(title=title)


bunch = _censor(
    y_uncensored,
    independent_censoring=True,
    X=X,
    features_censoring_rate=0.5,
    censoring_relative_scale=1,
    random_state=seed,
)
y_censored_indep = bunch.y_censored
shape_censoring_indep = bunch.shape_censoring
scale_censoring_indep = bunch.scale_censoring
plot_events(y_censored_indep, kind="independent")


# %%
from hazardous.data._competing_weibull import _censor


bunch = _censor(
    y_uncensored,
    independent_censoring=False,
    X=X,
    features_censoring_rate=0.5,
    censoring_relative_scale=1,
    random_state=seed,
)
y_censored_dep = bunch.y_censored
shape_censoring_dep = bunch.shape_censoring
scale_censoring_dep = bunch.scale_censoring
plot_events(y_censored_dep, kind="dependent")

# %%

from lifelines import AalenJohansenFitter


def plot_marginal_incidence(y_censored, y_uncensored, kind):
    aj = AalenJohansenFitter(calculate_variance=False, seed=seed)
    n_events = y_uncensored["event"].max()

    censoring_fraction = (y_censored["event"] == 0).mean()

    # Compute the estimate of the CIFs on a coarse grid.
    _, axes = plt.subplots(figsize=(12, 5), ncols=n_events, sharey=True)

    plt.suptitle(
        f"Cause-specific cumulative incidence functions, {kind}"
        f" ({censoring_fraction:.1%} censoring)"
    )

    for event_id, ax in enumerate(axes, 1):
        aj.fit(
            y_uncensored["duration"],
            y_uncensored["event"],
            event_of_interest=event_id,
        )
        aj.plot(label="Aalen-Johansen_uncensored", ax=ax)

        aj.fit(
            y_censored["duration"],
            y_censored["event"],
            event_of_interest=event_id,
        )
        aj.plot(label="Aalen-Johansen_censored", ax=ax)

        ax.set_xlim(0, 8_000)
        ax.set_ylim(0, 0.5)
        ax.set_title(f"{event_id=!r}")
    plt.legend()


plot_marginal_incidence(y_censored_indep, y_uncensored, kind="independent")
# %%

plot_marginal_incidence(y_censored_dep, y_uncensored, kind="dependent")

# %%
from hazardous import GradientBoostingIncidence

from hazardous.metrics._brier_score import (
    brier_score_incidence,
    brier_score_incidence_oracle,
)


def plot_brier_scores_comparisons(X, y, shape, scale, kind, event_of_interest=1):
    gbi = GradientBoostingIncidence(
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
    gbi.fit(X, y)
    y_pred = gbi.predict_cumulative_incidence(X)

    time_grid = gbi.time_grid_
    debiased_bs_scores = brier_score_incidence_oracle(
        y_train=y,
        y_test=y,
        y_pred=y_pred,
        times=time_grid,
        shape_censoring=shape,
        scale_censoring=scale,
        event_of_interest=event_of_interest,
    )

    bs_scores = brier_score_incidence(
        y_train=y,
        y_test=y,
        y_pred=y_pred,
        times=time_grid,
        event_of_interest=event_of_interest,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_grid, debiased_bs_scores, label="from true distrib")
    ax.plot(time_grid, bs_scores, label="from estimate distrib with km")
    ax.set(
        title=f"Time-varying Brier score, {kind} censoring",
    )
    ax.legend()


plot_brier_scores_comparisons(
    X,
    y_censored_indep,
    shape=shape_censoring_indep,
    scale=scale_censoring_indep,
    kind="independent",
    event_of_interest=event_of_interest,
)

# %%

plot_brier_scores_comparisons(
    X,
    y_censored_dep,
    shape=shape_censoring_dep,
    scale=scale_censoring_dep,
    kind="dependent",
    event_of_interest=event_of_interest,
)

# %%

from hazardous._ipcw import IPCWEstimator, IPCWCoxEstimator, IPCWSampler


def plot_censoring_survival_proba(
    X,
    y_censored,
    shape_censoring,
    scale_censoring,
    kind,
):
    t_max = y_uncensored["duration"].max()
    time_grid = np.linspace(0, t_max, 100)

    estimator_marginal = IPCWEstimator().fit(y_censored)
    g_hat_marginal = estimator_marginal.compute_censoring_survival_proba(time_grid)

    estimator_conditional = IPCWCoxEstimator().fit(y_censored, X=X)

    sampler = IPCWSampler(
        shape=shape_censoring,
        scale=scale_censoring,
    ).fit(y_censored)

    g_hat_conditional, g_star = [], []
    for time_step in time_grid:
        time_step = np.full(y_censored.shape[0], fill_value=time_step)

        g_star_ = sampler.compute_censoring_survival_proba(time_step)
        g_star.append(g_star_.mean())

        g_hat_conditional_ = estimator_conditional.compute_censoring_survival_proba(
            time_step,
            X=X,
        )
        g_hat_conditional.append(g_hat_conditional_.mean())

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_grid, g_hat_marginal, label="$\hat{G}$ with KM")
    ax.plot(time_grid, g_hat_conditional, label="$\hat{G}$ with Cox")
    ax.plot(time_grid, g_star, label="$G^*$")
    ax.set_title(f"Censoring survival proba, with {kind} censoring")
    plt.legend()


plot_censoring_survival_proba(
    X,
    y_censored_indep,
    shape_censoring_indep,
    scale_censoring_indep,
    kind="independent",
)

# %%

plot_censoring_survival_proba(
    X,
    y_censored_dep,
    shape_censoring_dep,
    scale_censoring_dep,
    kind="dependent",
)

# %%


def plot_ipcw(X, y_uncensored, y_censored, shape_censoring, scale_censoring, kind):
    n_samples = y_censored.shape[0]
    t_max = y_uncensored["duration"].max()
    time_grid = np.linspace(0, t_max, 100)

    estimator_marginal = IPCWEstimator().fit(y_censored)
    ipcw_pred_marginal = estimator_marginal.compute_ipcw_at(time_grid)

    estimator_conditional = IPCWCoxEstimator().fit(y_censored, X=X)

    sampler = IPCWSampler(
        shape=shape_censoring,
        scale=scale_censoring,
    ).fit(y_censored)

    individual_ipcw_sampled, ipcw_sampled, ipcw_pred_conditional = [], [], []
    n_indiv = min(100, n_samples)
    for t in time_grid:
        t = np.full(n_samples, fill_value=t)

        ipcw_sampled_ = sampler.compute_ipcw_at(t)
        ipcw_sampled.append(ipcw_sampled_.mean())
        individual_ipcw_sampled.append(ipcw_sampled_[:n_indiv])

        ipcw_pred_conditional_ = estimator_conditional.compute_ipcw_at(
            t,
            X=X,
        )
        ipcw_pred_conditional.append(ipcw_pred_conditional_.mean())
    individual_ipcw_sampled = np.vstack(individual_ipcw_sampled).T

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_grid, ipcw_sampled, label="$1/G^*$")
    ax.plot(time_grid, ipcw_pred_marginal, label="$1/\hat{G}$, using KM")
    ax.plot(time_grid, ipcw_pred_conditional, label="$1/\hat{G}$, using Cox")
    ax.set_title(f"IPCW, with {kind} censoring")
    plt.legend()

    _, ax = plt.subplots(figsize=(8, 4))

    for idx in range(n_indiv):
        ax.plot(time_grid, individual_ipcw_sampled[idx])
    ax.set_title(f"Weights for each sample, {kind} censoring")


plot_ipcw(
    X,
    y_uncensored,
    y_censored_indep,
    shape_censoring=shape_censoring_indep,
    scale_censoring=scale_censoring_indep,
    kind="independent",
)

# %%

plot_ipcw(
    X,
    y_uncensored,
    y_censored_dep,
    shape_censoring=shape_censoring_dep,
    scale_censoring=scale_censoring_dep,
    kind="dependent",
)

# %%
