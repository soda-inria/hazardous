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

from hazardous.data._competing_weibull import (
    make_complex_features_with_sparse_matrix,
)


independent_censoring = False
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


y_censored, shape_censoring, scale_censoring = _censor(
    y_uncensored,
    independent_censoring=independent_censoring,
    X=X,
    features_censoring_rate=0.5,
    censoring_relative_scale=40,
    random_state=seed,
)

ax = sns.histplot(
    y_censored,
    x="duration",
    hue="event",
    palette="magma",
    multiple="stack",
)
ax.set(title="Duration distributions")

# %%
from hazardous import GradientBoostingIncidence


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
gbi.fit(X, y_censored)

# %%
from matplotlib import pyplot as plt

from hazardous.metrics._brier_score import (
    brier_score_incidence,
    brier_score_true_probas_incidence,
)


time_grid = gbi.time_grid_
y_pred = gbi.predict_cumulative_incidence(X, times=time_grid)

scores = brier_score_true_probas_incidence(
    y_train=y_censored,
    y_test=y_censored,
    y_pred=y_pred,
    times=time_grid,
    shape_censoring=shape_censoring,
    scale_censoring=scale_censoring,
    event_of_interest=event_of_interest,
)

bs_scores = brier_score_incidence(
    y_train=y_censored,
    y_test=y_censored,
    y_pred=y_pred,
    times=time_grid,
    event_of_interest=event_of_interest,
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_grid, scores, label="from true distrib")
ax.plot(time_grid, bs_scores, label="from estimate distrib with km")

ax.set(
    title="Time-varying Brier score",
)
ax.legend()
# %%
