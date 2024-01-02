# %%

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from hazardous.data._competing_weibull import (
    make_synthetic_competing_weibull,
)


seed = 0
n_samples = 30_000
n_events = 3

X, y = make_synthetic_competing_weibull(
    n_events=n_events,
    n_samples=n_samples,
    base_scale=1_000,
    n_features=10,
    features_rate=0.3,
    degree_interaction=2,
    random_state=seed,
    independent_censoring=True,
    complex_features=True,
    return_X_y=True,
    target_rounding=None,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

sns.histplot(y_test, x="duration", hue="event", multiple="stack", palette="magma")


# %%
# When n_iter_before_feedback < n_iter, both GBI and DGBI should be identical.

from hazardous._gradient_boosting_incidence_debiased import (
    GradientBoostingIncidenceDebiased,
)

t_min, t_max = y_train["duration"].min(), y_train["duration"].max()
time_grid = np.linspace(t_min, t_max, 100)


gbi_debiased = GradientBoostingIncidenceDebiased(
    n_iter=100,
    n_iter_before_feedback=200,
    random_state=seed,
    max_leaf_nodes=5,
)
gbi_debiased.fit(X_train, y_train)
y_pred_debiased = gbi_debiased.predict_cumulative_incidence(X_test, time_grid)

fig, ax = plt.subplots()
for idx in range(y_pred_debiased.shape[2]):
    ax.plot(
        time_grid,
        y_pred_debiased[:, :, idx].mean(axis=0),
        label=gbi_debiased.classes_[idx],
    )
ax.plot(
    time_grid,
    y_pred_debiased.sum(axis=2).mean(axis=0),
    "k--",
)


# %%
from hazardous._gradient_boosting_incidence import GradientBoostingIncidence

y_pred_biased = []
for idx in range(1, n_events + 1):
    gbi = GradientBoostingIncidence(
        n_iter=100,
        max_leaf_nodes=5,
        learning_rate=0.03,
        event_of_interest=idx,
        random_state=seed,
    )
    gbi.fit(X_train, y_train)
    y_pred = gbi.predict_cumulative_incidence(X_test, time_grid)
    y_pred_biased.append(y_pred)


# %%

from lifelines import AalenJohansenFitter


fig, axes = plt.subplots(figsize=(7, 10), nrows=n_events)

for idx, ax in enumerate(axes):
    aj = (
        AalenJohansenFitter(seed=seed, calculate_variance=False)
        .fit(
            durations=y["duration"],
            event_observed=y["event"],
            event_of_interest=idx + 1,
        )
        .plot(ax=ax, label="AJ")
    )

    ax.plot(
        time_grid,
        y_pred_debiased[:, :, idx + 1].mean(axis=0),
        label="GBID",
    )
    ax.plot(
        time_grid,
        y_pred_biased[idx].mean(axis=0),
        label="GBI",
    )
    ax.plot()
    ax.set_title(f"event {idx}")
    ax.xaxis.label.set_visible(False)
    ax.legend()


# %%
