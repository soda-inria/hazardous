# %%

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

seed = 0
independent_censoring = False

bunch = make_synthetic_competing_weibull(
    n_events=1,
    n_samples=30_000,
    n_features=5,
    independent_censoring=independent_censoring,
    complex_features=True,
    target_rounding=None,
    censoring_relative_scale=1,
    random_state=seed,
)
bunch.X

# %%
X, y = bunch.X, bunch.y
y["duration"] = np.clip(y["duration"], a_min=None, a_max=7500)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape, y_train.shape

# %%
ax = sns.histplot(y, x="duration", hue="event", multiple="stack", palette="magma")
ax.set_title(f"Censoring rate {(y['event'] == 0).mean():.2%}")

# %%
t_min, t_max = y_train["duration"].min(), y_train["duration"].max()
time_grid = np.linspace(t_min, t_max, 100)

# %%
from hazardous._kaplan_meier import KaplanMeierEstimator

km = KaplanMeierEstimator().fit(y)
y_pred_marginal = km.predict_proba(time_grid)[:, 1]

# %%
from hazardous._gradient_boosting_incidence import GradientBoostingIncidence

gbi = GradientBoostingIncidence(n_iter=100).fit(X_train, y_train)
y_pred_gbi = gbi.predict_cumulative_incidence(X_test, times=time_grid)
y_pred_gbi.shape

# %%
from hazardous._gb_multi_incidence import GBMultiIncidence

gbmi = GBMultiIncidence(n_iter=100).fit(X_train, y_train)
y_pred_gbmi = gbmi.predict_cumulative_incidence(X_test, times=time_grid)
y_pred_gbmi.shape

# %%
fig, ax = plt.subplots()
ax.plot(time_grid, y_pred_marginal, label="KM")
ax.plot(time_grid, y_pred_gbi.mean(axis=0), label="GBI")
ax.plot(time_grid, y_pred_gbmi[1].mean(axis=0), label="GBMI")
ax.legend()
ax.set_title("Marginal any event incidence probability")
plt.grid()

# %%
from hazardous.metrics._brier_score import (
    brier_score_incidence_oracle,
    integrated_brier_score_incidence_oracle,
)

fig, ax = plt.subplots()

if not independent_censoring:
    shape_censoring = bunch.shape_censoring[X_test.index]
    scale_censoring = bunch.scale_censoring[X_test.index]

models = {
    "KM": np.vstack([y_pred_marginal] * y_test.shape[0]),
    "GBI": y_pred_gbi,
    "GBMI": y_pred_gbmi[1],
}

for name, y_pred in models.items():
    debiased_brier_score = brier_score_incidence_oracle(
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        times=time_grid,
        shape_censoring=shape_censoring,
        scale_censoring=scale_censoring,
    )

    idbs = integrated_brier_score_incidence_oracle(
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        times=time_grid,
        shape_censoring=shape_censoring,
        scale_censoring=scale_censoring,
    )

    ax.plot(time_grid, debiased_brier_score, label=f"{name}, IBS: {idbs:.3f}")

ax.set_title("Debiased brier score")
ax.legend()
plt.grid()
# %%
# Comparison of scores (using true incidence probabilities or estimated with KM)
# The scores are both representing Integrated Brier Score (IBS) over time.

ibs_km = gbmi.score(X_test, y_test)
print("Score on Test set (IBS using KM incidence probabilities)", ibs_km.round(3))

shape_censoring = bunch.shape_censoring[X_test.index]
scale_censoring = bunch.scale_censoring[X_test.index]

ibs_oracle = gbmi.score(
    X_test,
    y_test,
    shape_censoring=shape_censoring,
    scale_censoring=scale_censoring,
)
print(
    "Score on Test set (IBS using oracle incidence probabilities)", ibs_oracle.round(3)
)
# %%
