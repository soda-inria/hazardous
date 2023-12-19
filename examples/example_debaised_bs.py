# Here, we create a synthetic dataset, either with a censoring independant
# of the covariates of dependant with a rate given by the user.
# We train a GBI to obtain some prediction. Givent a time grid, we compute
# the true Brier score (with the real distribution of the censoring) and the BS
# with an estimate of the proba of censoring (Kapplan Meier).
#

# %%
import seaborn as sns

from hazardous.data._competing_weibull import make_synthetic_competing_weibull


independent_censoring = True
event_of_interest = 1
seed = 0

X, _, y_uncensored = make_synthetic_competing_weibull(
    n_samples=1_000_000,
    base_scale=1_000,
    n_features=10,
    features_rate=0.3,
    degree_interaction=2,
    independent_censoring=independent_censoring,
    features_censoring_rate=0.2,
    return_uncensored_data=True,
    return_X_y=True,
    feature_rounding=None,
    target_rounding=None,
    censoring_relative_scale=None,
    random_state=seed,
    complex_features=True,
)

sns.histplot(
    y_uncensored,
    x="duration",
    hue="event",
    palette="magma",
    title="Duration distributions",
)

# %%
from hazardous.data._competing_weibull import _censor
from hazardous.evaluation.debiased_brier_score import compute_true_probas


bunch_data = _censor(
    y_uncensored,
    independent_censoring=independent_censoring,
    X=X,
    return_censoring_params=True,
    features_censoring_rate=0.5,
    censoring_relative_scale=0.5,
    random_state=seed,
)
y_censored = bunch_data.y_censored
shape_censoring = bunch_data.shape_censoring
scale_censoring = bunch_data.scale_censoring

probas = compute_true_probas(
    y_censored,
    shape_censoring,
    scale_censoring,
    independent_censoring=independent_censoring,
)
time_grid = probas.time_grid
censored_proba_duration = probas.censored_proba_duration
censored_proba_time_grid = probas.censored_proba_time_grid

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
gbi

# %%
from matplotlib import pyplot as plt

from hazardous.metrics._brier_score import brier_score_incidence
from hazardous.evaluation.debiased_brier_score import brier_score_true_probas


y_pred = gbi.predict_cumulative_incidence(X, times=time_grid)

scores = brier_score_true_probas(
    y_censored,
    y_pred,
    times=time_grid,
    event_of_interest=event_of_interest,
    censored_proba_duration=censored_proba_duration,
    censored_proba_time_grid=censored_proba_time_grid,
)

bs_scores = brier_score_incidence(
    y_censored, y_censored, y_pred, time_grid, event_of_interest=event_of_interest
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_grid, scores, label="from true distrib")
ax.plot(time_grid, bs_scores, label="from estimate distrib with km")

ax.set(
    title="Time-varying Brier score",
)
ax.legend()
# %%
