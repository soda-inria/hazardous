"""
==============================
Exploring the accuracy in time
==============================

In this notebook, we showcase how the accuracy in time metric is defined, behaves, and
how to interpret it.
"""

# %%
# Definition of the Accuracy in Time
# ==================================
# Here is a little bit of context about this metric introduced in
# `Alberge et al. (2024) <https://hal.science/hal-04617672v4>`_:
#
# - The accuracy in time is a generalization of the accuracy metric in the survival
#   and the competing risks setting, representing the proportion of correctly
#   predicted labels at a fixed time.
# - This metric is computed for different user-provided time horizons, specified
#   either as direct timestamps or quantiles of the observed durations.
# - For a given patient at a fixed time, we compare the actual observed event to
#   the most likely predicted one. For example, imagine a patient who experiences
#   death due to cancer at time :math:`t`. Before this time, the model should predict
#   with the highest probability that the patient will survive. After :math:`t`,
#   the model should predict the cancer-related death event with the highest
#   probability. Censored patients are excluded from the computation after their
#   censoring time.
# - The mathematical formula is:
#
# .. math::
#     \mathrm{acc}(\zeta) = \frac{1}{n_{nc}} \sum_{i=1}^n ~ I\{\hat{y}_i=y_{i,\zeta}\}
#        \overline{I\{\delta_i = 0 \cap t_i \leq \zeta \}}
#
# where:
#
# - :math:`I` is the indicator function.
# - :math:`\zeta` is a fixed time horizon.
# - :math:`n_{nc}` is the number of uncensored individuals at :math:`\zeta`.
# - :math:`\delta_i` is the event experienced by the individual :math:`i` at
#   :math:`t_i`.
# - :math:`\hat{y} = \text{arg}\max\limits_{k \in [0, K]} \hat{F}_k(\zeta|X=x_i)`
#   where :math:`\hat{F}_0(\zeta|X=x_i) \triangleq \hat{S}(\zeta|X=x_i)`.
#   :math:`\hat{y}` is the most probable predicted event for individual :math:`i`
#   at :math:`\zeta`.
# - :math:`y_{i,\zeta} = \delta_i ~ I\{t_i \leq \zeta \}` is the observed event
#   for individual :math:`i` at :math:`\zeta`.

# %%
# Usage
# =====
#
# Generating synthetic data
# -------------------------
#
# We begin by generating a linear, synthetic dataset. For each individual, we uniformly
# sample a shape and scale value, which we use to parameterize a Weibull distribution,
# from which we sample a duration.
from hazardous.data import make_synthetic_competing_weibull
from sklearn.model_selection import train_test_split


X, y = make_synthetic_competing_weibull(n_events=3, n_samples=10_000, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train.shape, y_train.shape

# %%
# Next, we display the distribution of our target.
import seaborn as sns
from matplotlib import pyplot as plt


sns.histplot(
    y_test,
    x="duration",
    hue="event",
    multiple="stack",
    palette="colorblind",
)
plt.show()

# %%
# Computing the Accuracy in Time
# -------------------------------------------
#
# After training ``SurvivalBoost``, we compute its accuracy in time for 16 quantiles
# of the time grid, i.e. at 16 evenly-spaced times of observation –:math:`\zeta` in our
# formula above.

import numpy as np
from hazardous import SurvivalBoost
from hazardous.metrics import accuracy_in_time


results = []

time_grid = np.arange(0, 4000, 100)
surv = SurvivalBoost(show_progressbar=False).fit(X_train, y_train)
y_pred = surv.predict_cumulative_incidence(X_test, times=time_grid)

quantiles = np.linspace(0.125, 1, 16)
accuracy, taus = accuracy_in_time(y_test, y_pred, time_grid, quantiles=quantiles)
results.append(dict(model_name="Survival Boost", accuracy=accuracy, taus=taus))

# %%
# We also compute the accuracy in time of the Aalen-Johansen estimator, which is
# a marginal model (it doesn't use covariates X), similar to the Kaplan-Meier estimator,
# except that it computes cumulative incidence functions of competing risks instead
# of a survival function.
from scipy.interpolate import interp1d
from lifelines import AalenJohansenFitter
from hazardous.utils import check_y_survival


def predict_aalen_johansen(y_train, time_grid, n_sample_test):
    event, duration = check_y_survival(y_train)
    event_ids = sorted(set(event) - set([0]))

    y_pred = []
    for event_id in event_ids:
        aj = AalenJohansenFitter(calculate_variance=False).fit(
            durations=duration,
            event_observed=event,
            event_of_interest=event_id,
        )
        cif = aj.cumulative_density_
        y_pred_ = interp1d(
            x=cif.index,
            y=cif[cif.columns[0]],
            kind="linear",
            fill_value="extrapolate",
        )(time_grid)

        y_pred.append(
            # shape: (n_sample_test, 1, n_time_steps)
            np.tile(y_pred_, (n_sample_test, 1, 1))
        )

    y_survival = (1 - np.sum(np.concatenate(y_pred, axis=1), axis=1))[:, None, :]
    y_pred.insert(0, y_survival)

    return np.concatenate(y_pred, axis=1)


y_pred_aj = predict_aalen_johansen(y_train, time_grid, n_sample_test=X_test.shape[0])

accuracy, taus = accuracy_in_time(y_test, y_pred_aj, time_grid, quantiles=quantiles)
results.append(dict(model_name="Aalan-Johansen", accuracy=accuracy, taus=taus))

# %%
# Results
# -------
#
# We display the accuracy in time to compare SurvivalBoost with the Aalen-Johansen's
# estimator. Higher is better.
import pandas as pd


fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

results = pd.DataFrame(results).explode(column=["accuracy", "taus"])

sns.lineplot(
    results,
    x="taus",
    y="accuracy",
    hue="model_name",
    ax=ax,
    legend=False,
)

sns.scatterplot(
    results,
    x="taus",
    y="accuracy",
    hue="model_name",
    ax=ax,
    s=50,
    zorder=100,
    style="model_name",
)
plt.tight_layout()
plt.show()


# %%
# Note that the accuracy is high at very beginning
# (:math:`t < 1000`), because both models predict that every individual survive, which
# is true in most cases. Then, beyond the time horizon 1000, the discriminative power
# of the conditional ``SurvivalBoost`` yields a better accuracy than the marginal,
# unbiased, Aalen-Johansen's estimator.
#
# Understanding the accuracy in time
# ----------------------------------
#
# We can drill into this metric by counting the observed events cumulatively across
# time, and compare that to predictions.
#
# We display below the distribution of ground truth labels. Each color bar group
# represents the event distribution at some given time horizons.
# Almost no individual have experienced an event at the very beginning (the very high
# blue bars, corresponding to censoring).
# Then, as time passes by, events occur and the number of censored individual at each
# time horizon shrinks.
def plot_event_in_time(y_in_time, title):
    event_in_times = []
    for event_id in range(4):
        event_in_times.append(
            dict(
                event_count=(y_in_time == event_id).sum(axis=0),
                time_grid=time_grid,
                event=event_id,
            )
        )

    event_in_times = pd.DataFrame(event_in_times).explode(["event_count", "time_grid"])

    ax = sns.barplot(
        event_in_times,
        x="time_grid",
        y="event_count",
        hue="event",
        palette="colorblind",
    )
    ax.set_xticks(ax.get_xticks()[::10])
    ax.set_xlabel("Time")
    ax.set_ylabel("Total events at $t$")
    ax.set_title(title)


time_grid_2d = np.tile(time_grid, (y_test.shape[0], 1))
mask_event_happened = y_test["duration"].values[:, None] <= time_grid_2d
y_test_class = mask_event_happened * y_test["event"].values[:, None]

# In the same fashion as the accuracy-in-time, we don't count individual that were
# censored in the past.
mask_past_censoring = mask_event_happened * (y_test["event"] == 0).values[:, None]
y_test_class[mask_past_censoring] = -1

plot_event_in_time(y_test_class, title="Ground truth")

# %%
# Now, we compare this ground truth to the classes predicted by ``SurvivalBoost``.
# Interestingly, it seems too confident about the censoring event at the
# beginning (:math:`t < 500`), but then becomes underconfident in the middle
# (:math:`t > 1500`) and very overconfident about the class 3 in the end
# (:math:`t > 3000`).
# Overall, we can see that the predicted labels gets closer to the ground truth as the
# time progress, which correspond to the improvement of the accuracy in time
# we saw for the large time horizons.

y_pred_class = y_pred.argmax(axis=1)
y_pred_class[mask_past_censoring] = -1
plot_event_in_time(y_pred_class, title="Survival Boost")

# %%
# Finally, we show the predicted classes from the Aalen-Johansen model.
# These predictions remain constant across all individuals, as the model is marginal,
# and the global cumulative incidences are simply duplicated for each individual.
# Once again, the changes in predicted labels align with the "bumps" observed in
# the accuracy-over-time figure for the Aalen-Johansen model.

y_pred_class_aj = y_pred_aj.argmax(axis=1)
y_pred_class_aj[mask_past_censoring] = -1
plot_event_in_time(y_pred_class_aj, title="Aalen-Johansen")
# %%
