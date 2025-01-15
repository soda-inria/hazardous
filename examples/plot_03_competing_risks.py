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
# Here is a little bit of context about this metric introduced in [Alberge2024]_:
#
# - It is a generalization of the accuracy metric in the survival and the competing
#   risks setting, which is the proportion of correct predictions (e.g. highest
#   predicted probability) at a fixed time.
# - It is computed for different time horizons, given by the user, with direct time
#   stamps or quantiles of the observed durations.
# - For a given patient at a fixed time, we compare the actual observed event to the
#   most predicted one. The censored patients are removed from the computation after
#   their censoring duration. In other word, let's imagine a patient who has
#   experienced death by cancer at time :math:`t`. Before this time, we want the model
#   to predict with the highest probability that this patient will survive. After
#   :math:`t`, we want the model to predict the death by cancer event with the highest
#   probability.
# - The mathematical formula is:
#
# .. math::
#     \mathrm{acc}(\zeta) = \frac{1}{n_{nc}} \sum_{i=1}^n ~ I\{\hat{y}_i=y_{i,\zeta}\}
#        \overline{I\{\delta_i = 0 \cap t_i \leq \zeta \}}
#
# where:
#
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
#
#      **References:**
#
#      .. [Alberge2024] J. Alberge, V. Maladiere,  O. Grisel, J. Abécassis,
#      G. Varoquaux, "Survival Models: Proper Scoring Rule and Stochastic Optimization
#      with Competing Risks", 2024

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
# After training SurvivalBoost, we compute its accuracy in time.
#

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
# estimator. Higher is better. Note that the accuracy is high at very beginning
# (:math:`t < 1000`), because both models predict that every individual survive, which
# is true in most cases. Then, beyond the time horizon 1000, the discriminative power
# of the conditional SurvivalBoost yields a better accuracy than the marginal, unbiased,
# Aalen-Johansen's estimator.
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
plt.show()


# %%
# Understanding the accuracy in time
# ----------------------------------
#
# We can drill into this metric by counting the observed events cumulatively across
# time, and compare that to predictions.
#
# We display below the distribution of ground truth labels. Each color bar group
# represents the event distribution at some given horizon.
# Almost no individual have experienced an event at the very beginning.
# Then, as time passes by, events occur and the number of censored individual at each
# time horizon shrinks. Therefore, the very last distribution represents the overall
# event distribution of the dataset.
def plot_event_in_time(y_in_time):
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


time_grid_2d = np.tile(time_grid, (y_test.shape[0], 1))
y_test_class = (y_test["duration"].values[:, None] <= time_grid_2d) * y_test[
    "event"
].values[:, None]
plot_event_in_time(y_test_class)
# %%
# Now, we compare this ground truth to the classes predicted by SurvivalBoost.
# Interestingly, it seems too confident about the censoring event at the
# beginning (:math:`t < 500`), but then becomes underconfident in the middle
# (:math:`t > 1500`) and very overconfident about the class 3 in the end
# (:math:`t > 3000`).

y_pred_class = y_pred.argmax(axis=1)
plot_event_in_time(y_pred_class)

# %%
# Finally, we compare this to the classes predicted by the Aalen-Johansen model.
# They are constant in individuals because this model is marginal and we simply
# duplicated the global cumulative incidences for each individual.
y_pred_class_aj = y_pred_aj.argmax(axis=1)
plot_event_in_time(y_pred_class_aj)
# %%
