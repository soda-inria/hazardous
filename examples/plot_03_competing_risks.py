"""
==============================
Exploring the accuracy in time
==============================

In this notebook, we showcase how the accuracy in time metric behaves, and how
to interpret it.
"""
# %%
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

# %%
# We train a Survival Boost model and compute its accuracy in time.
import numpy as np
import pandas as pd
from hazardous import SurvivalBoost
from hazardous.metrics import accuracy_in_time

y_test = pd.DataFrame({"event": [1, 0, 1], "duration": [5, 10, 15]})
y_pred = np.array(
    [
        [[0.2, 0.5], [0.8, 0.5]],  # Sample 1
        [[0.7, 0.6], [0.3, 0.4]],  # Sample 2
        [[0.1, 0.4], [0.9, 0.6]],  # Sample 3
    ]
)
# (1, 2) correct = 2/3, then (3) correct and (2) doesn't count = 1/2
expected_acc_in_time = np.array([2 / 3, 1 / 2])

time_grid = np.array([5, 10])
expected_taus = time_grid

acc_in_time, taus = accuracy_in_time(y_test, y_pred, time_grid)
acc_in_time, taus

# %%

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
# We display the accuracy in time to compare Survival Boost with Aalen-Johansen.
# Higher is better. Note that the accuracy is high at very beginning (t < 1000), because
# both models predict that every individual survive.
# Then, beyond the time horizon 1000, the discriminative power of the conditional
# Survival Boost yields a better accuracy than the marginal, unbiased, Aalen-Johansen.
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


# %%
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
# Now, we compare this ground truth to the classes predicted by our Survival Boost
# model. Interestingly, it seems too confident about the censoring event at the
# beginning (t < 500), but then becomes underconfident in the middle (t > 1500) and
# very overconfident about the class 3 in the end (t > 3000).

y_pred_class = y_pred.argmax(axis=1)
plot_event_in_time(y_pred_class)

# %%
# Finally, we compare this to the classes predicted by the Aalen-Johansen model.
# They are constant in individuals because this model is marginal and we simply
# duplicated the global cumulative incidences for each individual.
y_pred_class_aj = y_pred_aj.argmax(axis=1)
plot_event_in_time(y_pred_class_aj)
# %%
