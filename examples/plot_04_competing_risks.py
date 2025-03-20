"""
==============================
Competing Risks Example
==============================

In this notebook, we showcase how to use the `SurvivalBoost` class to model
competing risks data. We generate a synthetic dataset with three competing
events and fit a model to predict the cumulative incidence functions and the
survival function.
"""

# %%
# Usage
# =====
#
# Generating synthetic data
# -------------------------
#
# We begin by generating a linear, synthetic dataset with three competing events.

import numpy as np
from hazardous.data import make_synthetic_competing_weibull

np.random.seed(0)

n_events = 3
X, y = make_synthetic_competing_weibull(
    n_events=n_events, n_samples=10_000, return_X_y=True
)

# %%
# Next, we display the distribution of our target. Event 0 corresponds to censoring.
import seaborn as sns
from matplotlib import pyplot as plt


sns.histplot(
    y,
    x="duration",
    hue="event",
    multiple="stack",
    palette="colorblind",
)
plt.show()

# %%
from sklearn.model_selection import train_test_split
from hazardous import SurvivalBoost


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
survival_boost = SurvivalBoost(n_iter=10, show_progressbar=False).fit(X_train, y_train)
survival_boost

# %%

incidence_curves = survival_boost.predict_cumulative_incidence(
    X_test,
    times=None,
)

survival_curves = incidence_curves[:, 0]  # survival function S(t)

# %%
# Let's plot the estimated survival function and the incidence functions for
# some patients.We will also plot a symbol at the time of death or censoring.
# The symbol will be a skull and crossbones for the event that occurred,
# a cross for meaning that an other event of interest has happened, and a
# question mark for the censoring event.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

patient_ids_to_plot = [0, 1, 2, 3]

for event_id in range(n_events + 1):
    for idx in patient_ids_to_plot:
        ax[event_id].plot(
            survival_boost.time_grid_,
            incidence_curves[idx, event_id],
            label=f"Patient {idx}",
        )

        # plot symbols for death or censoring
        event = y_test.iloc[idx]["event"]
        duration = y_test.iloc[idx]["duration"]

        # find the index of time closest to duration
        jdx = np.searchsorted(survival_boost.time_grid_, duration)
        smiley = "\N{SKULL AND CROSSBONES}" if event == event_id else "✖"
        smiley = "\N{QUESTION MARK}" if event == 0 else smiley
        ax[event_id].text(
            duration,
            incidence_curves[idx, event_id, jdx],
            smiley,
            fontsize=20,
            color=ax[event_id].lines[idx].get_color(),
        )

    ax[event_id].set_xlabel("Months")
    if event_id == 0:
        ax[event_id].set_title("Survival Function S(t)")
    else:
        ax[event_id].set_title(
            f"Estimated Incidence Probabilities for Event {event_id}"
        )


ax[0].set_ylabel("Predicted Survival Probability")
ax[1].set_ylabel("Predicted Incidence Probabilities")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.show()

# %%
# Let's compute the incidence functions and the survival function for a given patient
# knowing its censoring time.
# Given a patient that has been censored at a given time, we do know that she/he has not
# experienced any event before this time. We can use this information to compute the
# survival function and the incidence functions for the whole period.
five_first_censored_patients = X_test[y_test["event"] == 0].iloc[:5]
censoring_times = y_test[y_test["event"] == 0]["duration"].iloc[:5]

censoring_function_knowing_censoring_time = survival_boost.predict_incidence_after_s(
    five_first_censored_patients, censoring_times.values
)


# %%
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

patient_ids_to_plot = [0, 1, 2, 3, 4]

for event_id in range(n_events + 1):
    for idx in patient_ids_to_plot:
        ax[event_id].plot(
            survival_boost.time_grid_,
            censoring_function_knowing_censoring_time[idx, event_id],
            label=f"Patient {idx}",
        )
        duration = censoring_times.iloc[idx]

        # find the index of time closest to duration
        jdx = np.searchsorted(survival_boost.time_grid_, duration)
        smiley = "\N{QUESTION MARK}"
        ax[event_id].text(
            duration,
            censoring_function_knowing_censoring_time[idx, event_id, jdx],
            smiley,
            fontsize=20,
            color=ax[event_id].lines[idx].get_color(),
        )

    ax[event_id].set_xlabel("Months")
    if event_id == 0:
        ax[event_id].set_title("Survival Function S(t)")
    else:
        ax[event_id].set_title(
            f"Estimated Incidence Probabilities for Event {event_id}"
        )


ax[0].set_ylabel("Predicted Survival Probability")
ax[1].set_ylabel("Predicted Incidence Probabilities")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.show()
# %%
