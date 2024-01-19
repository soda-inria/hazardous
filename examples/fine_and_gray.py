# %%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from hazardous.data._competing_weibull import make_synthetic_competing_weibull

seed = 0
independent_censoring = False
complex_features = True

bunch = make_synthetic_competing_weibull(
    n_samples=1000,
    n_events=3,
    n_features=5,
    return_X_y=False,
    independent_censoring=independent_censoring,
    censoring_relative_scale=1.5,
    random_state=seed,
    complex_features=complex_features,
)
X, y, y_uncensored = bunch.X, bunch.y, bunch.y_uncensored

censoring_rate = (y["event"] == 0).mean()
censoring_kind = "independent" if independent_censoring else "dependent"
ax = sns.histplot(
    y,
    x="duration",
    hue="event",
    multiple="stack",
    palette="magma",
)
ax.set_title(f"{censoring_kind} censoring rate {censoring_rate:.2%}")

# %%
from hazardous._fine_and_gray import FineGrayEstimator

fg = FineGrayEstimator(event_of_interest=1).fit(X, y)
fg.coef_

# %%
# Increasing the value of a feature matching a positive coefficient
# increases the probability of incidence of our event of interest.
X_test = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
y_pred = fg.predict_cumulative_incidence(X_test)

fig, ax = plt.subplots()
for idx in range(X_test.shape[0]):
    ax.plot(fg.times_, y_pred[idx, :], label=f"sample {idx}")
ax.grid()
ax.legend()

# %%
# Reversely, doing so for a negative coefficient decreases
# the probability of incidence.
X_test = np.array(
    [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 3.0, 0.0],
    ]
)
y_pred = fg.predict_cumulative_incidence(X_test)

fig, ax = plt.subplots()
for idx in range(X_test.shape[0]):
    ax.plot(fg.times_, y_pred[idx, :], label=f"sample {idx}")
ax.grid()
ax.legend()

# %%
# Let's compare Fine and Gray marginal incidence to AalenJohansen
# and assess of potential biases.
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from lifelines import AalenJohansenFitter


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

n_events = y["event"].nunique() - 1
fig, axes = plt.subplots(ncols=n_events, sharey=True, figsize=(12, 5))

for ax, event_id in tqdm(zip(axes, range(1, n_events + 1))):
    fg = FineGrayEstimator(event_of_interest=event_id).fit(X_train, y_train)

    y_pred = fg.predict_cumulative_incidence(X_test)
    times = fg.times_

    for idx in range(5):
        ax.plot(
            times,
            y_pred[idx, :],
            label=f"F&G sample {idx}",
            linestyle="--",
        )

    ax.plot(
        times,
        y_pred.mean(axis=0),
        label="F&G marginal",
        linewidth=3,
    )

    with warnings.catch_warnings(record=True):
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        aj = AalenJohansenFitter(calculate_variance=False, seed=seed).fit(
            durations=y["duration"],
            event_observed=y["event"],
            event_of_interest=event_id,
        )

        aj_uncensored = AalenJohansenFitter(calculate_variance=False, seed=seed).fit(
            durations=y_uncensored["duration"],
            event_observed=y_uncensored["event"],
            event_of_interest=event_id,
        )

    aj.plot(ax=ax, label="AJ", color="k")
    aj_uncensored.plot(ax=ax, label="AJ uncensored", color="k", linestyle="--")

    ax.set_title(f"Event {event_id}")
    ax.grid()
    ax.legend()

# %%
from scipy.interpolate import interp1d
from hazardous.metrics import brier_score_incidence


fig, axes = plt.subplots(ncols=n_events, sharey=True, figsize=(12, 5))

for ax, event_id in tqdm(zip(axes, range(1, n_events + 1))):
    fg = FineGrayEstimator(event_of_interest=event_id).fit(X_train, y_train)

    y_pred = fg.predict_cumulative_incidence(X_test)
    times = fg.times_

    fg_brier_score = brier_score_incidence(
        y_train,
        y_test,
        y_pred,
        times,
        event_of_interest=event_id,
    )

    ax.plot(times, fg_brier_score, label="FG brier score")

    with warnings.catch_warnings(record=True):
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        aj = AalenJohansenFitter(calculate_variance=False, seed=seed).fit(
            durations=y["duration"],
            event_observed=y["event"],
            event_of_interest=event_id,
        )

    times_aj = aj.cumulative_density_.index
    y_pred_aj = aj.cumulative_density_.to_numpy()[:, 0]
    y_pred_aj = interp1d(
        x=times_aj,
        y=y_pred_aj,
        kind="linear",
    )(times)

    y_pred_aj = np.vstack([y_pred_aj for _ in range(X_test.shape[0])])

    aj_brier_score = brier_score_incidence(
        y_train,
        y_test,
        y_pred_aj,
        times,
        event_of_interest=event_id,
    )

    ax.plot(times, aj_brier_score, label="AJ brier score")

    ax.set_title(f"Event {event_id}")
    ax.grid()
    ax.legend()
# %%
