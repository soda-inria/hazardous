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

fg = FineGrayEstimator().fit(X, y)
fg.coefs_

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
    ax.plot(fg.times_, y_pred[1, idx, :], label=f"sample {idx}")
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
    ax.plot(fg.times_, y_pred[1, idx, :], label=f"sample {idx}")
ax.grid()
ax.legend()

# %%
# Let's compare Fine and Gray marginal incidence to AalenJohansen
# and assess of potential biases.
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from hazardous._aalan_johansen import AalenJohansenEstimator


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

n_events = y["event"].nunique() - 1
fig, axes = plt.subplots(ncols=n_events, sharey=True, figsize=(12, 5))

fg = FineGrayEstimator().fit(X_train, y_train)
y_pred_fg = fg.predict_cumulative_incidence(X_test)
times = fg.times_

aj = AalenJohansenEstimator().fit(y_train)
y_pred_aj = aj.predict_cumulative_incidence(X_test, times)

aj_uncensored = AalenJohansenEstimator().fit(y_uncensored)
y_pred_aj_uncensored = aj_uncensored.predict_cumulative_incidence(X_test, times)

for event_idx, ax in tqdm(enumerate(axes)):
    for sample_idx in range(5):
        ax.plot(
            times,
            y_pred_fg[event_idx + 1, sample_idx, :],
            label=f"F&G sample {sample_idx}",
            linestyle="--",
        )

    ax.plot(
        times,
        y_pred_fg[event_idx + 1].mean(axis=0),
        label="F&G marginal",
        linewidth=3,
    )

    ax.plot(
        times,
        y_pred_aj[event_idx + 1][0],
        label="AJ",
        color="k",
    )
    ax.plot(
        times,
        y_pred_aj_uncensored[event_idx + 1][0],
        label="AJ uncensored",
        color="k",
        linestyle="--",
    )

    ax.set_title(f"Event {event_idx}")
    ax.grid()
    ax.legend()

fig.suptitle("Marginal incidence")

# %%
from hazardous.metrics import brier_score_incidence


times = fg.times_

fig, axes = plt.subplots(ncols=n_events, sharey=True, figsize=(12, 5))

for idx, ax in tqdm(enumerate(axes)):
    fg_brier_score = brier_score_incidence(
        y_train,
        y_test,
        y_pred_fg[idx + 1],
        times,
        event_of_interest=idx + 1,
    )

    ax.plot(times, fg_brier_score, label="FG brier score")

    aj_brier_score = brier_score_incidence(
        y_train,
        y_test,
        y_pred_aj[idx + 1],
        times,
        event_of_interest=idx + 1,
    )

    ax.plot(times, aj_brier_score, label="AJ brier score")

    ax.set_title(f"Event {idx+1}")
    ax.grid()
    ax.legend()

# %%
