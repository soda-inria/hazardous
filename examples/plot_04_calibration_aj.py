"""
===========================
Calibration of survival models with competing risks
===========================

In this example, we illustrate how the Aalen-Johansen calibration metric works.
The Aalen-Johansen (AJ) calibration metric measures marginal calibration of a
survival model in the presence of competing risks. It compares the mean predicted
cumulative incidence function (CIF) against the non-parametric AJ estimator fitted
on the evaluation cohort.

We demonstrate:

1. The AJ estimator itself as a calibrated baseline.
2. Why the AJ estimator does not achieve exactly zero calibration error when
   evaluated on a different data split.
3. SurvivalBoost as a model with individual-level predictions.
4. The three granularities of the metric:
   :func:`~hazardous.metrics.aj_calibration`,
   :func:`~hazardous.metrics.aj_calibration_per_event`, and
   :func:`~hazardous.metrics.aj_calibration_at_t`.
"""

# %%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from hazardous.utils import make_time_grid
from hazardous._km_sampler import _AalenJohansenSampler
from hazardous._survival_boost import SurvivalBoost
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.metrics import (
    aj_calibration,
    aj_calibration_at_t,
    aj_calibration_per_event,
)

# %%
# Generation of one synthetic dataset with 3 competing events,
# and display of the distribution of the target.

n_samples = 10_000
n_events = 3

X, y = make_synthetic_competing_weibull(
    n_samples=n_samples,
    return_X_y=True,
    n_events=n_events,
    censoring_relative_scale=1.5,
    random_state=0,
)

sns.histplot(data=y, x="duration", hue="event", bins=50, kde=True)
plt.title("Distribution of the target")
plt.show()

# %%
# Data splits: train / test.
# The model is fitted on the train set, and the calibration is measured on the test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# Common time grid used throughout this example.
times = make_time_grid(
    event=y_train["event"], duration=y_train["duration"], n_time_grid_steps=100
)

# %%
# AJ estimator as a calibrated baseline.
# The AJ estimator is a marginal model: it predicts the same CIF for every
# individual. Fitting on the training set and evaluating on the test set gives a
# calibration score close (but not exactly equal) to zero.

aalen_sampler = _AalenJohansenSampler().fit(y_train)

incidence_funcs_aj = dict(aalen_sampler.incidence_func_)
incidence_funcs_aj[0] = aalen_sampler.survival_func_


def _aj_predictions(sampler_funcs, n_samples, n_events, times):
    """Broadcast marginal AJ predictions to shape (n_samples, n_events+1, n_times)."""
    preds = np.array(
        [np.tile(sampler_funcs[i](times), (n_samples, 1)) for i in range(n_events + 1)]
    )
    return preds.transpose(1, 0, 2)  # (n_samples, n_events+1, n_times)


inc_probs_aj_test = _aj_predictions(incidence_funcs_aj, len(y_test), n_events, times)

aj_cal_test = aj_calibration(y_test, times, inc_probs_aj_test)

print(f"AJ calibration score (test set): {aj_cal_test:.6f}")

# %%
# Why does the AJ estimator have a non-zero calibration score?
#
# The AJ calibration metric compares the *mean predicted CIF* against the
# AJ estimator on the test set.
# When the model is itself the AJ fitted on the training set, its predictions
# are the AJ CIFs from y_train. The reference inside the metric is the AJ
# fitted on y_test. These two AJ estimates are obtained on different
# random splits and therefore differ slightly due to sampling variability.
# This produces a small but non-zero calibration error, even though the model is
# theoretically marginally calibrated.
#
# The plot below shows the AJ CIF on the three splits for each event:
# they are close but not identical, which explains the residual score.

aj_test_sampler = _AalenJohansenSampler().fit(y_test)

fig, axes = plt.subplots(ncols=n_events, figsize=(15, 4), sharey=False)
fig.suptitle(
    "AJ CIFs on different splits\n(small differences explain why AJ calibration ≠ 0)"
)

for event_id in range(1, n_events + 1):
    ax = axes[event_id - 1]
    ax.plot(
        times,
        incidence_funcs_aj[event_id](times),
        label="AJ fitted on train",
        linestyle="--",
        color="C0",
    )
    ax.plot(
        times,
        aj_test_sampler.incidence_func_[event_id](times),
        label="AJ fitted on test",
        linestyle=":",
        color="C2",
    )
    ax.set_title(f"Event {event_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative incidence")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()

# %%
# Fitting SurvivalBoost on the training set.
# SurvivalBoost produces *individual-level* CIF predictions, so its mean
# prediction can deviate from the marginal AJ reference, reflecting whether
# the model's is well-calibrated marginally.

survivalboost = SurvivalBoost(n_iter=50, show_progressbar=False, random_state=0)
survivalboost.fit(X_train, y_train)

# Predictions on the calibration cohort — shape (n_test, n_events+1, n_times)
inc_probs_sb = survivalboost.predict_cumulative_incidence(X_test, times=times)

# %%
# Comparing mean SurvivalBoost predictions against the AJ reference.
# A well-calibrated model has its mean CIF close to the non-parametric AJ.

aj_ref = {
    0: aj_test_sampler.survival_func_(times),
    **{k: aj_test_sampler.incidence_func_[k](times) for k in range(1, n_events + 1)},
}

fig, axes = plt.subplots(ncols=n_events + 1, figsize=(18, 4))
fig.suptitle(
    "Mean SurvivalBoost CIFs vs AJ reference on the calibration set\n"
    "(good calibration: curves should overlap)"
)

for event_id in range(n_events + 1):
    ax = axes[event_id]
    mean_sb = inc_probs_sb[:, event_id, :].mean(axis=0)
    ax.plot(times, mean_sb, label="SurvivalBoost (mean)", color="C0")
    ax.plot(times, aj_ref[event_id], label="AJ reference", linestyle="--", color="C1")
    ax.set_title(f"{'Survival (event 0)' if event_id == 0 else f'Event {event_id}'}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %%
# AJ calibration at three levels of granularity.
#
# 1. ``aj_calibration``: single scalar score aggregated across all events.
# 2. ``aj_calibration_per_event``: one scalar score per event, before aggregation.
# 3. ``aj_calibration_at_t``: pointwise difference δ_k(t) at each time, and
#    an example of reading it at a single chosen time point.

# 1 — Overall score (mean across events by default)
score_overall = aj_calibration(y_test, times, inc_probs_sb)
print(f"\n[aj_calibration] overall score (mean): {score_overall:.6f}")

score_sum = aj_calibration(y_test, times, inc_probs_sb, reduction="sum")
print(f"[aj_calibration] overall score (sum):  {score_sum:.6f}")

# 2 — Per-event scores
scores_per_event = aj_calibration_per_event(y_test, times, inc_probs_sb)
print("\n[aj_calibration_per_event] scores:")
for event_id, score in scores_per_event.items():
    print(f"  Event {event_id}: {score:.6f}")

# Score for event 1 only
score_event1 = aj_calibration_per_event(
    y_test, times, inc_probs_sb, event_of_interest=1
)
print(f"\n[aj_calibration_per_event] event_of_interest=1: {score_event1:.6f}")

# 3 — Pointwise differences at a single time point
t_star = np.median(times)  # example time point
t_star_idx = np.argmin(
    np.abs(times - t_star)
)  # index of closest time point in the array

diffs_at_t_star = aj_calibration_at_t(
    y_test,
    times[[t_star_idx]],
    inc_probs_sb[:, :, t_star_idx : t_star_idx + 1],
)
print(f"\n[aj_calibration_at_t] pointwise δ_k at t={t_star:.2f}:")
for event_id, diff in diffs_at_t_star.items():
    print(f"  Event {event_id}: {diff[0]:.4f}")

# Alternatively, retrieve differences over all times and index manually
diffs_all = aj_calibration_at_t(y_test, times, inc_probs_sb)
print("[aj_calibration_at_t] same values via full-time array indexing:")
for event_id, diff in diffs_all.items():
    print(f"  Event {event_id} at t={t_star:.2f}: {diff[t_star_idx]:.4f}")

# %%
# Visualizing the pointwise calibration error for SurvivalBoost.
# Values close to zero indicate good marginal calibration at that time point.

fig, axes = plt.subplots(ncols=n_events + 1, figsize=(18, 4))
fig.suptitle("Pointwise AJ calibration error AJ_k(t) — SurvivalBoost")

for event_id, diff in diffs_all.items():
    ax = axes[event_id]
    ax.plot(times, diff, color="C0")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(
        t_star, color="C2", linewidth=0.8, linestyle=":", label=f"t*={t_star:.1f}"
    )
    ax.scatter([t_star], [diff[t_star_idx]], color="C2", zorder=5)
    ax.set_title(f"{'Survival (event 0)' if event_id == 0 else f'Event {event_id}'}")
    ax.set_xlabel("Time")
    ax.set_ylabel("AJ_k(t)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %%
