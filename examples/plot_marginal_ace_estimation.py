"""
The Alternating Censoring Estimator (ACE) should be equivalent to
using the Gradient Boosting Incidence (GBI) to predict survival to censoring,
both with a marginal IPCW estimator.
"""
# %%

import numpy as np
from matplotlib import pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d


# This estimator emulates the any event survival estimator provided
# by the GBI.
class KaplanMeierEstimator:
    def fit(self, y):
        self.km_ = KaplanMeierFitter()
        self.km_.fit(
            durations=y["duration"],
            event_observed=y["event"] > 0,
        )

        df = self.km_.cumulative_density_
        self.cumulative_density_ = df.values[:, 0]
        self.unique_times_ = df.index

        self.cumulative_density_func_ = interp1d(
            self.unique_times_,
            self.cumulative_density_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return self

    def predict_proba(self, times):
        if times.ndim == 2:
            times = times[:, 0]
        any_event = self.cumulative_density_func_(times).reshape(-1, 1)
        # The class 0 is the survival to any event S(t).
        # The class 1 is the any event incidence.
        return np.hstack([1 - any_event, any_event])


# %%
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

seed = 0
X, y = make_synthetic_competing_weibull(
    n_events=3,
    n_samples=30_000,
    n_features=5,
    independent_censoring=False,
    complex_features=False,
    return_X_y=True,
    target_rounding=None,
    random_state=seed,
)
X.shape

# %%
from hazardous._gb_multi_incidence import WeightedMultiClassTargetSampler
from hazardous._ipcw import AlternatingCensoringEst


incidence_est = KaplanMeierEstimator()
ace_est = AlternatingCensoringEst(
    incidence_est=incidence_est,
    max_leaf_nodes=5,
    min_samples_leaf=50,
    monotonic_cst=None,  # [1] + [0] * 6,
)
ws = WeightedMultiClassTargetSampler(
    y,
    ipcw_est=ace_est,
    random_state=seed,
    n_iter_before_feedback=100,
)

# Mock the training of the tree-based incidence estimator.
incidence_est.fit(y)

ws.fit(X)

# %%
t_min, t_max = y["duration"].min(), y["duration"].max()
time_grid = np.linspace(t_min, t_max, 100)
y_pred_ace = []

est = ws.ipcw_est.censoring_est_
for t in time_grid:
    t = np.full((X.shape[0], 1), fill_value=t)
    X_with_time = np.hstack([t, X])
    y_pred_at_t = est.predict_proba(X_with_time)[:, 0]
    y_pred_ace.append(y_pred_at_t)
y_pred_ace = np.array(y_pred_ace).T
y_pred_ace.shape

# %%
from hazardous._gradient_boosting_incidence import GradientBoostingIncidence

y_censored = y.copy()
y_censored["event"] = y_censored["event"] == 0

gbi = GradientBoostingIncidence(n_iter=100)
gbi.fit(X, y_censored)
y_pred_gbi = gbi.predict_survival_function(X, times=time_grid)
y_pred_gbi.shape

# %%
from hazardous._ipcw import IPCWEstimator

ipcw_est = IPCWEstimator().fit(y)
y_pred_marginal = ipcw_est.compute_censoring_survival_proba(time_grid)

# %%
fig, ax = plt.subplots()
ax.plot(time_grid, y_pred_gbi.mean(axis=0), label="GBI")
ax.plot(time_grid, y_pred_ace.mean(axis=0), label="ACE")
ax.plot(time_grid, y_pred_marginal, label="KM")
ax.legend()
plt.grid()
fig.suptitle("Marginal survival to censoring probabilities G(t)")
# %%

# For some reasons, the ACE gives smoother results than the GBI estimator,
# although both underlying boosting trees share the same hyper-parameters.
n_samples = 3
fig, axes = plt.subplots(figsize=(12, 5), ncols=n_samples)

for sample_id in range(n_samples):
    axes[sample_id].plot(time_grid, y_pred_gbi[sample_id], label=f"GBI {sample_id}")
    axes[sample_id].plot(time_grid, y_pred_ace[sample_id], label=f"ACE {sample_id}")
    axes[sample_id].legend()
fig.suptitle("Individual survival to censoring probabilities G(t|x)")
plt.grid()
# %%
