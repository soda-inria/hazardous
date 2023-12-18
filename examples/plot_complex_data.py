"""
==========================================
Modeling competing risks on synthetic data
==========================================

This example introduces a synthetic data generation tool that can
be helpful to study the relative performance and potential biases
of predictive competing risks estimators on right-censored data.
Some of the input features and their second order multiplicative
interactions are statistically associated with the parameters of
the distributions from which the competing events are sampled.

"""

# %%
import numpy as np
import hazardous.data._competing_weibull as competing_w
from time import perf_counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import pandas as pd
import seaborn as sns

from hazardous import GradientBoostingIncidence
from lifelines import AalenJohansenFitter

seed = 0
rng = check_random_state(seed)


# %%
# In the following cell, we verify that the synthetic dataset is well defined.

df_features = pd.DataFrame(rng.randn(100_000, 10))
df_features.columns = [f"feature_{i}" for i in range(10)]

df_shape_scale_star = competing_w.compute_shape_and_scale(
    df_features,
    features_rate=0.2,
    n_events=3,
    degree_interaction=2,
    random_state=0,
)
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for idx, col in enumerate(df_shape_scale_star.columns):
    sns.histplot(df_shape_scale_star[col], ax=axes[idx // 3][idx % 3])

# %%

X, y_censored, y_uncensored = competing_w.make_synthetic_competing_weibull(
    n_samples=30_000,
    base_scale=1_000,
    n_features=10,
    features_rate=0.5,
    degree_interaction=2,
    independent_censoring=True,
    features_censoring_rate=0.2,
    return_uncensored_data=True,
    return_X_y=True,
    feature_rounding=3,
    target_rounding=None,
    censoring_relative_scale=1.5,
    random_state=0,
    complex_features=True,
)

# %%
n_samples = len(X)
calculate_variance = n_samples <= 5_000
aj = AalenJohansenFitter(calculate_variance=calculate_variance, seed=0)

y_censored["event"].value_counts()

gb_incidence = GradientBoostingIncidence(
    learning_rate=0.1,
    n_iter=20,
    max_leaf_nodes=15,
    hard_zero_fraction=0.1,
    min_samples_leaf=5,
    loss="ibs",
    show_progressbar=False,
    random_state=seed,
)


# %%
#
# CIFs estimated on uncensored data
# ---------------------------------
#
# Let's now estimate the CIFs on uncensored data and plot them against the
# theoretical CIFs:


def plot_cumulative_incidence_functions(
    X, y, gb_incidence=None, aj=None, X_test=None, y_test=None
):
    n_events = y["event"].max()
    t_max = y["duration"].max()
    _, axes = plt.subplots(figsize=(12, 4), ncols=n_events, sharey=True)
    # Compute the estimate of the CIFs on a coarse grid.
    coarse_timegrid = np.linspace(0, t_max, num=100)
    censoring_fraction = (y["event"] == 0).mean()
    plt.suptitle(
        "Cause-specific cumulative incidence functions"
        f" ({censoring_fraction:.1%} censoring)"
    )

    for event_id, ax in enumerate(axes, 1):
        if gb_incidence is not None:
            tic = perf_counter()
            gb_incidence.set_params(event_of_interest=event_id)
            gb_incidence.fit(X, y)
            duration = perf_counter() - tic
            print(f"GB Incidence for event {event_id} fit in {duration:.3f} s")
            tic = perf_counter()
            cifs_pred = gb_incidence.predict_cumulative_incidence(X, coarse_timegrid)
            cif_mean = cifs_pred.mean(axis=0)
            duration = perf_counter() - tic
            print(f"GB Incidence for event {event_id} prediction in {duration:.3f} s")
            print("Brier score on training data:", -gb_incidence.score(X, y))
            if X_test is not None:
                print(
                    "Brier score on testing data:", -gb_incidence.score(X_test, y_test)
                )
            ax.plot(
                coarse_timegrid,
                cif_mean,
                label="GradientBoostingIncidence",
            )
            ax.set(title=f"Event {event_id}")

        if aj is not None:
            tic = perf_counter()
            aj.fit(y["duration"], y["event"], event_of_interest=event_id)
            duration = perf_counter() - tic
            print(f"Aalen-Johansen for event {event_id} fit in {duration:.3f} s")
            aj.plot(label="Aalen-Johansen", ax=ax)
            ax.set_xlim(0, 8_000)
            ax.set_ylim(0, 0.5)
        if event_id == 1:
            ax.legend(loc="lower right")
        else:
            ax.legend().remove()


# %%

X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_censored, test_size=0.3, random_state=seed
)
y_train_u = y_uncensored.loc[y_train_c.index]
y_test_u = y_uncensored.loc[y_test_c.index]

gb_incidence = GradientBoostingIncidence(
    learning_rate=0.1,
    n_iter=20,
    max_leaf_nodes=15,
    hard_zero_fraction=0.1,
    min_samples_leaf=5,
    loss="ibs",
    show_progressbar=False,
    random_state=seed,
)

plot_cumulative_incidence_functions(
    X_train,
    y_train_u,
    gb_incidence=None,
    aj=aj,
    X_test=X_test,
    y_test=y_test_u,
)

plot_cumulative_incidence_functions(
    X_train,
    y_train_c,
    gb_incidence=None,
    aj=aj,
    X_test=X_test,
    y_test=y_test_c,
)

# %%
