"""
=====================================
Survival analysis with SurvivalBoost
=====================================

Introduction
------------

Survival analysis is a time-to-event regression problem, with censored data. We
call censored all individuals that didn't experience the event during the range
of the observation window.

In our setting, we're mostly interested in right-censored data, meaning we that
the event of interest did not occur before the end of the observation period
(typically the time of collection of the dataset).


Importing the dataset
---------------------


We will use the The Molecular Taxonomy of Breast Cancer International Consortium
(METABRIC) dataset as an example, available through pycox.datasets. This is the
processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
can be found at https://doi.org/10.1186/s12874-018-0482-1.

"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from pycox.datasets import metabric
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw

from hazardous.metrics import integrated_brier_score_survival
from hazardous import SurvivalBoost

np.random.seed(0)

df = metabric.read_df()
df_train, df_test = train_test_split(df, test_size=0.2)
df_train, df_val = train_test_split(df_train, test_size=0.2)
# reset index
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)

# %% For each individual `i` in the METABRIC dataset, the target `y_i` is
# comprised of two elements:
#
# - The column `event` where $0$ marks censoring and $1$ is indicative that the
#   event of interest (death) has actually happened before the end of the
#   observation window.
# - The censored time-to-event column `duration` $d_i=min(t_{i}, c_i) > 0$, that
#   is the minimum between the date of the experienced event $t_i$ and the
#   censoring date $c_i$.
#
# In this dataset, we have 40% of censoring.

# %%
df["event"].value_counts(normalize=True)

# %%
# Using SurvivalBoost to estimate the survival function
# -----------------------------------------------------

# Here our quantity of interest is the survival probability:
#
# $$S(t | X=x)=P(T > t | X=x)$$
#
# This represents the probability that an event doesn't occur at or before some
# given time $t$, i.e. that it happens at some time $T > t$, given the patient
# features x.

# SurvivalBoost is a sklearn-compatible model which expect a dataframe (or
# array-like) input as X, and a dataframe with columns "event" and "duration" as
# y. This allows SurvivalBoost to estimate the survival function $S$.

survival_boost = SurvivalBoost(
    learning_rate=0.05,
    n_iter=100,
    n_iter_before_feedback=50,
    hard_zero_fraction=0.1,
    min_samples_leaf=50,
    show_progressbar=False,
    n_horizons_per_observation=3,
    random_state=0,
)

# %%
survival_boost.fit(
    df_train.drop(columns=["event", "duration"]), df_train[["event", "duration"]]
)

# %% SurvivalBoost can then predict the survival function for each patient and
# each time given by the user (by default, the time grid learnt during fit).

cum_functions = survival_boost.predict_cumulative_incidence(
    df_test.drop(columns=["event", "duration"]), times=None
)

survival_func = cum_functions[:, 0]  # survival function S(t)
cum_inc = cum_functions[:, 1]  # cumulative incidence of the event (here death)

# %%
# Let's plot the estimated survival function for some patients.

fig, ax = plt.subplots()

patient_ids_to_plot = [0, 1, 2, 3]

for j in patient_ids_to_plot:
    ax.plot(survival_boost.time_grid_, survival_func[j], label=f"Patient {j}")

    # plot symbols for death or censoring
    event = df_test.iloc[j]["event"]
    duration = df_test.iloc[j]["duration"]

    # find the index of time closest to duration
    index = np.abs(survival_boost.time_grid_ - duration).argmin()
    smiley = "☠️" if event == 1 else "✖"
    ax.text(
        duration,
        survival_func[j, index],
        smiley,
        fontsize=20,
        color=ax.lines[patient_ids_to_plot.index(j)].get_color(),
    )

ax.legend()
ax.set_title("")
ax.set_xlabel("Months")
ax.set_ylabel("Predicted Cumulative Incidence of Death")

plt.show()

# %%
# Measuring features impact on predictions
# ----------------------------------------

# We can also observe the survival function by age group, or by chemotherapy
# treatment to show the impact the model attribute to these features. We do
# something akin to Partial Dependence Plots, where we sample the feature
# independently from the other features, to eliminate correlations.

# we create a synthetic dataset where age is resampled to reduce cofounder bias.


synthetic_df = df_train.copy()
synthetic_df["x8"] = np.linspace(20, 80, len(synthetic_df))  # Vary age from 20 to 80

fig, ax = plt.subplots()

# Predict cumulative incidence on the synthetic dataset
survival_func_synthetic = survival_boost.predict_survival_function(
    synthetic_df.drop(columns=["event", "duration"])
)

# Create age bins
age_bins = pd.cut(synthetic_df["x8"], bins=[0, 30, 40, 50, 60, 70, 80, 90, 100])
age_groups = sorted(
    age_bins.unique(), key=lambda x: x.left
)  # Sort age groups by the left bin edge

# Create a colormap
cmap = plt.get_cmap("viridis", len(age_groups))

for i, age_group in enumerate(age_groups):
    # Get the indices of patients in the current age group
    indices = synthetic_df[age_bins == age_group].index
    # Calculate the mean and std cumulative incidence for the current age group
    mean_cum_inc = survival_func_synthetic[indices].mean(axis=0)
    std_cum_inc = survival_func_synthetic[indices].std(axis=0)
    # Plot with color from colormap
    ax.plot(
        survival_boost.time_grid_,
        mean_cum_inc,
        label=f"Age {age_group}",
        color=cmap(i),
        linewidth=3,
    )
    # Add ribbon for std
    ax.fill_between(
        survival_boost.time_grid_,
        mean_cum_inc - std_cum_inc,
        mean_cum_inc + std_cum_inc,
        color=cmap(i),
        alpha=0.3,
    )

ax.legend()
ax.set_title("Survival function by Age")
ax.set_xlabel("Months")
ax.set_ylabel("Estimated Survival probability")

plt.show()

# %%
# Unsurprisingly, the cumulative incidence of death mostly increases with age.
# We can do the same thing with Chemotherapy treatement.

# Create a synthetic dataset where chemotherapy (x6) varies

synthetic_df = df_train.copy()
synthetic_df["x6"] = np.tile(
    [0, 1], len(synthetic_df) // 2
)  # Alternate between 0 and 1

fig, ax = plt.subplots()

# Predict survival function on the synthetic dataset
survival_func_synthetic = survival_boost.predict_survival_function(
    synthetic_df.drop(columns=["event", "duration"])
)

# Create a colormap
cmap = plt.get_cmap("viridis", 2)  # Only two groups: 0 and 1

for i, chemo_group in enumerate([0, 1]):
    # Get the indices of patients in the current chemotherapy group
    indices = synthetic_df[synthetic_df["x6"] == chemo_group].index
    # Calculate the mean and std survival function for the current chemotherapy group
    mean_surv_func = survival_func_synthetic[indices].mean(axis=0)
    std_surv_func = survival_func_synthetic[indices].std(axis=0)
    # Plot with color from colormap
    ax.plot(
        survival_boost.time_grid_,
        mean_surv_func,
        label=(
            "Treated with Chemotherapy"
            if chemo_group == 1
            else "Not Treated with Chemotherapy"
        ),
        color=cmap(i),
        linewidth=3,
    )
    # Add ribbon for std
    ax.fill_between(
        survival_boost.time_grid_,
        mean_surv_func - std_surv_func,
        mean_surv_func + std_surv_func,
        color=cmap(i),
        alpha=0.3,
    )

ax.legend()
ax.set_title("Survival Function by Given Chemotherapy Treatment")
ax.set_xlabel("Months")
ax.set_ylabel("Estimated Survival Probability")

plt.show()

# %% People treated with Chemotherapy probably have more serious stages of
# cancer, which is reflected by the lower estimated survival function. This is a
# reminder that the estimate is not causal.

# Let's now attempt to quantify how a survival curve estimated on a training set
# performs on a test set.
#
# Survival model evaluation using the Integrated Brier Score (IBS) and the
# Concordance Index (C-index)
# ------------------------------------------------------------------------------

# The Brier score and the C-index are measures that **assess the quality of a
# predicted survival curve** on a finite data sample.
#
# - **The Brier score in time is a proper scoring rule**, meaning that an
# estimate of the survival probabilities at a given time $t$ has minimal Brier
# score if and only if it matches the oracle survival probabilities induced by
# the underlying data generating process. In that respect the **Brier score**
# assesses both the **calibration** and the **ranking power** of a survival
# probability estimator. It is comprised between 0 and 1 (lower is better). It
# answers the question "how close to the real probabilities are our estimates?".
#
# - On the other hand, the **C-index** only assesses the **ranking power**: it
#   represents the probability that, for a randomly selected pair of patients,
#   the patient with the higher estimated survival probablity will survive
#   longer than the other. It is comprised between 0 and 1 (higher is better),
#   with 0.5 corresponding to a random prediction.
#
# <details><summary>Mathematical formulation (Brier score)</summary>
#
# $$\mathrm{BS}^c(t) = \frac{1}{n} \sum_{i=1}^n I(d_i \leq t \land \delta_i = 1)
#         \frac{(0 - \hat{S}(t | \mathbf{x}_i))^2}{\hat{G}(d_i)} + I(d_i > t)
#         \frac{(1 - \hat{S}(t | \mathbf{x}_i))^2}{\hat{G}(t)}$$
#
# In the survival analysis context, the Brier Score can be seen as the Mean
# Squared Error (MSE) between our probability $\hat{S}(t)$ and our target label
# $\delta_i \in {0, 1}$, weighted by the inverse probability of censoring
# $\frac{1}{\hat{G}(t)}$. In practice we estimate $\hat{G}(t)$ using a variant
# of the Kaplan-Estimator with swapped event indicator.
#
# - When no event or censoring has happened at $t$ yet, i.e. $I(d_i > t)$, we
# penalize a low probability of survival with $(1 - \hat{S}(t|\mathbf{x}_i))^2$.
# - Conversely, when an individual has experienced an event before $t$, i.e.
# $I(d_i \leq t \land \delta_i = 1)$, we penalize a high probability of survival
# with $(0 - \hat{S}(t|\mathbf{x}_i))^2$.
#
# </details>

# <details><summary>Mathematical formulation (C-index)</summary>
#
# $$\mathrm{C_{index}} = \frac{\sum_{i,j} I(d_i < d_j \space \land \space
# \delta_i = 1 \space \land \space \mu_i < \mu_j)}{\sum_{i,j} I(d_i < d_j \space
# \land \space \delta_i = 1)}$$
# </details>

# Additionnaly, we compute the Integrated Brier Score (IBS) which we will use to
# summarize the Brier score in time: $$IBS = \frac{1}{t_{max} -
# t_{min}}\int^{t_{max}}_{t_{min}} BS(t) dt$$

brier_score_hazardous = integrated_brier_score_survival(
    df_train[["event", "duration"]],
    df_test[["event", "duration"]],
    survival_func,
    times=survival_boost.time_grid_,
)
print(f"Brier score for SurvivalBoost: {brier_score_hazardous:.4f}")

# %%
# We can compare this Brier score to the Brier score of a simple Kaplan-Meier
# estimator, which doesn't take the patient features into account.

km_model = KaplanMeierFitter()
km_model.fit(df_train["duration"], df_train["event"])
survival_curve_agg_km = km_model.survival_function_at_times(survival_boost.time_grid_)
# To get individual survival curves,
# we duplicate the survival curve for each patient
survival_curve_km = np.tile(survival_curve_agg_km, (len(df_test), 1))

brier_score_km = integrated_brier_score_survival(
    df_train[["event", "duration"]],
    df_test[["event", "duration"]],
    survival_curve_km,
    times=survival_boost.time_grid_,
)
print(f"Brier score for Kaplan-Meier: {brier_score_km:.4f}")

# %%
# The concordance_index metric from scikit-survival
# expects a structured array of events and durations

events_train, durations_train = df_train["event"].values, df_train["duration"].values
events_test, durations_test = df_test["event"].values, df_test["duration"].values
train_data = np.array(
    [(events_train[i], durations_train[i]) for i in range(len(events_train))],
    dtype=[("e", bool), ("t", float)],
)
test_data = np.array(
    [(events_test[i], durations_test[i]) for i in range(len(events_test))],
    dtype=[("e", bool), ("t", float)],
)

# %%
time_index = 75
concordance_index_km = concordance_index_ipcw(
    train_data,
    test_data,
    1 - survival_curve_km[:, time_index],
    tau=survival_boost.time_grid_[time_index],
)[0]
print(
    "Concordance index for Kaplan-Meier at"
    f" {survival_boost.time_grid_[time_index]:.0f} months: {concordance_index_km:.2f}"
)

# %% 0.5 corresponds to random chance, which makes sense as the Kaplan-Meier
# estimator doesn't depend on the patient features.

time_index = 75
concordance_index_hazardous = concordance_index_ipcw(
    train_data,
    test_data,
    1 - survival_func[:, time_index],
    tau=survival_boost.time_grid_[time_index],
)[0]
print(
    "Concordance index for SurvivalBoost at"
    f" {survival_boost.time_grid_[time_index]:.0f} months:"
    f" {concordance_index_hazardous:.2f}"
)

# %%
