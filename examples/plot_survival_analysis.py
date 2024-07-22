"""
=====================================
Survival analysis with SurvivalBoost
=====================================

Survival analysis is a time-to-event regression problem, with censored data. We
call censored all individuals that didn't experience the event during the
period of observation.

In our setting, we're mostly interested in right-censored data, meaning we that
the event of interest did not occur before the end of the observation period
(typically the time of collection of the dataset).

We will use the The Molecular Taxonomy of Breast Cancer International Consortium
(METABRIC) dataset as an example, available through ``pycox.datasets``. This is the
processed data set used in the
`DeepSurv paper (Katzman et al. 2018) <https://doi.org/10.1186/s12874-018-0482-1>`_.
"""
# %%
import numpy as np
import pandas as pd

from pycox.datasets import metabric

np.random.seed(0)

df = metabric.read_df()
X = df.drop(columns=["event", "duration"])
y = df[["event", "duration"]]
y

# %%
#
# Notice that the target ``y`` is comprised of two columns:
#
# - ``event`` where :math:`0` marks censoring and :math:`1` is indicative that the
#   event of interest (death) has actually happened before the end of the
#   observation window.
# - ``duration``, the censored time-to-event :math:`D = min(T, C) > 0`.
#   This is the minimum between the date of the experienced event, represented by the
#   random variable :math:`T`, and the censoring date, represented by :math:`C`.
#
# In this dataset, we have 40% of censoring.

# %%
y["event"].value_counts(normalize=True)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# %%
#
# Using SurvivalBoost to estimate the survival function
# -----------------------------------------------------
#
# Here our quantity of interest is the survival probability:
#
# .. math::
#
#     S(t | X=x) = P(T > t | X=x)
#
# This represents the probability that an event doesn't occur at or before some
# given time :math:`t`, i.e. that it happens at some time :math:`T > t`,
# given the patient features x.
#
# SurvivalBoost is a scikit-learn compatible model which expect a dataframe (or
# array-like) input ``X``, and a dataframe with columns "event" and "duration"
# ``y``. This allows SurvivalBoost to estimate the survival function :math:`S`.
from hazardous import SurvivalBoost

survival_boost = SurvivalBoost(
    learning_rate=0.05,
    n_iter=100,
    n_iter_before_feedback=50,
    hard_zero_fraction=0.1,
    min_samples_leaf=50,
    show_progressbar=False,
    n_horizons_per_observation=3,
    random_state=0,
).fit(X_train, y_train)

survival_boost

# %%
#
# SurvivalBoost can then predict the survival function for each patient,
# according to the time grid of horizons passed by the user.
# **The time grid is learned during fit but can be passed during the prediction**
# with the parameter ``times``.

predicted_curves = survival_boost.predict_cumulative_incidence(
    X_test,
    times=None,
)

survival_func = predicted_curves[:, 0]  # survival function S(t)
incidence_func = predicted_curves[
    :, 1
]  # cumulative incidence of the event (here death)

# %%
# Let's plot the estimated survival function for some patients.
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

patient_ids_to_plot = [0, 1, 2, 3]

for idx in patient_ids_to_plot:
    ax.plot(survival_boost.time_grid_, survival_func[idx], label=f"Patient {idx}")

    # plot symbols for death or censoring
    event = y_test.iloc[idx]["event"]
    duration = y_test.iloc[idx]["duration"]

    # find the index of time closest to duration
    jdx = np.searchsorted(survival_boost.time_grid_, duration)
    smiley = "☠️" if event == 1 else "✖"
    ax.text(
        duration,
        survival_func[idx, jdx],
        smiley,
        fontsize=20,
        color=ax.lines[idx].get_color(),
    )

ax.legend()
ax.set_title("")
ax.set_xlabel("Months")
ax.set_ylabel("Predicted Survival Probability")

plt.show()

# %%
#
# Measuring features impact on predictions
# ----------------------------------------
#
# We can also observe the survival function by age group, or by chemotherapy
# treatment to show the impact the model attribute to these features. We do
# something akin to Partial Dependence Plots, where we sample the feature
# independently from the other features, to eliminate correlations.
#
# We create a synthetic dataset where age (``x8``) is resampled to reduce
# confounder bias.

X_synthetic = X_train.copy()
# Age varies from 20 to 80
X_synthetic["x8"] = np.linspace(20, 80, len(X_synthetic))

# Predict cumulative incidence on the synthetic dataset
survival_func_synthetic = survival_boost.predict_survival_function(X_synthetic)

# Create age bins and sort them by the left bin edge
age_bins = pd.cut(X_synthetic["x8"], bins=[0, 30, 40, 50, 60, 70, 80, 90, 100])
age_groups = sorted(age_bins.unique(), key=lambda x: x.left)

# Create a colormap
fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis", len(age_groups))

for idx, age_group in enumerate(age_groups):
    # Get the mask of patients in the current age group
    mask = age_bins == age_group

    # Calculate the mean and std cumulative incidence for the current age group
    mean_survival = survival_func_synthetic[mask].mean(axis=0)
    std_survival = survival_func_synthetic[mask].std(axis=0)

    # Plot with color from colormap
    ax.plot(
        survival_boost.time_grid_,
        mean_survival,
        label=f"Age {age_group}",
        color=cmap(idx),
        linewidth=3,
    )
    # Add ribbon for std
    ax.fill_between(
        survival_boost.time_grid_,
        mean_survival - std_survival,
        mean_survival + std_survival,
        color=cmap(idx),
        alpha=0.3,
    )

ax.legend()
ax.set_title("Survival function by age")
ax.set_xlabel("Months")
ax.set_ylabel("Estimated Survival Probability")

plt.show()

# %%
#
# Unsurprisingly, the cumulative incidence of death mostly increases with age.
# We can do the same thing with chemotherapy treatement.
#
# Let's create a synthetic dataset where chemotherapy (``x6``)
# alternates between 0 and 1.

X_synthetic = X_train.copy()
X_synthetic["x6"] = np.tile([0, 1], X_synthetic.shape[0] // 2)

survival_func_synthetic = survival_boost.predict_survival_function(
    X_synthetic,
)

fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis", 2)

for chemo_group in [0, 1]:
    mask = X_synthetic["x6"] == chemo_group
    mean_survival = survival_func_synthetic[mask].mean(axis=0)
    std_survival = survival_func_synthetic[mask].std(axis=0)
    ax.plot(
        survival_boost.time_grid_,
        mean_survival,
        label=(
            "Treated with Chemotherapy"
            if chemo_group == 1
            else "Not Treated with Chemotherapy"
        ),
        color=cmap(chemo_group),
        linewidth=3,
    )
    ax.fill_between(
        survival_boost.time_grid_,
        mean_survival - std_survival,
        mean_survival + std_survival,
        color=cmap(chemo_group),
        alpha=0.3,
    )

ax.legend()
ax.set_title("Survival function by chemotherapy treatment")
ax.set_xlabel("Months")
ax.set_ylabel("Estimated Survival Probability")

plt.show()

# %%
#
# People treated with chemotherapy probably have more serious stages of
# cancer, which is reflected by the lower estimated survival function. This is a
# reminder that the estimate is not causal.
#
# Let's now attempt to quantify how a survival curve estimated on a training set
# performs on a test set.
#
# Survival model evaluation
# -------------------------
#
# The Brier score and the C-index are measures that **assess the quality of a
# predicted survival curve** on a finite data sample.
#
# - **The Brier score in time is a proper scoring rule**, meaning that an
#   estimate of the survival probabilities at a given time :math:`t` has minimal Brier
#   score if and only if it matches the oracle survival probabilities induced by
#   the underlying data generating process. In that respect the **Brier score**
#   assesses both the **calibration** and the **ranking power** of a survival
#   probability estimator. It is comprised between 0 and 1 (lower is better). It
#   answers the question *"how close to the real probabilities are our estimates?"*.
#
# - On the other hand, the **C-index** only assesses the **ranking power**: it
#   represents the probability that, for a randomly selected pair of patients,
#   the patient with the higher estimated survival probability will survive
#   longer than the other. It is comprised between 0 and 1 (higher is better),
#   with 0.5 corresponding to a random prediction.
#
# .. dropdown:: Mathematical formulation (Brier score)
#
#     .. math::
#
#         \mathrm{BS}^c(t) = \frac{1}{n} \sum_{i=1}^n I(d_i \leq t \land \delta_i = 1)
#         \frac{(0 - \hat{S}(t | \mathbf{x}_i))^2}{\hat{G}(d_i)} + I(d_i > t)
#         \frac{(1 - \hat{S}(t | \mathbf{x}_i))^2}{\hat{G}(t)}
#
#     In the survival analysis context, the Brier Score can be seen as the Mean
#     Squared Error (MSE) between our probability :math:`\hat{S}(t)` and our
#     target label :math:`\delta_i \in {0, 1}`, weighted by the inverse probability
#     of censoring :math:`\frac{1}{\hat{G}(t)}`.
#     In practice we estimate :math:`\hat{G}(t)` using a variant of
#     the Kaplan-Estimator with swapped event indicator.
#
#     - When no event or censoring has happened at :math:`t` yet, i.e.
#       :math:`I(d_i > t)`, we penalize a low probability of survival with
#       :math:`(1 - \hat{S}(t|\mathbf{x}_i))^2`.
#     - Conversely, when an individual has experienced an event before :math:`t`, i.e.
#       :math:`I(d_i \leq t \land \delta_i = 1)`, we penalize a high probability
#       of survival with :math:`(0 - \hat{S}(t|\mathbf{x}_i))^2`.
#
# .. dropdown:: Mathematical formulation (C-index)
#
#     .. math::
#
#         \mathrm{C_{index}} = \frac{\sum_{i,j} I(d_i < d_j \space \land \space
#         \delta_i = 1 \space \land \space \mu_i < \mu_j)}
#         {\sum_{i,j} I(d_i < d_j \space \land \space \delta_i = 1)}
#
# Additionnaly, we compute the Integrated Brier Score (IBS) which we will use to
# summarize the Brier score in time:
#
# .. math::
#
#     \mathrm{IBS} = \frac{1}{t_{max} - t_{min}}\int^{t_{max}}_{t_{min}}
#     \mathrm{BS(t)} dt
#
from hazardous.metrics import integrated_brier_score_survival

brier_score_hazardous = integrated_brier_score_survival(
    y_train,
    y_test,
    survival_func,
    times=survival_boost.time_grid_,
)
print(f"Brier score for SurvivalBoost: {brier_score_hazardous:.4f}")

# %%
#
# We can compare this Brier score to the Brier score of a simple Kaplan-Meier
# estimator, which doesn't take the patient features into account.
from lifelines import KaplanMeierFitter

km_model = KaplanMeierFitter()
km_model.fit(y["duration"], y["event"])
survival_curve_agg_km = km_model.survival_function_at_times(
    survival_boost.time_grid_,
)

# To get individual survival curves, we duplicate the survival curve
# for each patient
survival_curve_km = np.tile(survival_curve_agg_km, (len(X_test), 1))

brier_score_km = integrated_brier_score_survival(
    y_train,
    y_test,
    survival_curve_km,
    times=survival_boost.time_grid_,
)
print(f"Brier score for Kaplan-Meier: {brier_score_km:.4f}")


# %%
#
# Let's also compute the concordance index for both the Kaplan-Meier and
# SurvivalBoost.
#
# The concordance_index metric from scikit-survival expects a structured
# array of events and durations.
#
def make_recarray(y):
    event, duration = y["event"].values, y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(len(event))],
        dtype=[("e", bool), ("t", float)],
    )


train_data = make_recarray(y_train)
test_data = make_recarray(y_test)

# %%
from sksurv.metrics import concordance_index_ipcw

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

# %%
#
# :math:`0.5` corresponds to random chance, which makes sense as the Kaplan-Meier
# estimator doesn't depend on the patient features.
#
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
