# %%
import numpy as np
from complex_synthetic_data import complex_data
from time import perf_counter
import matplotlib.pyplot as plt

from hazardous import GradientBoostingIncidence
from lifelines import AalenJohansenFitter

rng = np.random.default_rng(0)
seed = 0
DEFAULT_SHAPE_RANGES = (
    (0.7, 0.9),
    (1.0, 1.0),
    (2.0, 3.0),
)

DEFAULT_SCALE_RANGES = (
    (1, 20),
    (1, 10),
    (1.5, 5),
)
n_events = 3

# %%

X, y = complex_data()
n_samples = len(X)
calculate_variance = n_samples <= 5_000
aj = AalenJohansenFitter(calculate_variance=calculate_variance, seed=0)

gb_incidence = GradientBoostingIncidence(
    learning_rate=0.03,
    n_iter=200,
    max_leaf_nodes=5,
    hard_zero_fraction=0.1,
    min_samples_leaf=50,
    loss="ibs",
    show_progressbar=False,
    random_state=0,
)

# %%
#
# CIFs estimated on uncensored data
# ---------------------------------
#
# Let's now estimate the CIFs on uncensored data and plot them against the
# theoretical CIFs:


def plot_cumulative_incidence_functions(X, y, gb_incidence=None, aj=None):
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
            cif_pred = gb_incidence.predict_cumulative_incidence(
                X[0:1], coarse_timegrid
            )[0]
            duration = perf_counter() - tic
            print(f"GB Incidence for event {event_id} prediction in {duration:.3f} s")
            ax.plot(
                coarse_timegrid,
                cif_pred,
                label="GradientBoostingIncidence",
            )
            ax.set(title=f"Event {event_id}")

        if aj is not None:
            tic = perf_counter()
            aj.fit(y["duration"], y["event"], event_of_interest=event_id)
            duration = perf_counter() - tic
            print(f"Aalen-Johansen for event {event_id} fit in {duration:.3f} s")
            aj.plot(label="Aalen-Johansen", ax=ax)

        if event_id == 1:
            ax.legend(loc="lower right")
        else:
            ax.legend().remove()


# %%

plot_cumulative_incidence_functions(
    X,
    y,
    gb_incidence=gb_incidence,
    aj=aj,
)
