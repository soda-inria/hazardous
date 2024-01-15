# %%
# We first show both models on uncensored data:
# Creating dummy features, we just want to show if the CIFs
# are well calibrated.

import numpy as np
from scipy.stats import weibull_min
import pandas as pd

seed = 0
rng = np.random.default_rng(seed)
n_samples = 3_000

X_dummy = np.zeros(shape=(n_samples, 1), dtype=np.float32)

base_scale = 1_000.0  # some arbitrary time unit

distributions = [
    {"event_id": 1, "scale": 10 * base_scale, "shape": 0.5},
    {"event_id": 2, "scale": 3 * base_scale, "shape": 1},
    {"event_id": 3, "scale": 3 * base_scale, "shape": 5},
]
event_times = np.concatenate(
    [
        weibull_min.rvs(
            dist["shape"],
            scale=dist["scale"],
            size=n_samples,
            random_state=rng,
        ).reshape(-1, 1)
        for dist in distributions
    ],
    axis=1,
)
first_event_idx = np.argmin(event_times, axis=1)

y_uncensored = pd.DataFrame(
    dict(
        event=first_event_idx + 1,  # 0 is reserved as the censoring marker
        duration=event_times[np.arange(n_samples), first_event_idx],
    )
)
y_uncensored["event"].value_counts().sort_index()
t_max = y_uncensored["duration"].max()

# %%

from hazardous._gb_multi_incidence import GBMultiIncidence
from hazardous import GradientBoostingIncidence
from lifelines import AalenJohansenFitter

gb_multi_incidence = GBMultiIncidence(
    learning_rate=0.03,
    n_iter=100,
    max_leaf_nodes=5,
    hard_zero_fraction=0.1,
    min_samples_leaf=50,
    loss="competing_risks",
    show_progressbar=False,
    random_state=0,
)

gb_incidence = GradientBoostingIncidence(
    learning_rate=0.03,
    n_iter=100,
    max_leaf_nodes=5,
    hard_zero_fraction=0.1,
    min_samples_leaf=50,
    loss="ibs",
    show_progressbar=False,
    random_state=0,
)

aj = AalenJohansenFitter(calculate_variance=False, seed=0)

# %%

from time import perf_counter
import matplotlib.pyplot as plt


def plot_cumulative_incidence_functions(
    X,
    y,
    gb_multi_incidence=None,
    gb_incidence=None,
    aj=None,
    X_test=None,
    y_test=None,
    verbose=False,
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
    if gb_multi_incidence is not None:
        tic = perf_counter()
        gb_multi_incidence.fit(X, y)
        duration = perf_counter() - tic
        if verbose:
            print(f"GB Incidence fit in {duration:.3f} s")
        cifs_pred_multi = gb_multi_incidence.predict_cumulative_incidence(
            X, coarse_timegrid
        )

    for event_id, ax in enumerate(axes, 1):
        if gb_multi_incidence is not None:
            cif_mean_multi = cifs_pred_multi[event_id].mean(axis=0)

            if verbose:
                brier_score_train = -gb_multi_incidence.score(X, y)
                print(f"Brier score on training data: {brier_score_train:.3f}")
                if X_test is not None:
                    brier_score_test = -gb_multi_incidence.score(X_test, y_test)
                    print(
                        f"Brier score on testing data: {brier_score_test:.3f}",
                    )
            ax.plot(
                coarse_timegrid,
                cif_mean_multi,
                label="GradientBoostingMultiIncidence",
            )
            ax.set(title=f"Event {event_id}")

        if gb_incidence is not None:
            tic = perf_counter()
            gb_incidence.set_params(event_of_interest=event_id)
            gb_incidence.fit(X, y)
            duration = perf_counter() - tic

            if verbose:
                print(f"GB Incidence for event {event_id} fit in {duration:.3f} s")

            tic = perf_counter()
            cifs_pred = gb_incidence.predict_cumulative_incidence(X, coarse_timegrid)
            cif_mean = cifs_pred.mean(axis=0)
            duration = perf_counter() - tic

            if verbose:
                print(
                    f"GB Incidence for event {event_id} prediction in {duration:.3f} s"
                )

            if verbose:
                brier_score_train = -gb_incidence.score(X, y)
                print(f"Brier score on training data: {brier_score_train:.3f}")
                if X_test is not None:
                    brier_score_test = -gb_incidence.score(X_test, y_test)
                    print(
                        f"Brier score on testing data: {brier_score_test:.3f}",
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
            if verbose:
                print(f"Aalen-Johansen for event {event_id} fit in {duration:.3f} s")

            aj.plot(label="Aalen-Johansen", ax=ax)
            ax.set_xlim(0, 8_000)
            ax.set_ylim(0, 0.5)

        if event_id == 1:
            ax.legend(loc="lower right")
        else:
            ax.legend().remove()

        if verbose:
            print("=" * 16, "\n")


# %%
plot_cumulative_incidence_functions(
    X_dummy,
    y_uncensored,
    gb_multi_incidence=gb_multi_incidence,
    gb_incidence=gb_incidence,
    aj=aj,
)
# %%
# Creating harder features with interactions, we want to show
# the different marginal CIFs for each model.

from hazardous.data._competing_weibull import make_synthetic_competing_weibull


bunch = make_synthetic_competing_weibull(
    n_samples=3_000,
    base_scale=1_000,
    n_features=10,
    features_rate=0.2,
    degree_interaction=2,
    independent_censoring=True,
    features_censoring_rate=0.2,
    feature_rounding=3,
    target_rounding=None,
    censoring_relative_scale=1.0,
    complex_features=True,
    random_state=seed,
)
X, y_censored, y_uncensored = bunch.X, bunch.y, bunch.y_uncensored
X.shape, y_censored.shape

# %%
from sklearn.model_selection import train_test_split

gb_multi_incidence = GBMultiIncidence(
    learning_rate=0.03,
    n_iter=100,
    max_leaf_nodes=5,
    hard_zero_fraction=0.1,
    min_samples_leaf=50,
    loss="inll",
    show_progressbar=False,
    random_state=0,
)

gb_incidence = GradientBoostingIncidence(
    learning_rate=0.03,
    n_iter=100,
    max_leaf_nodes=5,
    hard_zero_fraction=0.1,
    min_samples_leaf=50,
    loss="ibs",
    show_progressbar=False,
    random_state=0,
)

X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_censored, test_size=0.3, random_state=seed
)
y_train_u = y_uncensored.loc[y_train_c.index]
y_test_u = y_uncensored.loc[y_test_c.index]

plot_cumulative_incidence_functions(
    X_train,
    y_train_u,
    gb_incidence=gb_incidence,
    gb_multi_incidence=gb_multi_incidence,
    aj=aj,
    X_test=X_test,
    y_test=y_test_u,
)

plot_cumulative_incidence_functions(
    X_train,
    y_train_c,
    gb_incidence=gb_incidence,
    gb_multi_incidence=gb_multi_incidence,
    aj=aj,
    X_test=X_test,
    y_test=y_test_c,
)


# %%
# We show that, with the GBMultiIncidence model, by definition,
# the CIFS + survival function = 1. We also show that, with the
# GradientBoostingIncidence model, the CIFS + survival function !=1.


def plot_cifs(
    X,
    y,
    X_test,
    color_event={0: "red", 1: "blue", 2: "orange", 3: "green"},  # (n_events + 1)
    gb_multi_incidence=None,
    gb_incidence=None,
    verbose=False,
):
    n_events = y["event"].max()
    t_max = y["duration"].max()
    _, axes = plt.subplots(figsize=(12, 5), ncols=2, sharey=True)
    # Compute the estimate of the CIFs on a coarse grid.
    coarse_timegrid = np.linspace(0, t_max, num=100)
    censoring_fraction = (y["event"] == 0).mean()
    plt.suptitle(
        "Cause-specific cumulative incidence functions"
        f" ({censoring_fraction:.1%} censoring)"
    )

    if gb_multi_incidence is not None:
        gb_multi_incidence = GBMultiIncidence(
            learning_rate=0.03,
            n_iter=100,
            max_leaf_nodes=5,
            hard_zero_fraction=0.1,
            min_samples_leaf=50,
            loss="inll",
            show_progressbar=False,
            random_state=0,
        )
        tic = perf_counter()
        gb_multi_incidence.fit(X, y)
        duration = perf_counter() - tic
        if verbose:
            print(f"GB Multi Incidence fit in {duration:.3f} s")
        cifs_pred_multi = gb_multi_incidence.predict_cumulative_incidence(
            X_test, coarse_timegrid
        )

        cif_mean_multi = cifs_pred_multi.mean(axis=1)
        for event in range(1, n_events + 1):
            axes[0].plot(
                coarse_timegrid,
                cif_mean_multi[event],
                label=f"Event {event}",
                color=color_event[event],
            )
        axes[0].plot(
            coarse_timegrid,
            cif_mean_multi[0],
            label="Any-event survival",
            color=color_event[0],
        )
        axes[0].plot(
            coarse_timegrid,
            cif_mean_multi.sum(axis=0),
            label="Total CIFS",
            linestyle="--",
            color="black",
        )
        axes[0].set(
            title="GBMultiIncidence",
            ylabel="Cumulative Incidence",
        )
        axes[0].legend()

    if gb_incidence is not None:
        cif_models = {}
        for event_id in range(1, n_events + 1):
            gb_incidence_event = GradientBoostingIncidence(
                learning_rate=0.03,
                n_iter=100,
                max_leaf_nodes=5,
                hard_zero_fraction=0.1,
                min_samples_leaf=50,
                loss="ibs",
                show_progressbar=False,
                random_state=0,
            )
            gb_incidence_event.set_params(event_of_interest=event_id)
            gb_incidence_event.fit(X, y)
            cif_models[event_id] = gb_incidence_event

        total_mean_cif = np.zeros(coarse_timegrid.shape[0])

        gb_cif_cumulative_incidence_curves = {}
        for event_id in range(1, n_events + 1):
            cif_curves_k = cif_models[event_id].predict_cumulative_incidence(
                X_test, coarse_timegrid
            )
            gb_cif_cumulative_incidence_curves[event_id] = cif_curves_k
            mean_cif_curve_k = cif_curves_k.mean(axis=0)  # average over test points
            axes[1].plot(
                coarse_timegrid,
                mean_cif_curve_k,
                label=f"event {event_id}",
                color=color_event[event_id],
            )
            total_mean_cif += mean_cif_curve_k

        gb_incidence_event.set_params(event_of_interest="any")
        gb_incidence_event.fit(X, y)
        gb_cif_survival_curves = gb_incidence_event.predict_survival_function(
            X_test, coarse_timegrid
        )
        mean_survival_curve = gb_cif_survival_curves.mean(axis=0)
        total_mean_cif += mean_survival_curve
        axes[1].plot(
            coarse_timegrid,
            mean_survival_curve,
            label="Any-event survival",
            color=color_event[0],
        )
        axes[1].plot(
            coarse_timegrid,
            total_mean_cif,
            label="total",
            linestyle="--",
            color="black",
        )
        axes[1].set(
            title="GradientBoostingIncidence",
        )
        axes[1].legend()
    plt.show()


# %%
plot_cifs(
    X_train,
    y_train_c,
    gb_incidence=True,
    gb_multi_incidence=True,
    X_test=X_test,
)

# %%
# We study the effect of the Time Sampler.
# The KM estimator allows us to sample times according to the
# distribution of any events.

from hazardous._gb_multi_incidence import WeightedMultiClassTargetSampler


for uniform_sampling in [True, False]:
    sampler = WeightedMultiClassTargetSampler(
        y_train_c,
        hard_zero_fraction=0.01,
        random_state=None,
        ipcw_est=None,
        n_iter_before_feedback=20,
        uniform_sampling=uniform_sampling,
    )

    all_times = []
    for iter in range(100):
        times, _, _ = sampler.draw()
        all_times.append(times)
    all_times = np.array(all_times).flatten()

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    if uniform_sampling:
        ax.set_title("Uniform sampling")
    else:
        ax.set_title("Time sampling using KM")

    ax.hist(all_times)
# %%
