# %%
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

from lifelines import AalenJohansenFitter

from display_utils import aggregate_result, get_estimator, make_query, make_time_grid
from hazardous.data._competing_weibull import make_synthetic_competing_weibull

sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

USER = "vincentmaladiere"
DATASET_NAME = "weibull"

path_session_dataset = Path("2024-01-15") / DATASET_NAME
estimator_names = ["gbmi_10", "gbmi_20"]
results = aggregate_result(path_session_dataset, estimator_names)

data_params = dict(
    n_events=3,
    independent_censoring=True,
    complex_features=True,
    censoring_relative_scale=1.5,
    n_samples=10_000,
)

mask = make_query(data_params)
results = results.query(mask)

bunch = make_synthetic_competing_weibull(
    return_X_y=False,
    **data_params,
)
X, y, y_uncensored = bunch.X, bunch.y, bunch.y_uncensored
time_grid = make_time_grid(y["duration"])

print(f"Censoring rate {(y['event'] == 0).mean():.2%}")

fig, axes = plt.subplots(
    figsize=(5, 3),
    ncols=data_params["n_events"],
    sharey=True,
    dpi=300,
)

for estimator_name, results_est in results.groupby("estimator_name"):
    estimator = get_estimator(results_est, estimator_name)
    y_pred = estimator.predict_cumulative_incidence(X, times=time_grid)

    for idx, ax in enumerate(axes):
        y_pred_marginal = y_pred[idx + 1].mean(axis=0)
        ax.plot(time_grid, y_pred_marginal, label=estimator_name)
        ax.legend()

for idx, ax in enumerate(axes):
    aj = AalenJohansenFitter(calculate_variance=False).fit(
        durations=y["duration"],
        event_observed=y["event"],
        event_of_interest=idx + 1,
    )
    aj.plot(ax=ax, label="Aalen Johansen", color="k")

    aj.fit(
        durations=y["duration"],
        event_observed=y_uncensored["event"],
        event_of_interest=idx + 1,
    )
    aj.plot(ax=ax, label="Aalen Johansen uncensored", color="k", linestyle="--")

    ax.set(
        title=f"Event {idx+1}",
        xlabel="",
    )
    ax.get_xaxis().set_visible(False)

    if ax is not axes[-1]:
        ax.get_legend().remove()

sns.move_legend(axes[-1], "upper left", bbox_to_anchor=(1, 1))
sns.despine()

file_path = f"/Users/{USER}/Desktop/05_marginal_incidence.pdf"
fig.savefig(file_path, format="pdf")


# %%
