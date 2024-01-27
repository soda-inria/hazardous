# %%
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from hazardous.metrics._brier_score import integrated_brier_score_incidence
from display_utils import (
    aggregate_result,
    make_time_grid,
    make_query,
    get_estimator,
    load_dataset,
)

sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

USER = "vincentmaladiere"
DATASET_NAME = "weibull"
SEEDS = {"weibull": list(range(5)), "seer": [0]}[DATASET_NAME]


path_session_dataset = Path("2024-01-15") / DATASET_NAME
estimator_names = ["gbmi_10", "gbmi_20"]
results = aggregate_result(path_session_dataset, estimator_names)

data_params = dict(
    n_events=3,
    independent_censoring=True,
    complex_features=True,
    censoring_relative_scale=1.5,
)
mask = make_query(data_params)
results = results.query(mask)
n_samples_list = results["n_samples"].unique()

all_y_pred = []
for n_training_samples in tqdm(n_samples_list):
    results_samples = results.loc[results["n_samples"] == n_training_samples]

    for seed in SEEDS:
        data_params["n_samples"] = 10_000
        bunch = load_dataset(DATASET_NAME, data_params, random_state=seed)
        X_test, y_test = bunch.X, bunch.y
        time_grid = make_time_grid(y_test["duration"], n_steps=10)

        for estimator_name, results_est in results_samples.groupby("estimator_name"):
            mean_fit_time = results_est["mean_fit_time"].values[0]
            estimator = get_estimator(results_est, estimator_name)
            y_train = estimator.y_train
            y_pred = estimator.predict_cumulative_incidence(X_test, time_grid)

            for event_idx in range(data_params["n_events"]):
                ibs = integrated_brier_score_incidence(
                    y_train,
                    y_test,
                    y_pred[event_idx + 1],
                    time_grid,
                    event_of_interest=event_idx + 1,
                )

                all_y_pred.append(
                    dict(
                        n_samples=n_training_samples,
                        mean_fit_time=mean_fit_time,
                        seed=seed,
                        estimator_name=estimator_name,
                        event_of_interest=event_idx + 1,
                        test_score=ibs,
                    )
                )

all_y_pred = (
    pd.DataFrame(all_y_pred)
    .groupby(["n_samples", "mean_fit_time", "estimator_name", "seed"])[
        "test_score"
    ]  # agg events
    .mean()
    .reset_index()
    .groupby(["n_samples", "mean_fit_time", "estimator_name"])[
        "test_score"
    ]  # agg seeds
    .agg(["mean", "std"])
    .reset_index()
    .rename(
        columns=dict(
            mean="mean_test_score",
            std="std_test_score",
        )
    )
    .sort_values("mean_fit_time")
)

fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

sns.lineplot(
    all_y_pred,
    x="mean_fit_time",
    y="mean_test_score",
    hue="estimator_name",
    ax=ax,
)

for estimator_name, y_pred in all_y_pred.groupby("estimator_name"):
    low = y_pred["mean_test_score"] - y_pred["std_test_score"]
    high = y_pred["mean_test_score"] + y_pred["std_test_score"]
    ax.fill_between(
        y_pred["mean_fit_time"],
        y1=low,
        y2=high,
        alpha=0.3,
        interpolate=True,
    )
ax.legend()
ax.set(
    xlabel="Time to fit (s)",
    ylabel="Average IPSR",
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
sns.despine()

file_path = f"/Users/{USER}/Desktop/04_performance_vs_time.pdf"
fig.savefig(file_path, format="pdf")

# %%
