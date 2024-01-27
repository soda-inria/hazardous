# %%
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

from hazardous.metrics._brier_score import (
    brier_score_incidence,
)
from hazardous._aalan_johansen import AalenJohansenEstimator
from display_utils import (
    aggregate_result,
    get_estimator,
    make_query,
    make_time_grid,
    load_dataset,
)

sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

USER = "vincentmaladiere"
DATASET_NAME = "weibull"
SEER_PATH = "../hazardous/data/seer_cancer_cardio_raw_data.txt"
SEED = 0


path_session_dataset = Path("2024-01-15") / DATASET_NAME
estimator_names = ["gbmi_10", "gbmi_20"]

data_params = dict(
    n_events=3,
    independent_censoring=True,
    complex_features=True,
    censoring_relative_scale=1.5,
    n_samples=10_000,
)

df = aggregate_result(path_session_dataset, estimator_names)
mask = make_query(data_params)
df = df.query(mask)

bunch = load_dataset(DATASET_NAME, data_params)
X_test, y_test = bunch.X, bunch.y

time_grid = make_time_grid(y_test["duration"])

aj = AalenJohansenEstimator().fit(y_test)
y_pred_aj = aj.predict_cumulative_incidence(X_test, time_grid)

all_y_pred = {"Aalen-Johansen": y_pred_aj}
for estimator_name, df_est in df.groupby("estimator_name"):
    estimator = get_estimator(df_est, estimator_name)
    y_pred = estimator.predict_cumulative_incidence(X_test, time_grid)
    y_train = estimator.y_train
    all_y_pred[estimator_name] = y_pred

fig, axes = plt.subplots(
    figsize=(5, 3),
    sharey=True,
    ncols=data_params["n_events"],
    dpi=300,
)

for idx, ax in enumerate(axes):
    for estimator_name, y_pred in all_y_pred.items():
        brier_scores = brier_score_incidence(
            y_train,
            y_test,
            y_pred[idx + 1],
            times=time_grid,
            event_of_interest=idx + 1,
        )

        ax.plot(time_grid, brier_scores, label=estimator_name)

    ax.set_title(f"Event {idx+1}")

axes[0].set(ylabel="Brier Scores")
axes[-1].legend()
sns.move_legend(axes[-1], "upper left", bbox_to_anchor=(1, 1))
sns.despine()

file_path = f"/Users/{USER}/Desktop/06_brier_score.pdf"
fig.savefig(file_path, format="pdf")

# %%
