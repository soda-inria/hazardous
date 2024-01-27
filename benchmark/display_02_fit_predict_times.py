# %%
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

from display_utils import aggregate_result, make_query


sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

USER = "vincentmaladiere"
DATASET_NAME = "weibull"


path_session_dataset = Path("2024-01-15") / DATASET_NAME
estimator_names = ["gbmi_10", "gbmi_20"]
df = aggregate_result(path_session_dataset, estimator_names)

data_params = dict(
    n_events=3,
    independent_censoring=True,
    complex_features=True,
)
mask = make_query(data_params)
df = df.query(mask)

min_time = df["mean_fit_time"].min()
max_time = df["mean_fit_time"].max()

rename_estimator = {
    "gbmi_10": "GBMI 10",
    "gbmi_20": "GBMI 20",
}
df["estimator_name"] = df["estimator_name"].map(rename_estimator)

for estimator_name, df_est in df.groupby("estimator_name"):
    df_est = (
        df_est.sort_values(["n_samples", "censoring_relative_scale"])
        .pivot(
            index="n_samples",
            columns="censoring_relative_scale",
            values="mean_fit_time",
        )
        .sort_index(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    cbar_kws = {"label": "time (s)"}
    ax = sns.heatmap(
        df_est,
        vmin=min_time,
        vmax=max_time,
        cmap=sns.color_palette("Oranges"),
        cbar_kws=cbar_kws,
        ax=ax,
    )
    ax.set(
        xlabel="Censoring relative scale",
        ylabel="# training samples",
        title=estimator_name,
    )

    file_path = f"/Users/{USER}/Desktop/02_fit_predict_times_{estimator_name}.pdf"
    fig.savefig(file_path, format="pdf")

# %%
