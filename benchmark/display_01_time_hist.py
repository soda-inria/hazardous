# %%
import seaborn as sns
from matplotlib import pyplot as plt

from hazardous.data._competing_weibull import make_synthetic_competing_weibull


sns.set_style(style="white")
sns.set_context("paper")
plt.style.use("seaborn-v0_8-talk")

USER = "vincentmaladiere"


_, y = make_synthetic_competing_weibull(
    n_events=3,
    censoring_relative_scale=0.8,
    n_samples=100_000,
    n_features=10,
    return_X_y=True,
    independent_censoring=False,
)
cut_duration = y["duration"].quantile(0.99)
y = y.loc[y["duration"] < cut_duration]

fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
sns.histplot(
    y, x="duration", hue="event", multiple="stack", palette="colorblind", bins=25, ax=ax
)
sns.despine(ax=ax)
ax.set(
    xlabel="Time to event",
    ylabel="Total",
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

file_path = f"/Users/{USER}/Desktop/01_time_hist.pdf"
fig.savefig(file_path, format="pdf")

# %%
