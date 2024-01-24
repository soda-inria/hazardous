# %%
from matplotlib import pyplot as plt
import seaborn as sns

from hazardous.data._competing_weibull import make_synthetic_competing_weibull

seed = 0
independent_censoring = False
complex_features = True

bunch = make_synthetic_competing_weibull(
    n_samples=10_000,
    n_events=3,
    n_features=5,
    return_X_y=False,
    independent_censoring=independent_censoring,
    censoring_relative_scale=1.5,
    random_state=seed,
    complex_features=complex_features,
)
X, y, y_uncensored = bunch.X, bunch.y, bunch.y_uncensored

censoring_rate = (y["event"] == 0).mean()
censoring_kind = "independent" if independent_censoring else "dependent"
ax = sns.histplot(
    y,
    x="duration",
    hue="event",
    multiple="stack",
    palette="magma",
)
ax.set_title(f"{censoring_kind} censoring rate {censoring_rate:.2%}")

# %%
import seaborn as sns
from hazardous.data._seer import (
    load_seer,
    NUMERIC_COLUMN_NAMES,
    CATEGORICAL_COLUMN_NAMES,
)


X, y = load_seer(
    "../hazardous/data/seer_cancer_cardio_raw_data.txt",
    return_X_y=True,
    survtrace_preprocessing=True,
)
X = X[NUMERIC_COLUMN_NAMES + CATEGORICAL_COLUMN_NAMES]

# %%

X = X.dropna()
y = y.iloc[X.index]
y["event"] = y["event"].replace(3, 0)
censoring_rate = (y["event"] == 0).mean()
ax = sns.histplot(
    y,
    x="duration",
    hue="event",
    multiple="stack",
    palette="magma",
)
ax.set_title(f"Censoring rate {censoring_rate:.2%}")

print(X.isna().sum())

# %%

from sklearn.model_selection import train_test_split

from hazardous.survtrace._model import SurvTRACE

object_columns = X.select_dtypes(["object", "string"]).columns
X[object_columns] = X[object_columns].astype("category")

numeric_columns = X.select_dtypes("number").columns
X[numeric_columns] = X[numeric_columns].astype("float64")

X_train, X_test, y_train, y_test = train_test_split(X, y)

survtrace = SurvTRACE()
survtrace.fit(X_train, y_train)
# %%

from lifelines import AalenJohansenFitter

y_pred = survtrace.predict_cumulative_incidence(X_test)

n_events = y_pred.shape[0]
time_grid = survtrace.target_encoder_.time_grid_

fig, axes = plt.subplots(ncols=n_events)

for event_idx, ax in enumerate(axes):
    ax.plot(
        time_grid,
        y_pred[event_idx].mean(axis=0),
        label="SurvTRACE",
    )

    aj = AalenJohansenFitter(calculate_variance=False).fit(
        durations=y_test["duration"],
        event_observed=y_test["event"],
        event_of_interest=event_idx + 1,
    )
    aj.plot(ax=ax, label="AJ")
    ax.set_title(f"Event {event_idx+1}")
    ax.grid()
    ax.legend()

# %%

y_pred[:, 0, :]
# %%
