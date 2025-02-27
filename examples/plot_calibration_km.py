# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter

from hazardous._km_sampler import _KaplanMeierSampler  # , _AalenJohansenSampler
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.calibration._km_calibration import km_cal, recalibrate_survival_function
from hazardous.metrics import integrated_brier_score_survival

# %%
# KM-calibration without censoring
n_samples = 10000
X, y = make_synthetic_competing_weibull(
    n_samples=n_samples,
    return_X_y=True,
    n_events=1,
    censoring_relative_scale=0.5,
    random_state=0,
)
sns.histplot(data=y, x="duration", hue="event", bins=50, kde=True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.5)
X_train, X_conf, y_train, y_conf = train_test_split(
    X_train, y_train, random_state=0, test_size=0.5
)


# %%
kaplan_sampler = _KaplanMeierSampler()
kaplan_sampler.fit(y_train)

surv_func = kaplan_sampler.survival_func_
# Test if KM is KM-calibrated
times = kaplan_sampler.unique_times_
km_km = km_cal(
    y_conf,
    times,
    np.repeat(surv_func(times), len(y), axis=0).reshape(len(times), len(y)).T,
)

plt.plot(times, surv_func(times), label="KM")
plt.legend()
plt.title(km_km)
plt.show()

# %%
# Test if Cox is KM calibrated

cox = CoxPHFitter()
cox.fit(y_train, duration_col="duration", event_col="event")
cox_surv_probs = cox.predict_survival_function(X_test).T
times_cox = cox_surv_probs.columns
km_cox = km_cal(y_conf, cox_surv_probs.columns, cox_surv_probs.values)

plt.plot(times, surv_func(times), label="KM, km_cal = {}".format(km_km))
plt.plot(cox_surv_probs.mean(axis=0), label="Cox, km_cal = {}".format(km_cox))
plt.legend()
plt.title("KM vs Cox: marginal survival function")
plt.show()


# %%

cox_probs_recalibrated = recalibrate_survival_function(
    X_train,
    y_train,
    X_conf,
    times_cox,
    surv_probs=cox_surv_probs,
    surv_probs_conf=cox.predict_survival_function(X_test).T,
)
km_cox_recal = km_cal(y_train, times, cox_probs_recalibrated(times))

# %%
plt.plot(times, surv_func(times), label="KM, km_cal = {}".format(km_km))
plt.plot(cox_surv_probs.mean(axis=0), label="Cox, km_cal = {}".format(km_cox))
plt.plot(
    times,
    cox_probs_recalibrated(times).mean(axis=0),
    label="Cox recalibrated, km_cal = {}".format(km_cox_recal),
)
plt.legend()
plt.title("KM vs Cox: marginal survival function")
plt.show()

# %%
# Test if DeepHit is calibrated (hopefully not)

from sklearn_pandas import DataFrameMapper
import torchtuples as tt  # Some useful functions
from pycox.models import DeepHitSingle


def get_target(df):
    return (df["duration"].values, df["event"].values)


df = pd.DataFrame(X_train)
df = pd.concat([df, y_train], axis=1)
df = df.astype("float32")

df_train, df_val = train_test_split(df, test_size=0.2, random_state=0)


x_train = df_train.drop(columns=["duration", "event"])
x_val = df_val.drop(columns=["duration", "event"])
x_mapper = DataFrameMapper([(col, None) for col in x_train.columns])

x_train = x_mapper.fit_transform(df_train).astype("float32")
x_val = x_mapper.transform(df_val).astype("float32")


num_durations = 10
labtrans = DeepHitSingle.label_transform(num_durations)

y_train_ = labtrans.fit_transform(*get_target((df_train)))
y_val_ = labtrans.transform(*get_target(df_val))

train = (x_train, y_train_)
val = (x_val, y_val_)

# %%

in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

model = DeepHitSingle(
    net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts
)
model.fit(
    x_train,
    y_train_,
    batch_size=256,
    epochs=30,
    val_data=val,
    callbacks=[tt.callbacks.EarlyStopping()],
)

# %%
surv = model.predict_surv_df(x_val)
km_deephit = km_cal(y_train, surv.index, surv.values.T)

surv.mean(axis=1).plot(
    drawstyle="steps-post", label="DeepHit, km_cal = {}".format(km_deephit)
)
plt.plot(times, surv_func(times), label="KM, km_cal = {}".format(km_km))
plt.legend()
plt.ylabel("S(t | x)")
plt.xlabel("Time")

# %%
deephit_probs_recalibrated = recalibrate_survival_function(
    x_train,
    y_train,
    x_val,
    surv.index,
    surv_probs=surv.T,
    surv_probs_conf=model.predict_surv_df(x_val).T,
)
# %%
surv.mean(axis=1).plot(
    drawstyle="steps-post", label="DeepHit, km_cal = {}".format(km_deephit)
)
plt.plot(times, surv_func(times), label="KM, km_cal = {}".format(km_km))
plt.plot(
    times,
    deephit_probs_recalibrated(times).mean(axis=0),
    label="DeepHit recalibrated, km_cal = {}".format(
        km_cal(df_val[["duration", "event"]], times, deephit_probs_recalibrated(times))
    ),
)
plt.legend()
plt.ylabel("S(t | x)")
plt.xlabel("Time")

print(
    integrated_brier_score_survival(
        y_train, df_val[["duration", "event"]], surv.T.values, surv.index
    )
)
print(
    integrated_brier_score_survival(
        y_train,
        df_val[["duration", "event"]],
        deephit_probs_recalibrated(surv.index),
        surv.index,
    )
)
