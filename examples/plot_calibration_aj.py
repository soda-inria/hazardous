# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from hazardous._km_sampler import _AalenJohansenSampler
from hazardous.data._competing_weibull import make_synthetic_competing_weibull
from hazardous.calibration._aj_calibration import (
    aj_cal,
    recalibrate_incidence_functions,
)
from hazardous.metrics import (
    integrated_brier_score_incidence,
    integrated_brier_score_survival,
)

# %%
# KM-calibration without censoring
n_samples = 50000
n_events = 3

X, y = make_synthetic_competing_weibull(
    n_samples=n_samples,
    return_X_y=True,
    n_events=n_events,
    censoring_relative_scale=1.5,
    random_state=0,
)
sns.histplot(data=y, x="duration", hue="event", bins=50, kde=True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
X_train, X_conf, y_train, y_conf = train_test_split(
    X_train, y_train, random_state=0, test_size=0.5
)


# %%
aalen_sampler = _AalenJohansenSampler()
aalen_sampler.fit(y_train)

incidence_funcs_aj = aalen_sampler.incidence_func_
incidence_funcs_aj[0] = aalen_sampler.survival_func_


times = np.sort(y_train["duration"].unique())
incidences_probs = np.array(
    [
        np.repeat(incidence_funcs_aj[i](times), len(y_conf), axis=0)
        .reshape(len(times), len(y_conf))
        .T
        for i in range(n_events + 1)
    ]
)
incidences_probs = incidences_probs.swapaxes(0, 1)
# Test if AJ is AJ-calibrated


# %%
aj_cal_aj = aj_cal(y_conf, times, incidences_probs)

# %%
fig, ax = plt.subplots(ncols=n_events + 1, figsize=(15, 5))

for event_id in range(n_events + 1):
    inc_func = incidence_funcs_aj[event_id]
    inc_probs_AJ = inc_func(times)
    ax[event_id].plot(times, inc_probs_AJ, label="Event {}".format(event_id))
    ax[event_id].set_title("Incidence function for event {}".format(event_id))

plt.legend()
plt.show()


# %%
# Test if DeepHit is calibrated (hopefully not)

from sklearn_pandas import DataFrameMapper
import torchtuples as tt  # Some useful functions
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import torch


class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype("int64")


def get_target(df):
    return (df["duration"].values, df["event"].values)


df = pd.DataFrame(X_train)
df = pd.concat([df, y_train], axis=1)
df = df.astype("float32")

df_train, df_val = train_test_split(df, test_size=0.2, random_state=0)

x_mapper = DataFrameMapper(
    [(col, None) for col in df_train.drop(columns=["duration", "event"]).columns]
)

x_train = x_mapper.fit_transform(df_train).astype("float32")
x_val = x_mapper.transform(df_val).astype("float32")


num_durations = 10
labtrans = LabTransform(num_durations)
y_train_ = labtrans.fit_transform(*get_target(df_train))
y_val_ = labtrans.transform(*get_target(df_val))

train = (x_train, y_train_)
val = (x_val, y_val_)


# %%
class SimpleMLP(torch.nn.Module):
    """Simple network structure for competing risks."""

    def __init__(
        self,
        in_features,
        num_nodes,
        num_risks,
        out_features,
        batch_norm=True,
        dropout=None,
    ):
        super().__init__()
        self.num_risks = num_risks
        self.mlp = tt.practical.MLPVanilla(
            in_features,
            num_nodes,
            num_risks * out_features,
            batch_norm,
            dropout,
        )

    def forward(self, input):
        out = self.mlp(input)
        return out.view(out.size(0), self.num_risks, -1)


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """

    def __init__(
        self,
        in_features,
        num_nodes_shared,
        num_nodes_indiv,
        num_risks,
        out_features,
        batch_norm=True,
        dropout=None,
    ):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes_shared[:-1],
            num_nodes_shared[-1],
            batch_norm,
            dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1],
                num_nodes_indiv,
                out_features,
                batch_norm,
                dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out


in_features = x_train.shape[1]
num_nodes_shared = [64, 64]
num_nodes_indiv = [32]
num_risks = y_train_[1].max()
out_features = len(labtrans.cuts)
batch_norm = True
dropout = 0.1

# net = SimpleMLP(in_features, num_nodes_shared, num_risks, out_features)
net = CauseSpecificNet(
    in_features,
    num_nodes_shared,
    num_nodes_indiv,
    num_risks,
    out_features,
    batch_norm,
    dropout,
)

optimizer = tt.optim.AdamWR(
    lr=0.01, decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8
)
model = DeepHit(net, optimizer, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)

epochs = 512
batch_size = 256
callbacks = [tt.callbacks.EarlyStoppingCycle()]
verbose = False  # set to True if you want printout

model.fit(
    x_train,
    y_train_,
    batch_size=256,
    epochs=30,
    val_data=val,
    callbacks=callbacks,
)


def prepare_data(X, y):
    df_ = pd.DataFrame(X)
    df_ = pd.concat([df_, y], axis=1).astype("float32")
    x_ = x_mapper.transform(df_).astype("float32")
    y_ = labtrans.transform(*get_target(df_))
    return (x_, y_)


def predict_incidence_function(model, x):
    surv = model.predict_surv_df(x).values.T[:, None, :]
    cifs = model.predict_cif(x).swapaxes(0, 1).swapaxes(0, 2)
    return np.concatenate([surv, cifs], axis=1)


# %%
x_test, y_test_ = prepare_data(X_test, y_test)
x_conf, y_test_ = prepare_data(X_conf, y_conf)


incidences_probs_deephit_test = predict_incidence_function(model, x_test)
incidences_probs_deephit_conf = predict_incidence_function(model, x_conf)


aj_deephit = aj_cal(y_conf, labtrans.cuts, incidences_probs_deephit_test)

# %%
fig, ax = plt.subplots(ncols=n_events + 1, figsize=(15, 5))

for event_id in range(n_events + 1):
    inc_func = incidences_probs_deephit_test[:, event_id, :]
    inc_probs_AJ = incidence_funcs_aj[event_id](times)
    ax[event_id].plot(times, inc_probs_AJ, label="AJ")
    ax[event_id].plot(labtrans.cuts, inc_func.mean(axis=0), label="DeepHit")
    ax[event_id].set_title("Incidence function for event {}".format(event_id))

plt.legend()
plt.show()
# %%
deephit_probs_recalibrated = recalibrate_incidence_functions(
    X_conf,
    y_conf,
    times=labtrans.cuts,
    inc_probs=incidences_probs_deephit_test,
    inc_prob_conf=incidences_probs_deephit_conf,
    return_function=True,
)
# %%
fig, ax = plt.subplots(ncols=n_events + 1, figsize=(15, 5))

for event_id in range(n_events + 1):
    inc_func = incidences_probs_deephit_test[:, event_id, :]
    inc_probs_AJ = incidence_funcs_aj[event_id](times)
    inc_probs_deephit_recal = deephit_probs_recalibrated[:, event_id, :]
    ax[event_id].plot(times, inc_probs_AJ, label="AJ")
    ax[event_id].plot(labtrans.cuts, inc_func.mean(axis=0), label="DeepHit")
    ax[event_id].plot(
        labtrans.cuts,
        inc_probs_deephit_recal.mean(axis=0),
        label="DeepHit recalibrated",
    )
    ax[event_id].set_title("Incidence function for event {}".format(event_id))
plt.legend()
plt.show()
# %%
ibs = {}
ibs["deephit"] = {}
for event_id in range(n_events + 1):
    inc_func_deephit = incidences_probs_deephit_test[:, event_id, :]
    inc_probs_deephit_recal = deephit_probs_recalibrated[:, event_id, :]
    if event_id == 0:
        ibs_not_cal = integrated_brier_score_survival(
            y_train, y_test, inc_func_deephit, times=labtrans.cuts
        )

        ibs_recalibrated = integrated_brier_score_survival(
            y_train, y_test, inc_probs_deephit_recal, times=labtrans.cuts
        )

    else:
        ibs_not_cal = integrated_brier_score_incidence(
            y_train,
            y_test,
            inc_func_deephit,
            event_of_interest=event_id,
            times=labtrans.cuts,
        )
        ibs_recalibrated = integrated_brier_score_incidence(
            y_train,
            y_test,
            inc_probs_deephit_recal,
            event_of_interest=event_id,
            times=labtrans.cuts,
        )

    ibs["deephit"][event_id] = {
        "not_calibrated": ibs_not_cal,
        "recalibrated": ibs_recalibrated,
    }

# %%
# SurvivalBoost
from hazardous import SurvivalBoost

survboost = SurvivalBoost(n_iter=30)
survboost.fit(X_train, y_train)

survboost_probs = survboost.predict_cumulative_incidence(X_test, times=labtrans.cuts)

fig, ax = plt.subplots(ncols=n_events + 1, figsize=(15, 5))

for event_id in range(n_events + 1):
    inc_func = survboost_probs[:, event_id, :]
    inc_probs_AJ = incidence_funcs_aj[event_id](times)
    ax[event_id].plot(times, inc_probs_AJ, label="AJ")
    ax[event_id].plot(labtrans.cuts, inc_func.mean(axis=0), label="SB")
    ax[event_id].set_title("Incidence function for event {}".format(event_id))

plt.legend()
plt.show()
# %%
survboost_probs_recalibrated = recalibrate_incidence_functions(
    X_conf,
    y_conf,
    times=labtrans.cuts,
    inc_probs=survboost_probs,
    inc_prob_conf=survboost.predict_cumulative_incidence(X_conf, times=labtrans.cuts),
    return_function=True,
)
# %%
fig, ax = plt.subplots(ncols=n_events + 1, figsize=(15, 5))

for event_id in range(n_events + 1):
    inc_func = survboost_probs[:, event_id, :]
    inc_probs_AJ = incidence_funcs_aj[event_id](times)
    inc_probs_recal = survboost_probs_recalibrated[:, event_id, :]
    ax[event_id].plot(times, inc_probs_AJ, label="AJ")
    ax[event_id].plot(labtrans.cuts, inc_func.mean(axis=0), label="SurvivalBoost")
    ax[event_id].plot(
        labtrans.cuts,
        inc_probs_recal.mean(axis=0),
        label="SurvivalBoost recalibrated",
    )
    ax[event_id].set_title("Incidence function for event {}".format(event_id))
plt.legend()
plt.show()
# %%
ibs["survivalboost"] = {}
for event_id in range(n_events + 1):
    inc_func = survboost_probs[:, event_id, :]
    inc_probs_recal = survboost_probs_recalibrated[:, event_id, :]
    if event_id == 0:
        ibs_not_cal = integrated_brier_score_survival(
            y_train, y_test, inc_func, times=labtrans.cuts
        )

        ibs_recalibrated = integrated_brier_score_survival(
            y_train, y_test, inc_probs_recal, times=labtrans.cuts
        )

    else:
        ibs_not_cal = integrated_brier_score_incidence(
            y_train,
            y_test,
            inc_func,
            event_of_interest=event_id,
            times=labtrans.cuts,
        )
        ibs_recalibrated = integrated_brier_score_incidence(
            y_train,
            y_test,
            inc_probs_recal,
            event_of_interest=event_id,
            times=labtrans.cuts,
        )

    ibs["survivalboost"][event_id] = {
        "not_calibrated": ibs_not_cal,
        "recalibrated": ibs_recalibrated,
    }

# %%
