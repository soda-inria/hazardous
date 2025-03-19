import numpy as np
import pandas as pd

from hazardous.utils import check_y_survival


def d_calibration(
    fk,
    fk_infty,
    s_t,
    y_conf,
    event_of_interest="any",
):
    """
    Compute D-calibration for survival function.

    Parameters
    ----------
    fk : array-like, shape = (n_conf,)
        Incidence function for event k, each pb is computed such as
        F_k(y_i["duration]| x_i).
    fk_infty : array-like, shape = (n_conf,)
        Incidence function for event k at time infty.
    s_t : array-like, shape = (n_conf,)
        Survival function, each pb is computed such as S(y_i["duration]| x_i).
    y_conf : array-like, shape = (n_conf, 2)
        Conformal samples.
    n_buckets : int
        Number of buckets.

    Returns
    -------
    final_binning : DataFrame
        D-calibration values.
    """

    events, durations = check_y_survival(y_conf)
    buckets = np.linspace(0, 1, 101)

    if event_of_interest == "any":
        event_k = events > 0
    else:
        event_k = events == event_of_interest

    fk_t = fk[event_k]
    fk_infty_t = fk_infty[event_k]

    event_bins = np.digitize(fk_t / fk_infty_t, buckets, right=True)
    event_bins = np.clip(event_bins, 1, 100)
    event_bins = np.unique(event_bins, return_counts=True)

    event_binning = pd.DataFrame(index=range(1, 101))
    event_binning["count_event"] = 0
    event_binning.loc[event_bins[0], "count_event"] = event_bins[1]

    if sum(events == 0) == 0:
        return event_binning.set_index("buckets") / len(fk)

    fk_c = fk[events == 0]
    fk_infty_c = fk_infty[events == 0]
    s_c = s_t[events == 0]

    df = pd.DataFrame(fk_c / fk_infty_c, columns=["c"])
    df["fk_infty_c"] = fk_infty_c
    df["fk_c"] = fk_c
    df["s_c"] = s_c

    for buck in range(1, 100 + 1):
        li = buckets[buck - 1]
        li1 = buckets[buck]

        df[f"{buck}"] = 0.0
        df.loc[df["c"] <= li, f"{buck}"] = 0.01 * df["fk_infty_c"] / df["s_c"]
        df.loc[((df["c"] > li) & (df["c"] <= li1)), f"{buck}"] = (
            li1 * df["fk_infty_c"] - df["fk_c"]
        ) / df["s_c"]

    event_binning["censored_count"] = df.iloc[:, -100:].sum(axis=0).values

    final_binning = event_binning[["count_event", "censored_count"]].sum(axis=1)
    final_binning = pd.DataFrame(final_binning, columns=["count_event"]) / (
        sum(fk_infty)
    )
    return final_binning.cumsum(axis=0)
