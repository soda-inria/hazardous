import numpy as np
import pandas as pd


def d_calibration(
    fk,
    fk_infty,
    s_t,
    events,
    event_of_interest="any",
    n_buckets=10,
):
    """
    Compute D-calibration for survival function.

    Parameters
    ----------
    fk : array-like, shape = (n_samples,)
        Incidence function for event k, each pb is computed such as
        F_k(y_i["duration]| x_i).
    fk_infty : array-like, shape = (n_samples,)
        Incidence function for event k at time infty.
    s_t : array-like, shape = (n_samples,)
        Survival function, each pb is computed such as S(y_i["duration]| x_i).
    events : array-like, shape = (n_samples,)
        Event indicator.
    n_buckets : int
        Number of buckets.

    Returns
    -------
    final_binning : DataFrame
        D-calibration values.
    """

    buckets = np.linspace(0, 1, n_buckets + 1)
    bucket_length = 1 / n_buckets

    if event_of_interest == "any":
        event_k = events > 0
    else:
        event_k = events == event_of_interest

    fk_t = fk[event_k]
    fk_infty_t = fk_infty[event_k]

    event_bins = np.digitize(fk_t / fk_infty_t, buckets, right=True)
    event_bins = np.clip(event_bins, 1, n_buckets)

    event_binning = pd.DataFrame(
        np.unique(event_bins, return_counts=True), index=["buckets", "count_event"]
    ).T

    if sum(events == 0) == 0:
        return event_binning.set_index("buckets") / len(fk)

    fk_c = fk[events == 0]
    fk_infty_c = fk_infty[events == 0]
    s_c = s_t[events == 0]

    df = pd.DataFrame(fk_c / fk_infty_c, columns=["c"])
    df["fk_infty_c"] = fk_infty_c
    df["fk_c"] = fk_c
    df["s_c"] = s_c

    for buck in range(1, n_buckets + 1):
        li = buckets[buck - 1]
        li1 = buckets[buck]

        df[f"{buck}"] = 0.0
        df.loc[df["c"] <= li, f"{buck}"] = bucket_length * df["fk_infty_c"] / df["s_c"]
        df.loc[((df["c"] > li) & (df["c"] <= li1)), f"{buck}"] = (
            li1 * df["fk_infty_c"] - df["fk_c"]
        ) / df["s_c"]

    event_binning["censored_count"] = df.iloc[:, -10:].sum(axis=0).values

    event_binning.set_index("buckets", inplace=True)
    final_binning = event_binning[["count_event", "censored_count"]].sum(axis=1)
    return pd.DataFrame(final_binning, columns=["count_event"]) / (sum(fk_infty))
