import numpy as np
import pandas as pd


def d_calibration(
    inc_prob_t,
    inc_prob_infty,
    n_buckets,
    inc_prob_t_censor=None,
    inc_prob_infty_censor=None,
):
    buckets = np.linspace(0, 1, n_buckets + 1)
    event_bins = np.digitize(inc_prob_t / inc_prob_infty, buckets, right=True)
    event_bins = np.clip(event_bins, 1, n_buckets)
    event_binning = pd.DataFrame(
        np.unique(event_bins, return_counts=True), index=["buckets", "count_event"]
    ).T
    if inc_prob_t_censor is None:
        return event_binning.set_index("buckets") / len(inc_prob_t)

    df = pd.DataFrame(inc_prob_t_censor / inc_prob_infty_censor, columns=["c"])
    for buck in range(1, n_buckets + 1):
        li = buckets[buck - 1]
        li1 = buckets[buck]
        df[f"{buck}"] = 0.0
        df.loc[df["c"] <= li, f"{buck}"] = li1 - li
        df.loc[((df["c"] > li) & (df["c"] <= li1)), f"{buck}"] = li1 - df["c"]
        df[f"{buck}"] /= 1 - df["c"]

    event_binning["censored_count"] = df.iloc[:, 1:].sum(axis=0).values

    event_binning.set_index("buckets", inplace=True)
    final_binning = event_binning[["count_event", "censored_count"]].sum(axis=1)
    return pd.DataFrame(final_binning, columns=["count_event"]) / (
        len(inc_prob_t) + len(inc_prob_t_censor)
    )
