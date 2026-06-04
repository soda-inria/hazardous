from collections import namedtuple

import numpy as np
from scipy.stats import kstwo

KstestResult = namedtuple("KstestResult", ("statistic", "pvalue", "statistic_sign"))


def kstest_cdf(
    cdf1, cdf2, n_samples1, n_samples2, alternative="two-sided", method="auto"
):
    """
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
    """
    absolute_differences = np.abs(cdf1 - cdf2)

    ks_statistic = np.max(absolute_differences)
    d_location = np.argmax(absolute_differences)
    d_sign = np.sign(cdf1[d_location] - cdf2[d_location])
    en = np.round((n_samples1 * n_samples2) / (n_samples1 + n_samples2))
    prob = kstwo.sf(ks_statistic, en)
    prob = np.clip(prob, 0, 1)
    return KstestResult(ks_statistic, prob, statistic_sign=d_sign)
