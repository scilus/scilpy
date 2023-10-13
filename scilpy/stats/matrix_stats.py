# -*- coding: utf-8 -*-
import numpy as np


def ttest_stat_only(x, y, tail):
    t = np.mean(x) - np.mean(y)
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                 * np.var(y, ddof=1)) / (n1 + n2 - 2))
    denom = s * np.sqrt(1 / n1 + 1 / n2)
    if denom == 0:
        return 0
    if tail == 'both':
        return np.abs(t / denom)
    if tail == 'left':
        return -t / denom
    else:
        return t / denom


def ttest_paired_stat_only(x, y, tail):
    n = len(x - y)
    sample_ss = np.sum((x - y)**2) - np.sum(x - y)**2 / n
    unbiased_std = np.sqrt(sample_ss / (n - 1))

    z = np.mean(x - y) / unbiased_std
    t = z * np.sqrt(n)
    if tail == 'both':
        return np.abs(t)
    if tail == 'left':
        return -t
    else:
        return t
