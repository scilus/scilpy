# -*- coding: utf-8 -*-

import numpy as np


def nan_array_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    nan_a = np.argwhere(np.isnan(a))
    nan_b = np.argwhere(np.isnan(a))

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    return np.array_equal(a, b) and np.array_equal(nan_a, nan_b)
