# -*- coding: utf-8 -*-
import numpy as np

from scilpy.connectivity.matrix_tools import apply_reordering


def test_compute_olo():
    pass


def test_apply_olo():
    pass


def test_apply_reordering():
    conn_matrix = np.asarray([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]])
    output = apply_reordering(conn_matrix, [[0, 1, 3, 2],
                                            [1, 2, 3, 0]])
    # First changing rows 2 and 3
    expected_out = np.asarray([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [13, 14, 15, 16],
                              [9, 10, 11, 12]])
    # Permuting columns
    expected_out = np.asarray([[2, 3, 4, 1],
                              [6, 7, 8, 5],
                              [14, 15, 16, 13],
                              [10, 11, 12, 9]])
    assert np.array_equal(output, expected_out)
    

def test_evaluate_graph_measures():
    pass


def test_normalize_matrix_from_values():
    pass


def test_normalize_matrix_from_parcel():
    pass
