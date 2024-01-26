# -*- coding: utf-8 -*-

import pytest


@pytest.fixture(scope='function')
def amico_evaluator(mock_creator):
    """
    Mock to patch amico's kernel generation and fitting.
    Does not need to be namespace patched by scripts.
    """
    return mock_creator("amico", "Evaluation",
                        mock_attributes=["fit", "generate_kernels",
                                         "load_kernels", "save_results"])
