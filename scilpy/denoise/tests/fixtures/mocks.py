# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture(scope='function')
def bilateral_filtering(mock_creator, expected_results):
    """
    Mock to patch the angle aware bilateral filtering function.
    Needs to be namespace patched by scripts.
    """
    def _mock_side_effect(*args, **kwargs):
        if expected_results is None or len(expected_results) == 0:
            return None

        return nib.load(expected_results).get_fdata(dtype=np.float32)

    return mock_creator("scilpy.denoise.bilateral_filtering",
                        "angle_aware_bilateral_filtering",
                        side_effect=_mock_side_effect)
