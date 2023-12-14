
import nibabel as nib
import numpy as np
import pytest


@pytest.fixture(scope='function')
def bilateral_filtering(mock_creator, expected_results=None):
    def _mock_side_effect(*args, **kwargs):
        if expected_results is None or len(expected_results) == 0:
            return None

        _out_odf_fname = expected_results[0]
        return nib.load(_out_odf_fname).get_fdata(dtype=np.float32)

    return mock_creator("scripts.scil_execute_angle_aware_bilateral_filtering",
                       "angle_aware_bilateral_filtering",
                       side_effect=_mock_side_effect)
