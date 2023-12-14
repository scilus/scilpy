
import nibabel as nib
import numpy as np
import pytest


@pytest.fixture(scope='function')
def bilateral_filtering(mock_creator, expected_results):
    def _mock_side_effect(*args, **kwargs):
        if expected_results is None or len(expected_results) == 0:
            return None

        return nib.load(expected_results).get_fdata(dtype=np.float32)

    return mock_creator("scilpy.denoise.bilateral_filtering",
                       "angle_aware_bilateral_filtering",
                       side_effect=_mock_side_effect)


@pytest.fixture(scope='function')
def bilateral_filtering_script(mock_creator, expected_results):
    def _mock_side_effect(*args, **kwargs):
        if expected_results is None or len(expected_results) == 0:
            return None

        return nib.load(expected_results).get_fdata(dtype=np.float32)

    return mock_creator("scripts.scil_execute_angle_aware_bilateral_filtering",
                       "angle_aware_bilateral_filtering",
                       side_effect=_mock_side_effect)