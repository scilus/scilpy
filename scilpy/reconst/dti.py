# -*- coding: utf-8 -*-

import numpy as np

supported_tensor_formats = ['fsl', 'nifti', 'mrtrix', 'dipy']
tensor_format_description = \
    """
    Dipy's order is [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]
       Shape: [i, j , k, 6].
       Ref: https://github.com/dipy/dipy/blob/master/dipy/reconst/dti.py#L1639
    
    MRTRIX's order is : [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
       Shape: [i, j , k, 6].
       Ref: https://mrtrix.readthedocs.io/en/dev/reference/commands/dwi2tensor.html
    
    ANTS's order ('nifti format') is : [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz].
       Shape: [i, j , k, 1, 6] (Careful, file is 5D).
       Ref: https://github.com/ANTsX/ANTs/wiki/Importing-diffusion-tensor-data-from-other-software
    
    FSL's order is [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
       Shape: [i, j , k, 6].
       Ref: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide
       (Also used for the Fibernavigator)
    """


def convert_tensor_to_dipy_format(tensor, initial_format):
    """
    See description of formats at the top of this file.
    """
    assert initial_format in supported_tensor_formats, \
        "Tensor format not supported"

    if initial_format == 'nifti' or initial_format == 'dipy':
        correct_order = [0, 1, 2, 3, 4, 5]
        tensor = np.squeeze(tensor)
    elif initial_format == 'mrtrix':
        correct_order = [0, 3, 1, 4, 5, 2]
    else:  # initial_format == 'fsl':
        correct_order = [0, 1, 3, 2, 4, 5]

    return tensor[..., correct_order]


def convert_tensor_from_dipy_format(tensor, final_format):
    """
    See description of formats at the top of this file.
    """
    assert final_format in supported_tensor_formats, \
        "Tensor format not supported"

    if final_format == 'nifti' or final_format == 'dipy':
        correct_order = [0, 1, 2, 3, 4, 5]
    elif final_format == 'mrtrix':
        correct_order = [0, 2, 5, 1, 3, 4]
    else:  # final_format == 'fsl'.
        correct_order = [0, 1, 3, 2, 4, 5]

    tensor_reordered = tensor[..., correct_order]

    if final_format == 'nifti':
        # We need to add the fifth dimension
        tensor_reordered = tensor_reordered[:, :, :, None, :]

    return tensor_reordered


def convert_tensor_format(tensor, initial_format, final_format):
    """
    See description of formats at the top of this file.
    """
    tensor = convert_tensor_to_dipy_format(tensor, initial_format)
    return convert_tensor_from_dipy_format(tensor, final_format)

