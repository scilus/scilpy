#!/usr/bin/env python

"""
LabelSeg: automatic tract labelling without tractography.

This method takes as input fODF maps of order 6 (and a WM mask) and outputs 71
bundle label maps. These maps can then be used to perform tractometry/
tract profiling/radiomics. The bundle definitions follow TractSeg's minus the
whole CC.

This script requires PyTorch to be installed. To install it, see the official
website: https://pytorch.org/get-started/locally/

Parts of the implementation are based on or lifted from:
    SAM-Med3D: https://github.com/uni-medical/SAM-Med3D
    Multidimensional Positional Encoding: https://github.com/tatp22/multidim-positional-encoding

To cite: Antoine Théberge, Zineb El Yamani, François Rheault, Maxime Descoteaux, Pierre-Marc Jodoin (2025). LabelSeg. ISMRM Workshop on 40 Years of Diffusion: Past, Present & Future Perspectives, Kyoto, Japan.  # noqa
"""

import argparse
import nibabel as nib
import os

from argparse import RawTextHelpFormatter

from scilpy.io.utils import (
    assert_inputs_exist, assert_outputs_exist, add_overwrite_arg)
from scilpy.image.volume_operations import resample_volume

from scilpy.ml.labelseg.predict import predict
from scilpy.ml.labelseg.utils import get_data, get_model, download_weights
from scilpy.ml.utils import get_device


from dipy.utils.optpkg import optional_package

IMPORT_ERROR_MSG = "PyTorch is required to run this script. Please install" + \
                   " it first. See the official website for more info: " + \
                   "https://pytorch.org/get-started/locally/"  # noqa
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)

# TODO: Get bundle list from model
DEFAULT_BUNDLES = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'MLF_left', 'MLF_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'STR_left', 'STR_right', 'ST_FO_left', 'ST_FO_right', 'ST_OCC_left', 'ST_OCC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'T_OCC_left', 'T_OCC_right', 'T_PAR_left', 'T_PAR_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PREC_left', 'T_PREC_right', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'UF_left', 'UF_right']  # noqa E501

DEFAULT_CKPT = os.path.join('checkpoints', 'labelsegnet.ckpt')


def _build_arg_parser(parser):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('fodf', type=str,
                        help='fODF input')
    parser.add_argument('wm', type=str,
                        help='WM input')
    parser.add_argument('out_prefix', type=str,
                        help='Output destination and file prefix.')
    parser.add_argument('--mask', type=str, default=None,
                        help='Output destination and file prefix for '
                             'binary mask output.')
    parser.add_argument('--nb_labels', type=int, default=50)
    parser.add_argument('--checkpoint', type=str,
                        default=DEFAULT_CKPT,
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model. Default is '
                             '[%(default)s].')

    add_overwrite_arg(parser)

    return parser


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.fodf)
    assert_outputs_exist(parser, args, args.out_prefix,
                         [args.mask])

    # if file not exists
    if not os.path.exists(args.checkpoint):
        download_weights(args.checkpoint)

    fodf_in = nib.load(args.fodf)
    wm_in = nib.load(args.wm)

    img_size = 128  # TODO: get image size from model

    # TODO: infer sh order from model
    n_coefs = int((6 + 2) * (6 + 1) // 2)

    # Resampling volume to fit the model's input at training time
    resampled_fodf = resample_volume(fodf_in, ref_img=None,
                                     volume_shape=[img_size],
                                     iso_min=False,
                                     voxel_res=None,
                                     interp='lin',
                                     enforce_dimensions=False)

    resampled_wm = resample_volume(wm_in, ref_img=None,
                                   volume_shape=[img_size],
                                   iso_min=False,
                                   voxel_res=None,
                                   interp='nn',
                                   enforce_dimensions=False)

    # shape = fodf_in.get_fdata().shape[:3]
    fodf_data, wm_data = get_data(
        resampled_fodf, resampled_wm, img_size, n_coefs)

    # Load the model
    model = get_model(args.checkpoint, get_device())

    # Predict label maps. `predict` is a generator
    # yielding one label map per bundle (and a binary mask)
    for y_hat_mask, y_hat_label, b_name in predict(
        model, fodf_data, wm_data, args.nb_labels, n_coefs,
        DEFAULT_BUNDLES
    ):
        # Format the output as a nifti image
        label_img = nib.Nifti1Image(y_hat_label,
                                    resampled_fodf.affine,
                                    resampled_fodf.header)

        # Resample the image back to its original resolution
        label_img = resample_volume(label_img, ref_img=wm_in,
                                    # volume_shape=shape,
                                    iso_min=False,
                                    voxel_res=None,
                                    interp='nn',
                                    enforce_dimensions=False)
        # Save it
        nib.save(label_img, args.out_prefix + f'__{b_name}.nii.gz')

        # If the binary mask is also desired, perform the same
        # processing.
        if args.mask:
            mask_img = nib.Nifti1Image(y_hat_mask,
                                       resampled_wm.affine,
                                       resampled_wm.header)
            mask_img = resample_volume(mask_img, ref_img=wm_in,
                                       # volume_shape=shape,
                                       iso_min=False,
                                       voxel_res=None,
                                       interp='nn',
                                       enforce_dimensions=False)
            nib.save(label_img, args.out_prefix + f'__{b_name}.nii.gz')


if __name__ == "__main__":
    main()
