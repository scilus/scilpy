#!/usr/bin/env python

"""
BundleParc: automatic tract labelling without tractography.

This method takes as input fODF maps of order 6 (or above) and a whole-brain WM mask and outputs 71 bundle label maps. These maps can then be used to perform tractometry/tract profiling/radiomics. The bundle definitions follow TractSeg's minus the whole CC.

Example usage:
    $ scil_fodf_bundleparc.py fodf.nii.gz wm_mask.nii.gz --out_prefix sub-001__

Example output:
    sub-001__AF_left.nii.gz, sub-001__AF_right.nii.gz, ..., sub-001__UF_right.nii.gz

The output can be further processed with scil_bundle_mean_std.py to compute statistics for each bundle.

This script requires PyTorch to be installed. To install it, see the official website: https://pytorch.org/get-started/locally/

This script requires a GPU with ~6GB of available memory. If you use
half-precision (float16) inference, you may be able to run it with ~3GB of
GPU memory available. Otherwise, install the CPU version of PyTorch.

Parts of the implementation are based on or lifted from:
    SAM-Med3D: https://github.com/uni-medical/SAM-Med3D
    Multidimensional Positional Encoding: https://github.com/tatp22/multidim-positional-encoding

To cite: Antoine Théberge, Zineb El Yamani, François Rheault, Maxime Descoteaux, Pierre-Marc Jodoin (2025). LabelSeg. ISMRM Workshop on 40 Years of Diffusion: Past, Present & Future Perspectives, Kyoto, Japan.  # noqa
"""

import argparse
import logging
import nibabel as nib
import os

from argparse import RawTextHelpFormatter
from dipy.utils.optpkg import optional_package

from scilpy.io.utils import (
    assert_inputs_exist, assert_outputs_exist, add_overwrite_arg,
    add_verbose_arg)
from scilpy.image.volume_operations import resample_volume

from scilpy.ml.bundleparc.predict import predict
from scilpy.ml.bundleparc.utils import get_data, get_model, download_weights
from scilpy.ml.utils import get_device, IMPORT_ERROR_MSG
from scilpy import SCILPY_HOME


torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)

# TODO: Get bundle list from model
DEFAULT_BUNDLES = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP', 'MLF_left', 'MLF_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_III_left', 'SLF_III_right', 'SLF_II_left', 'SLF_II_right', 'SLF_I_left', 'SLF_I_right', 'STR_left', 'STR_right', 'ST_FO_left', 'ST_FO_right', 'ST_OCC_left', 'ST_OCC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'T_OCC_left', 'T_OCC_right', 'T_PAR_left', 'T_PAR_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PREC_left', 'T_PREC_right', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'UF_left', 'UF_right']  # noqa E501

DEFAULT_CKPT = os.path.join(SCILPY_HOME, 'checkpoints', 'bundleparc.ckpt')


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('in_fodf', type=str,
                        help='fODF input of order 6 or above. If the input '
                             'has more than 28 coefficients, only the first 28'
                             ' will be used.')
    parser.add_argument('in_wm', type=str,
                        help='Whole-brain WM mask input.')
    parser.add_argument('--out_prefix', type=str, default='',
                        help='Output file prefix. Default is nothing. ')
    parser.add_argument('--out_folder', type=str, default='',
                        help='Output destination. Default is the current '
                             'directory.')
    parser.add_argument('--nb_pts', type=int, default=50,
                        help='Number of divisions per bundle. '
                             'Default is [%(default)s].')
    parser.add_argument('--half_precision', action='store_true',
                        help='Use half precision (float16) for inference. '
                             'This reduces memory usage but may lead to '
                             'reduced accuracy.')
    parser.add_argument('--checkpoint', type=str,
                        default=DEFAULT_CKPT,
                        help='Checkpoint (.ckpt) containing hyperparameters '
                             'and weights of model. Default is '
                             '[%(default)s]. If the file does not exist, it '
                             'will be downloaded.')
    add_overwrite_arg(parser)
    add_verbose_arg(parser)

    return parser


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_fodf, args.in_wm])
    assert_outputs_exist(parser, args, args.out_prefix)

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # If the checkpoint does not exist, download it
    if not os.path.exists(args.checkpoint):
        download_weights(args.checkpoint, verbose=args.verbose)

    fodf_in = nib.load(args.in_fodf)
    wm_in = nib.load(args.in_wm)

    img_size = 128  # TODO in future release: get image size from model

    # TODO in future release: infer sh order from model
    n_coefs = 28
    actual_coefs = fodf_in.get_fdata().shape[-1]
    assert actual_coefs >= n_coefs, \
        f'Input fODFs should have at least {n_coefs} coefficients.'

    if actual_coefs > n_coefs:
        logging.warning('Input fODFs have more than 28 coefficients. '
                        f'Only the first {n_coefs} will be used.')

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
    # Hack so that get_data_as_mask works
    resampled_wm.set_filename(wm_in.get_filename())

    # Get data in the right format
    fodf_data, wm_data = get_data(
        resampled_fodf, resampled_wm, img_size, n_coefs)

    # Load the model
    model = get_model(args.checkpoint, get_device())

    # Predict label maps. `predict` is a generator
    # yielding one label map per bundle (and a binary mask)
    for y_hat_label, b_name in predict(
        model, fodf_data, wm_data, args.nb_pts,
        DEFAULT_BUNDLES, half_precision=args.half_precision,
        verbose=args.verbose
    ):
        # Format the output as a nifti image
        label_img = nib.Nifti1Image(y_hat_label,
                                    resampled_fodf.affine,
                                    resampled_fodf.header)

        # Resample the image back to its original resolution
        label_img = resample_volume(label_img, ref_img=wm_in,
                                    iso_min=False,
                                    voxel_res=None,
                                    interp='nn',
                                    enforce_dimensions=False)
        # Save it
        nib.save(label_img, os.path.join(
            args.out_folder, args.out_prefix + f'{b_name}.nii.gz'))


if __name__ == "__main__":
    main()
