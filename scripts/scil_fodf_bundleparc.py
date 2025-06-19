#!/usr/bin/env python

"""
BundleParc: automatic tract labelling without tractography.

This method takes as input fODF maps and outputs 71 bundle label maps. These maps can then be used to perform tractometry/tract profiling/radiomics. The bundle definitions follow TractSeg's minus the whole CC.

Inputs are presumed to come from Tractoflow and must be BET and cropped. fODFs must be of basis descoteaux07 and can be of order < 8 but accuracy may be reduced.

Model weights will be downloaded the first time the script is run, which will require an internet connection at runtime. Otherwise they can be manually downloaded from zenodo [1] and by specifying --checkpoint.

Example usage:
    $ scil_fodf_bundleparc.py fodf.nii.gz --out_prefix sub-001__

Example output:
    sub-001__AF_left.nii.gz, sub-001__AF_right.nii.gz, ..., sub-001__UF_right.nii.gz

The output can be further processed with scil_bundle_mean_std.py to compute statistics for each bundle.

The default value of 50 for --min_blob_size was found empirically on adult brains at a resolution of 1mm^3. The best value for your dataset may differ.

This script requires a GPU with ~6GB of available memory. If you use half-precision (float16) inference, you may be able to run it with ~3GB of GPU memory available. Otherwise, install the CPU version of PyTorch.

Parts of the implementation are based on or lifted from:
    SAM-Med3D: https://github.com/uni-medical/SAM-Med3D
    Multidimensional Positional Encoding: https://github.com/tatp22/multidim-positional-encoding

To cite: Antoine Théberge, Zineb El Yamani, François Rheault, Maxime Descoteaux, Pierre-Marc Jodoin (2025). LabelSeg. ISMRM Workshop on 40 Years of Diffusion: Past, Present & Future Perspectives, Kyoto, Japan.

[1]: https://zenodo.org/records/15579498
"""  # noqa

import argparse
import logging
import nibabel as nib
import numpy as np
import os

from argparse import RawTextHelpFormatter

from scilpy.io.utils import (
    assert_inputs_exist, assert_output_dirs_exist_and_empty,
    add_overwrite_arg, add_verbose_arg)
from scilpy.image.volume_operations import resample_volume

from scilpy.ml.bundleparc.predict import predict
from scilpy.ml.bundleparc.utils import DEFAULT_BUNDLES, \
    download_weights, get_model
from scilpy.ml.utils import get_device, IMPORT_ERROR_MSG
from scilpy import SCILPY_HOME


from dipy.utils.optpkg import optional_package
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)

DEFAULT_CKPT = os.path.join(SCILPY_HOME, 'checkpoints', 'bundleparc.ckpt')


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__ + '\n' + IMPORT_ERROR_MSG,
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('in_fodf',
                        help='fODF input.')
    parser.add_argument('--out_prefix', default='',
                        help='Output file prefix. Default is nothing. ')
    parser.add_argument('--out_folder', default='bundleparc',
                        help='Output destination. Default is [%(default)s].')
    parser.add_argument('--nb_pts', type=int, default=50,
                        help='Number of divisions per bundle. '
                             'Default is [%(default)s].')
    parser.add_argument('--min_blob_size', type=int, default=50,
                        help='Minimum blob size (in voxels) to keep. Smaller '
                             'blobs will be removed. Default is '
                             '[%(default)s].')
    parser.add_argument('--keep_biggest_blob', action='store_true',
                        help='If set, only keep the biggest blob predicted.')
    parser.add_argument('--half_precision', action='store_true',
                        help='Use half precision (float16) for inference. '
                             'This reduces memory usage but may lead to '
                             'reduced accuracy.')
    parser.add_argument('--bundles', choices=DEFAULT_BUNDLES, nargs='+',
                        default=DEFAULT_BUNDLES,
                        help='Bundles to predict. Default is every bundle.')
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT,
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

    assert_inputs_exist(parser, [args.in_fodf])
    assert_output_dirs_exist_and_empty(parser, args, args.out_folder,
                                       create_dir=True)

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if not os.path.exists(args.checkpoint):
        download_weights(args.checkpoint)

    device = get_device()
    # Load the model
    model = get_model(args.checkpoint, device, {'pretrained': True})

    fodf_in = nib.load(args.in_fodf)
    X, Y, Z, C = fodf_in.get_fdata(dtype=np.float32).shape

    # TODO in future release: infer these from model
    n_coefs = 45
    img_size = 128

    # Check the number of coefficients in the input fODF
    if C < n_coefs:
        logging.warning(f'Input fODFs have fewer than {n_coefs} coefficients. '
                        'Accuracy may be reduced.')
    if C > n_coefs:
        logging.warning(f'Input fODFs have more than {n_coefs} coefficients. '
                        f'Only the first {n_coefs} will be used.')

    # Resampling volume to fit the model's input at training time
    resampled_img = resample_volume(fodf_in, ref_img=None,
                                    volume_shape=[img_size],
                                    iso_min=False,
                                    voxel_res=None,
                                    interp='lin',
                                    enforce_dimensions=False)

    # Predict label maps. `predict` is a generator
    # yielding one label map per bundle and its name.
    for y_hat_label, b_name in predict(
        model, resampled_img.get_fdata(dtype=np.float32), n_coefs, args.nb_pts,
        args.bundles, args.min_blob_size, args.keep_biggest_blob,
        args.half_precision, logging.getLogger().getEffectiveLevel() <
        logging.WARNING
    ):

        # Format the output as a nifti image
        label_img = nib.Nifti1Image(y_hat_label,
                                    resampled_img.affine,
                                    resampled_img.header, dtype=np.uint16)

        # Resampling volume to fit the original image size
        resampled_label = resample_volume(label_img, ref_img=None,
                                          volume_shape=[X, Y, Z],
                                          iso_min=False,
                                          voxel_res=None,
                                          interp='nn',
                                          enforce_dimensions=False)
        # Save it
        nib.save(resampled_label, os.path.join(
            args.out_folder, f'{args.out_prefix}{b_name}.nii.gz'))


if __name__ == "__main__":
    main()
