#!/usr/bin/env python

"""
BundleParc: automatic tract labelling without tractography.

This method takes as input fODF maps and outputs 71 bundle label maps.
These maps can then be used to perform tractometry/tract profiling/radiomics.
The bundle definitions follow TractSeg's minus the whole CC.

**IMPORTANT**: fODF inputs must be BET and cropped, in SH format, and computed
with scilpy >= 3.0 (SH in world space). Any voxel order is accepted. Use
--sh_basis to give the SH basis of the input. fODFs can be of order < 8 but
accuracy may be reduced.

Before inference, the fODF is converted to the same format as the training data
(scilpy < 3.0 fODFs from Tractoflow): voxel order LAS (stride -1,2,3,4), SH in
the voxel space of that grid but mirrored in x, and descoteaux07_legacy basis.
The output labels are saved in the same voxel order as the input.

**IMPORTANT**: The image is expected to be roughly aligned with the scanner
axes (non-oblique). Predictions degrade with the angle between the voxel and
scanner axes. Registering the image to remove this tilt is left to the user.

Model weights will be downloaded the first time the script is run, which will
require an internet connection at runtime. Otherwise they can be manually
downloaded from zenodo [1] and by specifying --checkpoint.

Example usage:
    $ scil_fodf_bundleparc fodf.nii.gz --out_prefix sub-001__

Example output:
    sub-001__AF_left.nii.gz, sub-001__AF_right.nii.gz, ..., sub-001__UF_right.nii.gz

The output can be further processed with scil_bundle_mean_std to compute
statistics for each bundle.

The default value of 50 for --min_blob_size was found empirically on adult
brains at a resolution of 1mm^3. The best value for your dataset may differ.

This script requires a GPU with ~8GB of available memory. If you use
half-precision (float16) inference, you may be able to run it with ~4GB of GPU
memory available. Otherwise, install the CPU version of PyTorch. Execution on
MacOS is not supported for now.

Parts of the implementation are based on or lifted from:
    SAM-Med3D: https://github.com/uni-medical/SAM-Med3D
    Multidimensional Positional Encoding: https://github.com/tatp22/multidim-positional-encoding

To cite: 
    Antoine Théberge, Zineb El Yamani, François Rheault, Maxime Descoteaux,
    Pierre-Marc Jodoin (2025). LabelSeg. ISMRM Workshop on 40 Years of Diffusion:
    Past, Present & Future Perspectives, Kyoto, Japan.

[1]: Descoteaux, M., Deriche, R., Knösche, T. R., & Anwander, A. (2007).
    Deterministic and probabilistic tractography based on complex fibre
    orientation distributions.
    IEEE Transactions on Medical Imaging, 26(11), 1464-1477.
[2]: https://zenodo.org/records/19634429
"""  # noqa

import argparse
import logging
import nibabel as nib
import numpy as np
import os

from argparse import RawTextHelpFormatter
from functools import partial

from dipy.data import get_sphere

from scilpy.io.stateful_image import StatefulImage
from scilpy.io.utils import (
    assert_inputs_exist, assert_output_dirs_exist_and_empty,
    add_overwrite_arg, add_sh_basis_args, add_verbose_arg,
    parse_sh_basis_arg)
from scilpy.image.volume_operations import resample_volume
from scilpy.reconst.sh import convert_sh_basis, rotate_sh

from scilpy.ml.bundleparc.bundles import DEFAULT_BUNDLES
from scilpy.ml.bundleparc.labels import post_process_labels_discrete, \
    post_process_labels_mm, post_process_labels_continuous
from scilpy.ml.utils import IMPORT_ERROR_MSG
from scilpy import SCILPY_HOME


from dipy.utils.optpkg import optional_package
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)

DEFAULT_CKPT = os.path.join(SCILPY_HOME, 'checkpoints', 'bundleparc.ckpt')


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__ + '\n' + IMPORT_ERROR_MSG,
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('in_fodf',
                        help='Input fODF volume in nifti format '
                             '(SH in world space, any voxel order).')
    parser.add_argument('--out_prefix', default='',
                        help='Output file prefix. Default is nothing. ')
    parser.add_argument('--out_dir', default='bundleparc',
                        help='Output destination. Default is [%(default)s].')
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
    parser.add_argument('--volume_size', default=144, type=int,
                        help='Size of volume to resample to for inference. '
                             'Only modify if you know what you are doing.')
    parcel_group = parser.add_mutually_exclusive_group()
    parcel_group.add_argument('--nb_pts', type=int, default=10,
                              help='Number of divisions per bundle. Default is'
                                   ' [%(default)s].')
    parcel_group.add_argument('--mm', type=float,
                              help='If set, bundles will be split in sections '
                                   'roughly X mm wide.')
    parcel_group.add_argument('--continuous', action='store_true',
                              help='If set, the output label maps will be '
                                   'continuous ∈ [0, 1].')
    blob_group = parser.add_mutually_exclusive_group()
    blob_group.add_argument('--min_blob_size', type=int, default=50,
                            help='Minimum blob size (in voxels) to keep. '
                                 'Smaller blobs will be removed. Default is '
                                 '[%(default)s].')
    blob_group.add_argument('--keep_biggest_blob', action='store_true',
                            help='Only keep the biggest blob predicted.')

    add_sh_basis_args(parser)
    add_overwrite_arg(parser)
    add_verbose_arg(parser)

    return parser


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_fodf], [])
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Only the header is read here, the data is loaded below.
    fodf_in = nib.load(args.in_fodf)
    if len(fodf_in.shape) != 4:
        parser.error(
            f"Input fODF volume must be 4D (got {len(fodf_in.shape)}D).")

    # Angle (degrees) between each voxel axis and the closest scanner axis.
    obliquity = np.degrees(nib.affines.obliquity(fodf_in.affine)).max()
    if obliquity > 10:
        logging.warning(
            f"Input fODF is oblique ({obliquity:.1f} degrees between the "
            f"voxel and scanner axes). Predictions degrade with obliquity; "
            f"consider registering the image to remove the tilt first.")

    if not have_torch:
        parser.error(IMPORT_ERROR_MSG)

    # Imported here so that --help and input validation work without
    # PyTorch: these modules use torch at import time.
    from scilpy.ml.bundleparc.predict import predict
    from scilpy.ml.bundleparc.utils import download_weights, get_model
    from scilpy.ml.utils import get_device

    if not os.path.exists(args.checkpoint):
        download_weights(args.checkpoint)

    device = get_device()
    # Load the model
    model = get_model(args.checkpoint, device, {'pretrained': True})

    # The model was trained on LAS fODFs. The voxel order changes here, the
    # SH are moved to the frame of the training data below.
    fodf_simg = StatefulImage.load(args.in_fodf, to_orientation='LAS')
    X, Y, Z, C = fodf_simg.shape

    # TODO in future release: infer these from model
    n_coefs = 45

    # Check the number of coefficients in the input fODF
    if C < n_coefs:
        logging.warning(f'Input fODFs have fewer than {n_coefs} coefficients. '
                        'Accuracy may be reduced.')
    if C > n_coefs:
        logging.warning(f'Input fODFs have more than {n_coefs} coefficients. '
                        f'Only the first {n_coefs} will be used.')

    # The model was trained on descoteaux07_legacy coefficients.
    fodf_data = fodf_simg.get_fdata(dtype=np.float32)
    sh_basis, is_legacy = parse_sh_basis_arg(args)
    if sh_basis != 'descoteaux07' or not is_legacy:
        fodf_data = convert_sh_basis(
            fodf_data, get_sphere(name='repulsion724').subdivide(n=1),
            mask=np.any(fodf_data, axis=-1),
            input_basis=sh_basis, output_basis='descoteaux07',
            is_input_legacy=is_legacy, is_output_legacy=True,
            nbr_processes=1)

    # The training fODFs were fitted in the voxel space of the LAS grid, but
    # mirrored in x. Move the SH from world space to voxel space (removes the
    # tilt of oblique images), then mirror x, in a single rotation.
    to_voxel = StatefulImage._get_rotation_matrix(fodf_simg.affine).T
    fodf_data = rotate_sh(fodf_data, np.diag([-1., 1., 1.]) @ to_voxel,
                          basis_type='descoteaux07', is_legacy=True,
                          nbr_processes=1)
    fodf_simg = StatefulImage.create_from(
        nib.Nifti1Image(fodf_data, fodf_simg.affine), fodf_simg)

    # Resampling volume to fit the model's input at training time
    resampled_img = resample_volume(fodf_simg, ref_img=None,
                                    volume_shape=[args.volume_size],
                                    iso_min=False,
                                    voxel_res=None,
                                    interp='lin',
                                    enforce_dimensions=False)

    # Get the voxel size of the input fODF after resampling
    # Presuming isotropic resampling
    voxel_size = np.mean(resampled_img.header.get_zooms()[:3])

    # Get the label function to use for post-processing
    if args.continuous:
        label_function = post_process_labels_continuous
    elif args.mm is not None:
        label_function = partial(post_process_labels_mm, args.mm, voxel_size)
    else:
        label_function = partial(post_process_labels_discrete,
                                 args.nb_pts)

    # Predict label maps. `predict` is a generator
    # yielding one label map per bundle and its name.
    for y_hat_label, b_name in predict(
        model,
        resampled_img.get_fdata(dtype=np.float32),
        n_coefs,
        label_function,
        args.bundles,
        args.min_blob_size,
        args.keep_biggest_blob,
        half_precision=args.half_precision,
        verbose=logging.getLogger().getEffectiveLevel() < logging.WARNING
    ):
        # Format the output as a nifti image. Set dtype explicitly: the
        # label functions return uint8/uint16 labels, but resampled_img's
        # header still has the input fODF's float dtype.
        label_img = nib.Nifti1Image(y_hat_label,
                                    resampled_img.affine,
                                    header=resampled_img.header,
                                    dtype=y_hat_label.dtype)
        # Keeps the voxel order of the input, used when saving.
        label_simg = StatefulImage.create_from(label_img, fodf_simg)

        # Resampling volume to fit the original image size
        resampled_label = resample_volume(label_simg, ref_img=None,
                                          volume_shape=[X, Y, Z],
                                          iso_min=False,
                                          voxel_res=None,
                                          interp='nn',
                                          enforce_dimensions=False)
        # Save it, back in the voxel order of the input.
        resampled_label.save(os.path.join(
            args.out_dir, f'{args.out_prefix}{b_name}.nii.gz'))


if __name__ == "__main__":
    main()
