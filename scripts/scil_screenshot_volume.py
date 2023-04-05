import argparse

from itertools import zip_longest
import nibabel as nib
import numpy as np
from os.path import splitext
from dipy.io.utils import is_header_compatible

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (
    axis_name_choices,
    add_overwrite_arg,
    assert_inputs_exist,
    assert_outputs_exist
)
from scilpy.image.utils import check_slice_indices
from scilpy.viz.scene_utils import (
    compose_mosaic, screenshot_slice, screenshot_contour
)


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Positional arguments
    p.add_argument("in_vol", help="Input volume image file.")
    p.add_argument(
        "out_fname",
        help="Name of the output image(s). If multiple slices are provided "
             "(or none), their index will be append to the name "
             "(e.g. volume.jpg, volume.png becomes "
             "volume_slice_0.jpg, volume_slice_0.png)."
    )

    # Optional arguments
    p.add_argument("--mask", help="Input mask image file for overlay.")
    p.add_argument("--in_labelmap",  help="Labelmap image.")

    p.add_argument('--mask_as_contour', action='store_true',
                   help='If supplied, the creates contour '
                        'from mask. [%(default)s].')
    p.add_argument(
        "--slice_ids", nargs="+", type=int, help="Slice indices to screenshot."
    )
    p.add_argument(
        "--axis_name",
        default="axial",
        type=str,
        choices=axis_name_choices,
        help="Name of the axis to visualize. [%(default)s]"
    )
    p.add_argument(
        "--win_dims",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(768, 768),
        type=int,
        help="The dimensions for the vtk window. [%(default)s]"
    )
    p.add_argument(
        "--vol_cmap_name",
        default=None,
        help="Colormap name for the volume image data. [%(default)s]"
    )
    p.add_argument(
        "--labelmap_cmap_name",
        default="viridis",
        help="Colormap name for the labelmap image data. [%(default)s]"
    )

    add_overwrite_arg(p)

    return p


def _parse_args(parser):

    args = parser.parse_args()

    inputs = []

    inputs.append(args.in_vol)

    if args.mask:
        inputs.append(args.mask)
    if args.in_labelmap:
        inputs.append(args.in_labelmap)

    assert_inputs_exist(parser, inputs)

    assert_same_resolution(inputs)

    return args


def _get_data_from_inputs(args):

    vol_img = nib.load(args.in_vol)

    mask_img = None
    if args.mask:
        mask_img = nib.load(args.mask)

    labelmap_img = None
    if args.in_labelmap:
        labelmap_img = nib.load(args.in_labelmap)

    return vol_img, mask_img, labelmap_img


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    vol_img, mask_img, labelmap_img = _get_data_from_inputs(args)

    # Check if the screenshots can be taken
    if args.slice_ids:
        check_slice_indices(vol_img, args.axis_name, args.slice_ids)
        slice_ids = args.slice_ids
    else:
        if args.axis_name == "axial":
            idx = 2
        elif args.axis_name == "coronal":
            idx = 1
        elif args.axis_name == "sagittal":
            idx = 0
        slice_ids = np.arange(vol_img.shape[idx])

        # Generate the images
    vol_scene_container = screenshot_slice(
        vol_img,
        args.axis_name,
        slice_ids,
        args.win_dims
    )

    mask_scene_container = []
    if mask_img is not None:
        if args.mask_as_contour:
            mask_scene_container = screenshot_contour(
                mask_img,
                args.axis_name,
                slice_ids,
                args.win_dims
            )
        else:
            mask_scene_container = screenshot_slice(
                mask_img,
                args.axis_name,
                slice_ids,
                args.win_dims
            )

    labelmap_scene_container = []
    if labelmap_img:
        labelmap_scene_container = screenshot_slice(
            labelmap_img,
            args.axis_name,
            slice_ids,
            args.win_dims,
        )

    name, ext = splitext(args.out_fname)
    names = ["{}_slice_{}{}".format(name, s, ext) for s in slice_ids]

    # Compose the mosaic
    for (volume, label, contour, name) in list(
        zip_longest(vol_scene_container,
                    labelmap_scene_container,
                    mask_scene_container,
                    names,
                    fillvalue=tuple())):

        img = compose_mosaic(
            [volume],
            [np.ones_like(volume) * 255],
            args.win_dims,
            1,
            1,
            vol_cmap_name=args.vol_cmap_name,
            labelmap_scene_container=[label],
            mask_contour_scene_container=[[contour]]
        )

        # Save the snapshot
        img.save(name)
