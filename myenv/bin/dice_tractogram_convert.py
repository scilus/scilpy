#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-
import argparse
import os
import sys

from dipy.io.stateful_tractogram import set_sft_logger_level
from dipy.io.streamline import load_tractogram, save_tractogram

set_sft_logger_level("CRITICAL")

DESCRIPTION = """Tractogram conversion from and to '.tck', '.trk', '.fib',
             '.vtk' and 'dpy'. All the extensions except '.trk, need a NIFTI
             file as reference """


def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_tractogram", help="Output tractogram")
    parser.add_argument(
        "--reference",
        "-r",
        action="store",
        help="Space attributes used as reference for the input tractogram")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwriting of the output")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser


def check_extension(in_arg, out_arg, ref_arg, parser):
    if not in_arg.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
        parser.error("Invalid input tractogram format")
    elif not out_arg.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
        parser.error("Invalid input tractogram format")
    elif ref_arg is not None and not ref_arg.endswith(('.nii', 'nii.gz')):
        parser.error("Invalid reference format")


def check_path(args, parser):
    in_file = args.input_tractogram
    out_file = args.output_tractogram
    in_ref = args.reference

    if not os.path.isfile(in_file):
        parser.error("No such file {}".format(in_file))
    if os.path.isfile(out_file) and not args.force:
        parser.error("Output tractogram already exists, use -f to overwrite")
    if in_ref is not None:
        if not os.path.isfile(args.reference):
            parser.error("No such file {}".format(args.reference))


def load_inp(in_arg, ref=None):
    if ref is None:
        in_reference = "same"
        try:
            inp_tract = load_tractogram(in_arg,
                                        reference=in_reference,
                                        bbox_valid_check=True,
                                        trk_header_check=True)
        except ValueError:
            raise
    else:
        in_reference = ref
        try:
            inp_tract = load_tractogram(in_arg,
                                        reference=in_reference,
                                        # "to_space=Space.RASMM,"
                                        # "to_origin=Origin.NIFTI,"
                                        bbox_valid_check=True)
        except ValueError:
            raise
    return inp_tract


def main():
    parser = input_parser()
    options = parser.parse_args()
    check_path(options, parser)
    check_extension(options.input_tractogram, options.output_tractogram,
                    options.reference, parser)

    if options.reference is None:
        if options.input_tractogram.endswith(".trk"):
            try:
                sft_in = load_inp(options.input_tractogram)
            except Exception:
                raise
        else:
            parser.error("reference is required if the input format is '.tck'")
    else:
        try:
            sft_in = load_inp(options.input_tractogram, ref=options.reference)
        except Exception:
            raise
    try:
        save_tractogram(sft_in, options.output_tractogram)
    except (OSError, TypeError) as e:
        parser.error("Output not valid: {}".format(e))


if __name__ == "__main__":
    main()
