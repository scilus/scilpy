#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Functionnectome priors using *.trk files from a given folder.
Create a new folder containing all the probability maps derived from the input.
All the tractograms should be in the same space than the brain template.
The brain template defines which voxels are used to generate the probability map.
"""

import nibabel as nib
import numpy as np
import os
import glob
import argparse
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)

from dipy.io.streamline import load_tractogram
from dipy.tracking import utils
# from dipy.tracking.streamlinespeed import length
# from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamline import Streamlines
from scilpy.tracking.tools import filter_streamlines_by_length
# from multiprocessing import Pool  #TODO


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_dir_tractogram',
                   help='Path of the directory containing the input tractograms (.trk).')
    p.add_argument('in_template_file',
                   help='Path of the brain template file (.nii, nii.gz).')
    p.add_argument('out_dir',
                   help='Output directory for the priors.')

    # p.add_argument('--in_dir_region',  # TODO later
    #                help='Path to the directory containing the region masks.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_dir_tractogram,
                                 args.in_template_file,
                                 args.in_transfo], args.in_deformation)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    templ_i = nib.load(args.in_template_file).astype(bool)
    vox_list = np.argwhere(templ_i)
    base_mask = np.zeros(templ_i.shape, dtype=bool)

    trk_list = glob.glob(os.path.join(args.in_dir_tractogram, '*.trk'))
    for trk_F in trk_list:
        print(os.path.basename(trk_F))
        trk_sft = load_tractogram(trk_F, templ_i, bbox_valid_check=False)
        _ = trk_sft.remove_invalid_streamlines()  # Ideally, only remove the sections outsite of the boundingbox
        trk_sft = filter_streamlines_by_length(trk_sft, 25, 250)
        trk_sft.to_vox()
        for vox in vox_list:  # TODO Parallelize
            vox_mask = base_mask.copy()
            vox_mask[tuple(vox)] = True

            vox_strm = trk_sft.get_streamlines_copy()
            vox_strm = Streamlines(utils.target(vox_strm, templ_i.affine, vox_mask))
            vox_strm = vox_strm.get_streamlines_copy()

            vol_strm = np.zeros(templ_i.shape)
            for strm in vol_strm:
                arg_vox = tuple(np.round(strm).astype(int).T)
                vol_strm[arg_vox] = 1

            out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
            out_Ftmp = os.path.join(args.out_dir, out_name)

            if os.path.isfile(out_Ftmp):
                tmp_i = nib.load(out_Ftmp)
                tmp_vol = tmp_i.get_fdata()
                vol_strm += tmp_vol

            vol_str_i = nib.Nifti1Image(vol_strm, templ_i.affine)
            nib.save(vol_str_i, out_Ftmp)

    tmp_list = glob.glob(os.path.join(args.out_dir, '*tmp.nii.gz'))
    for tmp_F in tmp_list:  # TODO Parallelize
        tmp_i = nib.load(tmp_F)
        tmp_vol = tmp_i.get_fdata()
        out_vol = tmp_vol/len(trk_list)
        out_name = os.path.basename(tmp_F).replace('_tmp', '')
        outF = os.path.join(args.out_dir, out_name)
        out_i = nib.Nifti1Image(out_vol, templ_i.affine)
        nib.save(out_i, outF)
        os.remove(tmp_F)


if __name__ == "__main__":
    main()

# trk_in = nib.streamlines.load(trk_F)
# str_trkn = trk_in.streamlines
# vox_strn = Streamlines(utils.target(str_trkn, templ_i.affine, vox_mask))
