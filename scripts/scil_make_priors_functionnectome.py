#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Functionnectome priors using *.trk files from a given folder.
Create a new folder containing all the probability maps derived from the input.
All the tractograms should be in the same space than the brain template.
The brain template defines which voxels are used to generate the probability map.

For region-wise priors, the region masks must be provided as separate files,
with each file named as the region it contains (e.g. 'SupFront_Left.nii.gz')
"""

import nibabel as nib
import numpy as np
import os
import glob
import argparse
from multiprocessing import Pool
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.streamlines import hack_invalid_streamlines

from dipy.io.streamline import load_tractogram
# from dipy.tracking import utils
# from dipy.tracking.streamlinespeed import length
# from dipy.io.stateful_tractogram import StatefulTractogram
# from dipy.tracking.streamline import Streamlines
# from multiprocessing import shared_memory  # Require python 3.8+


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_dir_tractogram',
                   help='Path of the directory containing the input tractograms (.trk).')
    p.add_argument('in_template_file',
                   help='Path of the brain template file (.nii, nii.gz).')
    p.add_argument('out_dir',
                   help='Output directory for the priors.')

    p.add_argument('--in_reg_dir',
                   help='Path to the atlas file defining the regions.')
    p.add_argument('--parallel_proc',
                   help='Number of parallel processes to launch (Voxelwise priors only).')
    # p.add_argument('--from_endpoints', action='store_true',  # TODO
    #                help='Only streamlines with endpoints in the mask of the'
    #                     ' voxel/region are taken into account')
    add_overwrite_arg(p)

    return p


def visitation_mapping(area_mask, sft, out_Ftmp, affine):
    area_strm = filter_grid_roi(sft, area_mask, 'any', False)[0]

    # voxel = np.array(vox)
    # strm2 = [np.all(np.abs(strm-voxel)<0.5, 1).any() for strm in sft.streamlines]

    vol_strm = np.where(compute_tract_counts_map(area_strm.streamlines, area_mask.shape), 1, 0)
    # vox_strm = Streamlines(utils.target(trk_sft.streamlines, np.eye(4), vox_mask))
    # vol_strm = np.zeros(templ_i.shape)
    # for strm in vox_strm:
    #     arg_vox = tuple(np.round(strm).astype(int).T)
    #     vol_strm[arg_vox] = 1
    if os.path.isfile(out_Ftmp):
        tmp_i = nib.load(out_Ftmp)
        tmp_vol = tmp_i.get_fdata()
        vol_strm += tmp_vol

    vol_str_i = nib.Nifti1Image(vol_strm.astype('float32'), affine)
    nib.save(vol_str_i, out_Ftmp)
    return


def vox_multi(vox_batch, trk_sft, base_im, outdir):
    base_mask = np.zeros(base_im.shape, dtype=bool)
    for vox in vox_batch:
        vox_mask = base_mask.copy()
        vox_mask[tuple(vox)] = True
        out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
        out_Ftmp = os.path.join(outdir, out_name)
        visitation_mapping(vox_mask, trk_sft, out_Ftmp, base_im.affine)


def tmp2final(tmp_list, nb_trk, outdir, affine):
    for tmp_F in tmp_list:
        tmp_i = nib.load(tmp_F)
        tmp_vol = tmp_i.get_fdata()
        out_vol = tmp_vol/nb_trk
        out_name = os.path.basename(tmp_F).replace('_tmp', '')
        outF = os.path.join(outdir, out_name)
        out_i = nib.Nifti1Image(out_vol, affine)
        nib.save(out_i, outF)
        os.remove(tmp_F)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_template_file], args.in_reg_dir)
    if not os.path.isdir(args.in_dir_tractogram):
        parser.error(f'{args.in_dir_tractogram} is not a directory.')
    if not len(os.listdir(args.in_dir_tractogram)):
        parser.error(f'{args.in_dir_tractogram} is empty.')
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    templ_i = nib.load(args.in_template_file)
    # base_mask = np.zeros(templ_i.shape, dtype=bool)

    if args.in_reg_dir is None:
        templ_v = templ_i.get_fdata().astype(bool)
        vox_list = np.argwhere(templ_v)
    else:
        list_reg_files = glob.glob(os.path.join(args.in_reg_dir, '*.nii*'))

    if args.parallel_proc is not None:
        nb_proc = int(args.parallel_proc)
    else:
        nb_proc = 1

    trk_list = glob.glob(os.path.join(args.in_dir_tractogram, '*.trk'))
    for n, trk_F in enumerate(trk_list):
        print(os.path.basename(trk_F) + f' ({n+1}/{len(trk_list)})')
        trk_sft = load_tractogram(trk_F, templ_i, bbox_valid_check=False)
        trk_sft = filter_streamlines_by_length(trk_sft, 25, 250)
        trk_sft = hack_invalid_streamlines(trk_sft)
        trk_sft.to_vox()
        if args.in_reg_dir is None:
            if nb_proc == 1:
                vox_multi(vox_list, trk_sft, templ_i, args.out_dir)
                # for vox in vox_list:
                #     vox_mask = base_mask.copy()
                #     vox_mask[tuple(vox)] = True
                #     out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
                #     out_Ftmp = os.path.join(args.out_dir, out_name)
                #     visitation_mapping(vox_mask, trk_sft, out_Ftmp, templ_i.affine)
            else:  # Parallelize
                vox_batchs = np.array_split(vox_list, nb_proc)
                print('Starting parallel process')
                with Pool(processes=nb_proc) as pool:
                    pool.starmap(vox_multi, zip(vox_batchs,
                                                [trk_sft]*nb_proc,
                                                [templ_i]*nb_proc,
                                                [args.out_dir]*nb_proc))
        else:  # Regionwise priors
            for reg_F in list_reg_files:
                reg_i = nib.load(reg_F)
                reg_mask = reg_i.get_fdata().astype(bool)
                reg_fname = os.path.basename(reg_F)
                reg_name = reg_fname[:reg_fname.find('.nii')]
                out_name = f'probaMap_{reg_name}_tmp.nii.gz'
                out_Ftmp = os.path.join(args.out_dir, out_name)
                visitation_mapping(reg_mask, trk_sft, out_Ftmp, templ_i.affine)

    tmp_list = glob.glob(os.path.join(args.out_dir, '*tmp.nii.gz'))
    if (args.in_reg_dir is not None) or nb_proc == 1:
        tmp2final(tmp_list, len(trk_list), args.out_dir, templ_i.affine)
    else:  # Parallel proc
        tmpF_batch = np.array_split(tmp_list, nb_proc)
        with Pool(processes=nb_proc) as pool:
            pool.starmap(tmp2final, zip(tmpF_batch,
                                        [len(trk_list)]*nb_proc,
                                        [args.out_dir]*nb_proc,
                                        [templ_i.affine]*nb_proc,))


if __name__ == "__main__":
    main()
