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
import time
from itertools import compress
from multiprocessing import Pool
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.streamlines import hack_invalid_streamlines

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
# from dipy.tracking import utils
# from dipy.tracking.streamlinespeed import length
# from dipy.tracking.streamline import Streamlines
# from multiprocessing import shared_memory  # Require python 3.8+


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_dir_tractogram',
                   help='Path of the directory containing the input '
                   '\ntractograms (.trk).')
    p.add_argument('in_template_file',
                   help='Path of the brain template file (.nii, nii.gz).')
    p.add_argument('out_dir',
                   help='Output directory for the priors.')

    p.add_argument('--in_reg_dir',
                   help='Path to the atlas file defining the regions.')
    p.add_argument('--parallel_proc',
                   help='Number of parallel processes to launch '
                   '\n(Voxelwise priors only).')
    # p.add_argument('--from_endpoints', action='store_true',  # TODO
    #                help='Only streamlines with endpoints in the mask of the'
    #                     ' voxel/region are taken into account')
    add_overwrite_arg(p)

    return p


def save_tmp_map(vol, file, affine, emptyVol=False):
    if os.path.isfile(file) and not emptyVol:
        vol = vol.astype('float32')
        tmp_i = nib.load(file)
        tmp_vol = tmp_i.get_fdata()
        vol += tmp_vol
    vol_str_i = nib.Nifti1Image(vol.astype('float32'), affine)
    nib.save(vol_str_i, file)


def voxelize_tractogram(sft):
    '''
    Rounds all the points of the streamlines to the voxel they belong to and
    removes duplicate points
    '''
    sft.to_vox()
    sft.to_corner()
    voxed_strms = []
    for strm in sft.streamlines:
        strm_r = np.rint(strm).astype(np.int32)
        _, ind = np.unique(strm_r, return_index=True, axis=0)
        voxed_strms.append(strm_r[np.sort(ind)])
    return StatefulTractogram.from_sft(voxed_strms, sft)


def tupled_streamlines(sft):
    '''
    sft must be a voxelised sft (cf. voxelize_tractogram)
    Returns a list of streamlines with each point given as a tuple
    '''
    return [list(zip(st.T[0], st.T[1], st.T[2])) for st in sft.streamlines]


def voxeled_visiation_mapping(area_mask, streamlines, out_Ftmp, affine, tupled):
    ''' sft must be a voxelised sft (cf. voxelize_tractogram)'''
    # 1
    ind_vox = np.argwhere(area_mask)
    if tupled:
        ind_vox = [tuple(ind) for ind in ind_vox]
        indstrm = [any(ind in strm for ind in ind_vox) for strm in streamlines]
        area_strm = list(compress(streamlines, indstrm))
        # 2
        vol_strm = np.zeros(area_mask.shape, dtype=np.float32)
        for strm in area_strm:
            for v in strm:
                vol_strm[v] = 1
    else:
        indstrm = [any(any(np.equal(vox, strm).all(1)) for vox in ind_vox) for strm in streamlines]
        area_strm = streamlines[indstrm]
        # 2
        vol_strm = np.zeros(area_mask.shape, dtype=np.float32)
        for strm in area_strm:
            vol_strm[tuple(strm.T)] = 1
    # 3
    save_tmp_map(vol_strm, out_Ftmp, affine)


def visitation_mapping(area_mask, sft, out_Ftmp, affine):
    # 1
    area_strm, _ = filter_grid_roi(sft, area_mask, 'any', False)
    # 2
    vol_strm = np.where(compute_tract_counts_map(area_strm.streamlines, area_mask.shape), 1, 0)
    # 3
    save_tmp_map(vol_strm, out_Ftmp, affine)


def vox_multi(vox_batch, sft, base_im, outdir, voxeled=True, tupled=True):
    """
    Iterates over the voxels, creating the appropriate mask and file-name, and
    runs visitation_mapping()

    """
    if tupled and not voxeled:
        raise ValueError('The tractogram is not voxeled, but shoud be for tupling')
    base_mask = np.zeros(base_im.shape, dtype=bool)
    if tupled:
        streamlines = tupled_streamlines(sft)
    else:
        streamlines = sft.streamlines
    for vox in vox_batch:
        vox_mask = base_mask.copy()
        vox_mask[tuple(vox)] = True
        out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
        out_Ftmp = os.path.join(outdir, out_name)
        if voxeled:
            voxeled_visiation_mapping(vox_mask, streamlines, out_Ftmp, base_im.affine, tupled)
        else:
            visitation_mapping(vox_mask, sft, out_Ftmp, base_im.affine)


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


def Parcelate_tractogram(sft, templ_v, nb_parcel=16):
    '''
    Divide the input tractogram into sub-tractograms by parcelating the input
    templ_v into similarly sized sub volumes and filtering the tractogram by
    these subvolumes. Might not be well adapted for GM masks in input (for
    those, chose nb_parcel=8).
    (Different sub-tractograms can share streamlines)

    Parameters
    ----------
    sft : Stateful tractogram
        Tractogram to parcelate.
    templ_v : int
        Mask of all the voxels filtering the tractogram.
    nb_parcel : TYPE
        Number of sub-tractogram to generate. Chose powers of 2 for even parcels

    Returns
    -------
    list_subTract : list
        List of the generated sub-tractograms.
    list_parcels : list
        List of the masks corresponding to each sub-tractogram

    '''
    list_parcels = [templ_v]
    if nb_parcel == 1:
        list_subTract = [sft]
    else:
        dirList = ['x', 'y', 'z']  # direction for the cut (x, y or z)
        swapCount = 0  # Counts swap in cutting direction
        for cutCount in range(1, nb_parcel):
            dirCut = dirList[swapCount]
            parcelOri = list_parcels.pop(0)  # Takes the current biggest parcel...
            baryc = np.argwhere(parcelOri).mean(0).round().astype(int)
            split1 = np.zeros(parcelOri.shape, dtype=bool)
            split2 = np.zeros(parcelOri.shape, dtype=bool)
            if dirCut == 'x':
                split1[:baryc[0], ...] = parcelOri[:baryc[0], ...]
                split2[baryc[0]:, ...] = parcelOri[baryc[0]:, ...]
            elif dirCut == 'y':
                split1[:, :baryc[1], :] = parcelOri[:, :baryc[1], :]
                split2[:, baryc[1]:, :] = parcelOri[:, baryc[1]:, :]
            elif dirCut == 'z':
                split1[..., :baryc[2]] = parcelOri[..., :baryc[2]]
                split2[..., baryc[2]:] = parcelOri[..., baryc[2]:]
            list_parcels.extend([split1, split2])  # ... and divide it in 2
            if (cutCount & (cutCount+1) == 0):
                # If power of 2 (bit manipulation), change the dierction of the cut
                swapCount = (swapCount+1) % len(dirList)  # Circular indexing

        list_subTract = []
        for parcels in list_parcels:
            list_subTract.append(filter_grid_roi(sft, parcels, 'any', False)[0])
    return list_subTract, list_parcels


def main():
    t0 = time.time()
    t = t0
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
        # vox_list = np.argwhere(templ_v)
    else:
        list_reg_files = glob.glob(os.path.join(args.in_reg_dir, '*.nii*'))

    if args.parallel_proc is not None:
        nb_proc = int(args.parallel_proc)
    else:
        nb_proc = 1

    trk_list = glob.glob(os.path.join(args.in_dir_tractogram, '*.trk'))
    for n, trk_F in enumerate(trk_list):
        print(os.path.basename(trk_F) + f' ({n+1}/{len(trk_list)})')
        print('Loading the tractogram...')
        trk_sft = load_tractogram(trk_F, templ_i, bbox_valid_check=False)
        print('Filtering')
        trk_sft = filter_streamlines_by_length(trk_sft, 25, 250)
        trk_sft = hack_invalid_streamlines(trk_sft)
        trk_sft.to_vox()
        voxeled = True
        tupled = True
        if voxeled:
            trk_sft = voxelize_tractogram(trk_sft)
        tractomask = np.where(compute_tract_counts_map(trk_sft.streamlines, templ_i.shape),
                              True, False)
        empty_vol = templ_v > tractomask  # Voxels in templ but not in the tractogram
        templ_v = templ_v*tractomask  # Removing voxels with no streamlines
        if args.in_reg_dir is None:
            print('Preparing sub-tractograms to speed-up voxel-wise process...')
            nsub = 16
            sub_sfts, parcels = Parcelate_tractogram(trk_sft, templ_v, nsub)
            print(f'{nsub} sub-tractograms generated.')
            for n, (sub_trk_sft, parcel) in enumerate(zip(sub_sfts, parcels)):
                print(f'Starting process of sub-tragrogram {n+1}/{nsub}')
                vox_list = np.argwhere(parcel)
                if nb_proc == 1:
                    print('Starting')
                    vox_multi(vox_list, sub_trk_sft, templ_i, args.out_dir, voxeled, tupled)
                else:  # Parallelize
                    vox_batchs = np.array_split(vox_list, nb_proc)
                    print('Starting parallel process')
                    with Pool(processes=nb_proc) as pool:
                        poolCheck = pool.starmap_async(vox_multi, zip(vox_batchs,
                                                                      [sub_trk_sft]*nb_proc,
                                                                      [templ_i]*nb_proc,
                                                                      [args.out_dir]*nb_proc,
                                                                      [voxeled]*nb_proc,
                                                                      [tupled]*nb_proc))
                        while not poolCheck.ready():
                            time.sleep(1)
            print('"Saving" maps of voxels with no tract')
            vox_list = np.argwhere(empty_vol)
            vox_vol = np.zeros(templ_i.shape, dtype=bool)
            for vox in vox_list:
                out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
                out_Ftmp = os.path.join(args.out_dir, out_name)
                save_tmp_map(vox_vol, out_Ftmp, templ_i.affine, True)

        else:  # Regionwise priors
            for reg_F in list_reg_files:
                reg_i = nib.load(reg_F)
                reg_mask = reg_i.get_fdata().astype(bool)
                reg_fname = os.path.basename(reg_F)
                reg_name = reg_fname[:reg_fname.find('.nii')]
                out_name = f'probaMap_{reg_name}_tmp.nii.gz'
                out_Ftmp = os.path.join(args.out_dir, out_name)
                visitation_mapping(reg_mask, trk_sft, out_Ftmp, templ_i.affine)
        print(f'Elapsed time for current subject: {time.time()-t} sec')
        t = time.time()
    print('Last step: normalizing maps...')
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
    print('Done. Process over.')
    print(f'Total elapsed time: {time.time()-t0} sec')


if __name__ == "__main__":
    main()
