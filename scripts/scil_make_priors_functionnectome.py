#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Functionnectome priors using *.trk files from a given folder.
Create a new folder containing all the probability maps derived from the input.
All the tractograms should be in the same space than the brain template.
The brain template defines which voxels are used to generate the probability map.

For region-wise priors, the region masks must be provided as separate files,
with each file named as the region it contains (e.g. 'SupFront_Left.nii.gz')

# TODO Add a 'Resume' option (with numbered tmp files)
# TODO Improve RAM usage by sharing the tractogram between processes
"""

import nibabel as nib
import numpy as np
import sys
import os
import glob
import argparse
import time
from shutil import copyfile
from pathlib import Path
# from itertools import compress
from multiprocessing import (Pool,
                             current_process)
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.streamlines import hack_invalid_streamlines
from scilpy.tractanalysis.uncompress import uncompress
from dipy.io.streamline import load_tractogram
from nibabel.streamlines.array_sequence import ArraySequence as AS
# from scilpy.tracking.tools import resample_streamlines_step_size
# from dipy.io.stateful_tractogram import StatefulTractogram
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

    p.add_argument('--prep',
                   action='store_true',
                   help='Prepare the next tractogram while the current one is'
                   ' being processed. Speeds up a bit but increase RAM needs.'
                   '\n(Voxelwise priors only, requires parallel processing).')
    p.add_argument('-rd', '--in_reg_dir',
                   help='Path to the atlas file defining the regions.')
    p.add_argument('-pp', '--parallel_proc',
                   help='Number of parallel processes to launch '
                   '\n(Voxelwise priors only).')
    p.add_argument('-ep', '--from_endpoints',
                   action='store_true',
                   help='Only streamlines with endpoints in the mask of the'
                        ' voxel/region are taken into account')
    p.add_argument('-l', '--extremity_length',
                   default=1,
                   help='Number of away from an endpoint to consider as streamline'
                        ' extremity. Only works with "--from_endpoints"')
    add_overwrite_arg(p)

    return p


def save_tmp_map(vol, file, affine):
    if os.path.isfile(file):
        vol = vol.astype('float32')
        tmp_i = nib.load(file)
        tmp_vol = tmp_i.get_fdata(dtype=np.float32)
        vol += tmp_vol
    vol_str_i = nib.Nifti1Image(vol.astype('float32'), affine)
    nib.save(vol_str_i, file)


def voxelise_strm(strmls):
    for strm in strmls:
        strm_r = strm.astype('uint16')
        _, ind = np.unique(strm_r, return_index=True, axis=0)
        strm_v = strm_r[np.sort(ind)]
        yield strm_v


def voxelize_tractogram(sft):
    '''
    Resample the streamlines at the voxel level.
    Floor the points of the streamlines to the voxel they belong to and
    removes duplicate points. Output a list of streamlines with each point given as a tuple
    Or uses uncompress
    '''
    sft.to_vox()
    sft.to_corner()
    resamp = False
    for strm in sft.streamlines[range(1000)]:  # Check the first 1000 strm for compression
        steps = [np.linalg.norm(strm[i]-strm[i+1]) for i in range(len(strm)-1)]
        if max(steps) > 1.8:  # Max distance between neighbouring voxels should be 1.73
            print('Step size bigger than a voxel (tractogram probably compressed).'
                  ' Using uncompress function.')
            resamp = True
            break

    if resamp:
        voxed_strms = uncompress(sft.streamlines)  # Ouput fatter streamlines...
    else:  # Classic (thinner) but slower and bad with compressed streamlines
        voxed_strms = nib.streamlines.array_sequence.ArraySequence(voxelise_strm(sft.streamlines))
    return voxed_strms


def prep_streamlines(track_Files, ref=None, from_endpoints=False, distEnd=None):
    """
    Load and voxelise the tractogram.
    Only for voxel-wise priors.

    Parameters
    ----------
    trk_File : dict
        Path to the .trk file (and the temporary .npz saving file if needed)
    ref : nifti1.Nifti1Image
        Nibabel image of the reference template.
    from_endpoints : args.from_endpoints

    distEnd : vox2end

    Returns
    -------
    None.

    """
    trk_File = track_Files['trk_File']
    tmp_File = track_Files['tmp_File']
    if tmp_File:
        print("Preloading next tractogram")
    else:
        print("Loading tractogram")
    sft = load_tractogram(trk_File, ref, bbox_valid_check=False)
    print("Filtering")
    sft = filter_streamlines_by_length(sft, 25, 250)
    sft = hack_invalid_streamlines(sft)
    sft.to_vox()
    sft.to_corner()
    tractomask = np.where(compute_tract_counts_map(sft.streamlines, ref.shape),
                          True, False)
    templ_v = ref.get_fdata().astype(bool)
    empty_vol = templ_v > tractomask  # Voxels in templ but not in the tractogram
    templ_v = templ_v*tractomask  # Removing voxels with no streamlines
    voxel_list = np.argwhere(templ_v)
    voxel_list = [tuple(ind) for ind in voxel_list]
    print("Resampling at the voxel level")
    stream_tpled = voxelize_tractogram(sft)  # Not actually tupled anymore
    if from_endpoints:
        stream_tpled = nib.streamlines.array_sequence.ArraySequence(
                            get_endpoints(stream_tpled,
                                          distEnd))
    if tmp_File:
        stream_tpled.save(tmp_File)
        print("Preparation for next subject done. Waiting for current subject process to end.")
        return (empty_vol, voxel_list)
    else:
        return (empty_vol, voxel_list, stream_tpled)


def visitation_mapping(area_mask, sft, out_Ftmp, affine, endpoints):
    '''
    Creates a binary volume of all streamlines passing trhough the ROI/mask
    and save it
    '''
    # 1
    if endpoints:
        area_strm, _ = filter_grid_roi(sft, area_mask, 'either_end', False)
    else:
        area_strm, _ = filter_grid_roi(sft, area_mask, 'any', False)
    # 2
    vol_strm = np.where(compute_tract_counts_map(area_strm.streamlines, area_mask.shape), 1, 0)
    # 3
    save_tmp_map(vol_strm, out_Ftmp, affine)


def tmp2final(tmp_list, nb_trk, outdir, affine):
    for tmp_F in tmp_list:
        tmp_i = nib.load(tmp_F)
        tmp_vol = tmp_i.get_fdata(dtype=np.float32)
        out_vol = tmp_vol/nb_trk
        out_name = os.path.basename(tmp_F).replace('_tmp', '')
        outF = os.path.join(outdir, out_name)
        out_i = nib.Nifti1Image(out_vol, affine)
        nib.save(out_i, outF)
        os.remove(tmp_F)


def get_endpoints(strml, lenpLen):
    for strm in strml:
        if len(strm) > lenpLen*2:
            yield np.delete(strm, slice(lenpLen, -lenpLen), 0)
        else:
            yield strm


def loop_on_strm(voxel_list, streamlines=None):
    '''
    streamlines must be from a voxelised sft (cf. voxelize_tractogram) and tupled
    Returns a list of dict with each voxel a key and each value the index of
    the streamlines in that voxel

    voxel_list should be tupled
    '''
    voxdict = {tuple(v): [] for v in voxel_list}
    for ind, strm in enumerate(streamlines):
        for vox in strm:
            try:
                voxdict[tuple(vox)].append(ind)
            except KeyError:  # When the streamline voxel is out of the template
                pass
    for v in voxdict:
        voxdict[v] = np.array(voxdict[v], dtype='uint32')
    return voxdict


def loop_on_strm_multi(strm_list):
    streamlines = stream_tpled[strm_list]
    voxdict = {tuple(v): [] for v in voxel_list}
    for ind, strm in enumerate(streamlines):
        for vox in strm:
            try:
                voxdict[tuple(vox)].append(ind)
            except KeyError:  # When the streamline voxel is out of the template
                pass
    for v in voxdict:
        voxdict[v] = np.array(voxdict[v], dtype='uint32')
    return voxdict


def strm_multi(vlist, vdict=None, stream_tupled=None, out_dir=None, templ_i=None, logs=False):
    if 'dict_var' in globals():  # When in a worker of a pool
        # stream_tupled = dict_var['stream_tupled']  # In global variable already
        stream_tupled = stream_tpled
        vdict = vox_dict
        out_dir = dict_var['out_dir']
        templ_i = dict_var['templ_i']
    else:  # Must give input (vlist, vdict, stream_tupled, out_dir, templ_i)
        pass
    if logs:  # For testing
        current = current_process()
        logDir = str(Path(out_dir).parent.absolute())
        if current.name == 'MainProcess':
            logFile = os.path.join(logDir, 'log.txt')
        else:
            logFile = os.path.join(logDir, f'log_{current._identity[0]}.txt')
        for vox in vlist:
            vox = tuple(vox)
            out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
            logtxt = f'Processing and saving {out_name}\n'
            with open(logFile, "a") as log:
                log.write(logtxt)
            out_Ftmp = os.path.join(out_dir, out_name)
            vox_vol = np.zeros(templ_i.shape, dtype=np.float32)
            for strmInd in vdict[vox]:
                vox_vol[tuple(stream_tupled[strmInd].T)] = 1
            save_tmp_map(vox_vol, out_Ftmp, templ_i.affine)
        logtxt = 'Processing over for this worker\n'
        with open(logFile, "a") as log:
            log.write(logtxt)
    else:  # Normal case
        for vox in vdict:
            out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
            out_Ftmp = os.path.join(out_dir, out_name)
            vox_vol = np.zeros(templ_i.shape, dtype=np.float32)
            for strmInd in vdict[vox]:
                vox_vol[tuple(stream_tupled[strmInd].T)] = 1
            save_tmp_map(vox_vol, out_Ftmp, templ_i.affine)


def init_worker(out_dir, templ_i):
    global dict_var
    dict_var = {'out_dir': out_dir,
                'templ_i': templ_i}


def init_worker0(voxL):
    global voxel_list
    voxel_list = voxL
# %%


def main():
    global stream_tpled
    global vox_dict

    t0 = time.time()
    t = t0
    parser = _build_arg_parser()
    args = parser.parse_args()
    if 'win' in sys.platform.lower():
        raise EnvironmentError('Sorry, this program does not run on Windows currently.')

    assert_inputs_exist(parser, [args.in_template_file], args.in_reg_dir)
    if not os.path.isdir(args.in_dir_tractogram):
        parser.error(f'{args.in_dir_tractogram} is not a directory.')
    if not len(os.listdir(args.in_dir_tractogram)):
        parser.error(f'{args.in_dir_tractogram} is empty.')
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir)

    if args.parallel_proc is not None:
        nb_proc = int(args.parallel_proc)
    else:
        nb_proc = 1
    vox2end = int(args.extremity_length)

    templ_i = nib.load(args.in_template_file)
    if args.in_reg_dir is None:
        templ_v = templ_i.get_fdata().astype(bool)
        max_file_nb = templ_v.sum()
    else:
        list_reg_files = glob.glob(os.path.join(args.in_reg_dir, '*.nii*'))

    trk_list = glob.glob(os.path.join(args.in_dir_tractogram, '*.trk'))
    for n, trk_F in enumerate(trk_list):
        print(os.path.basename(trk_F) + f' ({n+1}/{len(trk_list)})')
        if args.in_reg_dir is None:
            # Declqring variables defined in later loops
            next_empty_vol = None
            next_voxel_list = None
            next_tmp = None
            if n == 0 or not args.prep:
                trackFiles = {'trk_File': trk_F, 'tmp_File': ''}
                (empty_vol, voxel_list, stream_tpled) = prep_streamlines(
                    trackFiles, templ_i, args.from_endpoints, vox2end)
            else:
                empty_vol = next_empty_vol
                voxel_list = next_voxel_list
                stream_tpled = AS.load(next_tmp)

            if nb_proc == 1:
                vox_dict = loop_on_strm(voxel_list, stream_tpled)
                print('Starting process')
                strm_multi(voxel_list, vox_dict, stream_tpled, args.out_dir, templ_i)
            else:  # Parallelize
                print('Starting parallel process: Indexing streamlines to voxels')
                strm_batch = np.array_split(range(len(stream_tpled)), nb_proc)
                with Pool(processes=nb_proc,
                          initializer=init_worker0,
                          initargs=(voxel_list,)) as pool:
                    poolComp = pool.map_async(loop_on_strm_multi, strm_batch)
                    vox_dictL = poolComp.get()
                vox_dict = vox_dictL.pop()
                while len(vox_dictL):
                    subdict = vox_dictL.pop()
                    for vox in vox_dict:
                        vox_dict[vox] = np.concatenate((vox_dict[vox], subdict[vox]))

                vox_batchs = np.array_split(voxel_list, nb_proc)
                print('Starting parallel process: Creating and saving the 3D maps')
                with Pool(processes=nb_proc,
                          initializer=init_worker,
                          initargs=(args.out_dir,
                                    templ_i)
                          ) as pool:
                    poolCheck = pool.map_async(strm_multi, vox_batchs)
                    if args.prep and n > 0:
                        os.remove(next_tmp)  # Cleaning previous subject's file
                    if args.prep and n < len(trk_list)-1:
                        next_file = trk_list[n+1]
                        next_tmp = os.path.splitext(next_file)[0] + '_tmp.npz'
                        trackFiles = {'trk_File': next_file, 'tmp_File': next_tmp}
                        (next_empty_vol, next_voxel_list) = prep_streamlines(
                            trackFiles, templ_i, args.from_endpoints, vox2end)
                    poolCheck.wait()

            print('"Saving" maps of voxels with no tract')
            vox_list0 = np.argwhere(empty_vol)
            nb_of_files = len(os.listdir(args.out_dir))
            if len(vox_list0) and nb_of_files < max_file_nb:
                vox_vol0 = np.zeros(templ_i.shape, dtype=bool)
                n0 = 0
                vox0 = vox_list0[n0]
                out_name0 = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox0)
                out_Ftmp0 = os.path.join(args.out_dir, out_name0)
                while os.path.isfile(out_Ftmp0) and n0 < len(vox_list0):
                    n0 += 1
                    vox0 = vox_list0[n0]
                    out_name0 = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox0)
                    out_Ftmp0 = os.path.join(args.out_dir, out_name0)
                save_tmp_map(vox_vol0, out_Ftmp0, templ_i.affine)
                for vox in vox_list0[n0:]:
                    out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
                    out_Ftmp = os.path.join(args.out_dir, out_name)
                    if not os.path.isfile(out_Ftmp):
                        copyfile(out_Ftmp0, out_Ftmp)

        else:  # Regionwise priors
            print('Loading the tractogram...')
            trk_sft = load_tractogram(trk_F, templ_i, bbox_valid_check=False)
            print('Filtering')
            trk_sft = filter_streamlines_by_length(trk_sft, 25, 250)
            trk_sft = hack_invalid_streamlines(trk_sft)
            trk_sft.to_vox()
            trk_sft.to_corner()
            for reg_F in list_reg_files:
                reg_i = nib.load(reg_F)
                reg_mask = reg_i.get_fdata().astype(bool)
                reg_fname = os.path.basename(reg_F)
                reg_name = reg_fname[:reg_fname.find('.nii')]
                out_name = f'probaMap_{reg_name}_tmp.nii.gz'
                out_Ftmp = os.path.join(args.out_dir, out_name)
                visitation_mapping(reg_mask,
                                   trk_sft,
                                   out_Ftmp,
                                   templ_i.affine,
                                   args.from_endpoints)
        print(f'Elapsed time for current subject: {time.time()-t} sec')
        t = time.time()
    print('Last step: normalizing maps...')
    tmp_list = glob.glob(os.path.join(args.out_dir, '*tmp.nii.gz'))
    if (args.in_reg_dir is not None) or nb_proc == 1:
        tmp2final(tmp_list, len(trk_list), args.out_dir, templ_i.affine)
    else:  # Parallel proc
        tmpF_batch = np.array_split(tmp_list, nb_proc)
        print('Starting parallel process')
        with Pool(processes=nb_proc) as pool:
            poolCheck = pool.starmap_async(tmp2final, zip(tmpF_batch,
                                                          [len(trk_list)]*nb_proc,
                                                          [args.out_dir]*nb_proc,
                                                          [templ_i.affine]*nb_proc,))
            poolCheck.wait()
    print('Done. Process over.')
    print(f'Total elapsed time: {time.time()-t0} sec')


if __name__ == "__main__":
    main()

# %% Old functions not usefull anymore


# def init_worker(stream_tupled, out_dir, templ_i):
#     global dict_var
#     dict_var = {'stream_tupled': stream_tupled,
#                 'out_dir': out_dir,
#                 'templ_i': templ_i}


# def tupled_streamlines(sft):
#     '''
#     sft must be a voxelised sft (cf. voxelize_tractogram)
#     Returns a list of streamlines with each point given as a tuple
#     '''
#     return [list(zip(st.T[0], st.T[1], st.T[2])) for st in sft.streamlines]


# def Parcelate_tractogram(sft, templ_v, nb_parcel=16):
#     '''
#     Divide the input tractogram into sub-tractograms by parcelating the input
#     templ_v into similarly sized sub volumes and filtering the tractogram by
#     these subvolumes. Might not be well adapted for GM masks in input (for
#     those, chose nb_parcel=8).
#     (Different sub-tractograms can share streamlines)

#     Parameters
#     ----------
#     sft : Stateful tractogram
#         Tractogram to parcelate.
#     templ_v : int
#         Mask of all the voxels filtering the tractogram.
#     nb_parcel : TYPE
#         Number of sub-tractogram to generate. Chose powers of 2 for even parcels

#     Returns
#     -------
#     list_subTract : list
#         List of the generated sub-tractograms.
#     list_parcels : list
#         List of the masks corresponding to each sub-tractogram

#     '''
#     list_parcels = [templ_v]
#     if nb_parcel == 1:
#         list_subTract = [sft]
#     else:
#         dirList = ['x', 'y', 'z']  # direction for the cut (x, y or z)
#         swapCount = 0  # Counts swap in cutting direction
#         for cutCount in range(1, nb_parcel):
#             dirCut = dirList[swapCount]
#             parcelOri = list_parcels.pop(0)  # Takes the current biggest parcel...
#             baryc = np.argwhere(parcelOri).mean(0).round().astype(int)
#             split1 = np.zeros(parcelOri.shape, dtype=bool)
#             split2 = np.zeros(parcelOri.shape, dtype=bool)
#             if dirCut == 'x':
#                 split1[:baryc[0], ...] = parcelOri[:baryc[0], ...]
#                 split2[baryc[0]:, ...] = parcelOri[baryc[0]:, ...]
#             elif dirCut == 'y':
#                 split1[:, :baryc[1], :] = parcelOri[:, :baryc[1], :]
#                 split2[:, baryc[1]:, :] = parcelOri[:, baryc[1]:, :]
#             elif dirCut == 'z':
#                 split1[..., :baryc[2]] = parcelOri[..., :baryc[2]]
#                 split2[..., baryc[2]:] = parcelOri[..., baryc[2]:]
#             list_parcels.extend([split1, split2])  # ... and divide it in 2
#             if (cutCount & (cutCount+1) == 0):
#                 # If power of 2 (bit manipulation), change the dierction of the cut
#                 swapCount = (swapCount+1) % len(dirList)  # Circular indexing

#         list_subTract = []
#         for parcels in list_parcels:
#             list_subTract.append(filter_grid_roi(sft, parcels, 'any', False)[0])
#     return list_subTract, list_parcels


# def voxeled_visitation_mapping(area_mask, streamlines, out_Ftmp, affine, tupled):
#     ''' sft must be a voxelised sft (cf. voxelize_tractogram)'''
#     # 1
#     ind_vox = np.argwhere(area_mask)
#     if tupled:
#         ind_vox = [tuple(ind) for ind in ind_vox]
#         indstrm = [any(ind in strm for ind in ind_vox) for strm in streamlines]
#         area_strm = list(compress(streamlines, indstrm))
#         # 2
#         vol_strm = np.zeros(area_mask.shape, dtype=np.float32)
#         for strm in area_strm:
#             for v in strm:
#                 vol_strm[v] = 1
#     else:
#         indstrm = [any(any(np.equal(vox, strm).all(1)) for vox in ind_vox) for strm in streamlines]
#         area_strm = streamlines[indstrm]
#         # 2
#         vol_strm = np.zeros(area_mask.shape, dtype=np.float32)
#         for strm in area_strm:
#             vol_strm[tuple(strm.T)] = 1
#     # 3
#     save_tmp_map(vol_strm, out_Ftmp, affine)


# def vox_multi(vox_batch, sft, base_im, outdir, voxeled=True, tupled=True):
#     """
#     Iterates over the voxels, creating the appropriate mask and file-name, and
#     runs visitation_mapping()

#     """
#     if tupled and not voxeled:
#         raise ValueError('The tractogram is not voxeled, but shoud be for tupling')
#     base_mask = np.zeros(base_im.shape, dtype=bool)
#     if tupled:
#         streamlines = tupled_streamlines(sft)
#     else:
#         streamlines = sft.streamlines
#     for vox in vox_batch:
#         vox_mask = base_mask.copy()
#         vox_mask[tuple(vox)] = True
#         out_name = 'probaMap_{:02d}_{:02d}_{:02d}_tmp.nii.gz'.format(*vox)
#         out_Ftmp = os.path.join(outdir, out_name)
#         if voxeled:
#             voxeled_visitation_mapping(vox_mask, streamlines, out_Ftmp, base_im.affine, tupled)
#         else:
#             visitation_mapping(vox_mask, sft, out_Ftmp, base_im.affine)
