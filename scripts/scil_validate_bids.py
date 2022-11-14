#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a json file with DWI, T1 and fmap informations from BIDS folder
"""

import os

import argparse
from bids import BIDSLayout
from glob import glob
import json
import logging
import pathlib

import coloredlogs

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument("in_bids",
                   help="Input BIDS folder.")

    p.add_argument("out_json",
                   help="Output json file.")

    p.add_argument("--fs",
                   help='Output freesurfer path. It will add keys wmparc and '
                        'aparc+aseg.')

    p.add_argument('--participants_label', nargs="+",
                   help='The label(s) of the specific participant(s) you'
                        ' want to be be analyzed. Participants should not '
                        'include "sub-". If this parameter is not provided all'
                        ' subjects should be analyzed.')

    p.add_argument('--clean',
                   action='store_true',
                   help='If set, it will remove all the participants that '
                        'are missing any information.')

    p.add_argument("--readout", type=float, default=0.062,
                   help="Default total readout time value [%(default)s].")

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def _load_bidsignore_(bids_root):
    """Load .bidsignore file from a BIDS dataset, returns list of regexps"""
    bids_root = pathlib.Path(bids_root)
    bids_ignore_path = bids_root / ".bidsignore"
    if bids_ignore_path.exists():
        import re
        import fnmatch

        bids_ignores = bids_ignore_path.read_text().splitlines()
        return tuple(
            [
                re.compile(fnmatch.translate(bi))
                for bi in bids_ignores
                if len(bi) and bi.strip()[0] != "#"
            ]
        )
    return tuple()


def get_metadata(bf):
    """ Return the metadata of a BIDSFile

    Parameters
    ----------
    bf : BIDSFile object

    Returns
    -------
    Dictionnary containing the metadata
    """
    filename = bf.path.replace(
        bf.entities['extension'], '')

    with open(filename + '.json', 'r') as handle:
        return json.load(handle)


def get_dwi_associations(fmaps, bvals, bvecs, sbrefs):
    """ Return DWI associations

    Parameters
    ----------
    fmaps : List of BIDSFile object
        List of field maps

    bvals : List of BIDSFile object
        List of b-value files

    bvecs : List of BIDSFile object
        List of b-vector files

    sbrefs : List of BIDSFile object
        List of sbref files

    Returns
    -------
    Dictionnary containing the files associated to a DWI
    {dwi_filename: {'bval': bval_filename,
                    'bvec': bvec_filename,
                    'fmap': fmap_filename}}
    """
    associations = {}

    # Associate b-value files
    for bval in bvals:
        dwi_filename = os.path.basename(bval.path).replace('.bval', '.nii.gz')
        if dwi_filename not in associations.keys():
            associations[dwi_filename] = {"bval": bval.path}
        else:
            associations[dwi_filename]["bval"] = bval.path

    # Associate b-vector files
    for bvec in bvecs:
        dwi_filename = os.path.basename(bvec.path).replace('.bvec', '.nii.gz')
        if dwi_filename not in associations.keys():
            associations[dwi_filename] = {"bvec": bvec.path}
        else:
            associations[dwi_filename]["bvec"] = bvec.path

    # Associate field maps
    for fmap in fmaps:
        metadata = get_metadata(fmap)
        if isinstance(metadata.get('IntendedFor', ''), list):
            intended = metadata.get('IntendedFor', '')
        else:
            intended = [metadata.get('IntendedFor', '')]

        for target in intended:
            dwi_filename = os.path.basename(target)
            if dwi_filename not in associations.keys():
                associations[dwi_filename] = {'fmap': [fmap]}
            elif 'fmap' in associations[dwi_filename].keys():
                associations[dwi_filename]['fmap'].append(fmap)
            else:
                associations[dwi_filename]['fmap'] = [fmap]

    # Associate sbref
    for sbref in sbrefs:
        dwi_filename = os.path.basename(sbref.path).replace('sbref', 'dwi')
        if dwi_filename not in associations.keys():
            associations[dwi_filename] = {'sbref': [sbref]}
        elif 'sbref' in associations[dwi_filename].keys():
            associations[dwi_filename]['sbref'].append(sbref)
        else:
            associations[dwi_filename]['sbref'] = [sbref]

    return associations


def get_data(nSub, dwi, t1s, fs, associations, default_readout, clean):
    """ Return subject data

    Parameters
    ----------
    nSub : String
        Subject name

    dwi : list of BIDSFile object
        DWI objects

    t1s : List of BIDSFile object
        List of T1s associated to the current subject

    associations : Dictionnary
        Dictionnary containing files associated to the DWI

    default_readout : Float
        Default readout time

    Returns
    -------
    Dictionnary containing the metadata
    """
    bvec_path = ['todo', '']
    bval_path = ['todo', '']
    dwi_path = ['todo', '']
    PE = ['todo', '']
    topup_fmap = ['', '']
    topup_sbref = ['', '']
    fmaps = ['', '']
    sbref = ['', '']
    nSess = 0
    if 'session' in dwi[0].entities:
        nSess = dwi[0].entities['session']

    nRun = 0
    if 'run' in dwi[0].entities:
        nRun = dwi[0].entities['run']

    for index, curr_dwi in enumerate(dwi):
        dwi_path[index] = curr_dwi.path

        if curr_dwi.filename in associations.keys():
            if "bval" in associations[curr_dwi.filename].keys():
                bval_path[index] = associations[curr_dwi.filename]['bval']
            if "bvec" in associations[curr_dwi.filename].keys():
                bvec_path[index] = associations[curr_dwi.filename]['bvec']
            if "fmap" in associations[curr_dwi.filename].keys():
                fmaps[index] = associations[curr_dwi.filename]['fmap']
                if len(fmaps[index]) == 1 and isinstance(fmaps[index][0], list):
                    fmaps[index] = [x for xs in fmaps[index] for x in xs]
            if "sbref" in associations[curr_dwi.filename].keys():
                sbref[index] = associations[curr_dwi.filename]['sbref']
                if len(sbref[index]) == 1 and isinstance(sbref[index][0], list):
                    sbref[index] = [x for xs in sbref[index] for x in xs]

            conversion = {"i": "x", "j": "y", "k": "z"}
            dwi_metadata = get_metadata(curr_dwi)
            if 'PhaseEncodingDirection' in dwi_metadata and index == 0:
                dwi_PE = dwi_metadata['PhaseEncodingDirection']
                dwi_PE = dwi_PE.replace(dwi_PE[0], conversion[dwi_PE[0]])
                if len(dwi_PE) == 1:
                    PE[index] = dwi_PE
                    PE[index+1] = dwi_PE + '-'
                else:
                    PE[index] = dwi_PE
                    PE[index+1] = dwi_PE[0]
            elif clean:
                return {}

        # Find b0 for topup, take the first one
        # Check fMAP
        totalreadout = default_readout
        fmaps = [fmap for fmap in fmaps if fmap != '']
        if not fmaps:
            if 'TotalReadoutTime' in dwi_metadata:
                totalreadout = dwi_metadata['TotalReadoutTime']
        else:
            if isinstance(fmaps[0], list):
                fmaps = [x for xs in fmaps for x in xs]

            for nfmap in fmaps:
                nfmap_metadata = get_metadata(nfmap)
                if 'PhaseEncodingDirection' in nfmap_metadata:
                    fmap_PE = nfmap_metadata['PhaseEncodingDirection']
                    fmap_PE = fmap_PE.replace(fmap_PE[0], conversion[fmap_PE[0]])

                    opposite_PE = PE.index(fmap_PE)
                    if 'TotalReadoutTime' in dwi_metadata:
                        if 'TotalReadoutTime' in nfmap_metadata:
                            dwi_RT = dwi_metadata['TotalReadoutTime']
                            fmap_RT = nfmap_metadata['TotalReadoutTime']
                            if dwi_RT != fmap_RT and totalreadout == '':
                                totalreadout = 'error_readout'
                                topup_fmap[opposite_PE] = 'error_readout'
                            elif dwi_RT == fmap_RT:
                                topup_fmap[opposite_PE] = nfmap.path
                                totalreadout = dwi_RT
                    else:
                        topup_fmap[opposite_PE] = nfmap.path
                        totalreadout = default_readout

        if sbref[index] != '' and len(sbref[index]) == 1:
            topup_sbref[index] = sbref[index][0].path

    if len(dwi) == 2:
        if not any(s == '' for s in topup_sbref):
            topup = topup_sbref
        elif not any(s == '' for s in topup_fmap):
            topup = topup_fmap
        else:
            topup = ['todo', 'todo']
    elif len(dwi) == 1:
        if topup_fmap[1] != '':
            topup = topup_fmap
        else:
            topup = ['', '']
    else:
        print("""
              BIDS structure unkown.Please send an issue:
              https://github.com/scilus/scilpy/issues
              """)

    t1_path = 'todo'
    wmparc_path = ''
    aparc_aseg_path = ''
    if fs:
        t1_path = fs[0]
        wmparc_path = fs[1]
        aparc_aseg_path = fs[2]
    else:
        t1_nSess = []
        if not t1s and clean:
            return {}

        for t1 in t1s:
            if 'session' in t1.get_entities().keys() and\
                    t1.get_entities()['session'] == nSess:
                t1_nSess.append(t1)
            elif 'session' not in t1.get_entities().keys():
                t1_nSess.append(t1)

        if len(t1_nSess) == 1:
            t1_path = t1_nSess[0].path

    return {'subject': nSub,
            'session': nSess,
            'run': nRun,
            't1': t1_path,
            'wmparc': wmparc_path,
            'aparc_aseg': aparc_aseg_path,
            'dwi': dwi_path[0],
            'bvec': bvec_path[0],
            'bval': bval_path[0],
            'rev_dwi': dwi_path[1],
            'rev_bvec': bvec_path[1],
            'rev_bval': bval_path[1],
            'topup': topup[0],
            'rev_topup': topup[1],
            'DWIPhaseEncodingDir': PE[0],
            'rev_DWIPhaseEncodingDir': PE[1],
            'TotalReadoutTime': totalreadout}


def associate_dwis(layout, nSub):
    """ Return subject data
    Parameters
    ----------
    layout: pyBIDS layout
        BIDS layout
    nSub: String
        Current subject to analyse
    Returns
    -------
    all_dwis: list
        List of dwi
    """
    all_dwis = []
    if layout.get_sessions(subject=nSub):
        for curr_sess in layout.get_sessions(subject=nSub):
            dwis = layout.get(subject=nSub,
                              session=curr_sess,
                              datatype='dwi', extension='nii.gz',
                              suffix='dwi')

            if len(dwis) == 1:
                all_dwis.append(dwis)
            elif len(dwis) > 1:
                all_runs = [curr_dwi.entities['run'] for curr_dwi in dwis if 'run' in curr_dwi.entities]
                if all_runs:
                    for curr_run in all_runs:
                        dwis = layout.get(subject=nSub,
                                          session=curr_sess,
                                          run=curr_run,
                                          datatype='dwi', extension='nii.gz',
                                          suffix='dwi')
                        if len(dwis) == 2:
                            all_dwis.append(dwis)
                        else:
                            print("ERROR MORE DWI THAN EXPECTED")
                elif len(dwis) == 2:
                    all_dwis.append(dwis)
                else:
                    print(dwis)
                    print("ERROR MORE DWI THAN EXPECTED")
    else:
        dwis = layout.get(subject=nSub,
                          datatype='dwi', extension='nii.gz',
                          suffix='dwi')
        if len(dwis) == 1:
            all_dwis.append(dwis)
        elif len(dwis) > 1:
            all_runs = [curr_dwi.entities['run'] for curr_dwi in dwis if 'run' in curr_dwi.entities]
            if all_runs:
                for curr_run in all_runs:
                    dwis = layout.get(subject=nSub,
                                      run=curr_run,
                                      datatype='dwi', extension='nii.gz',
                                      suffix='dwi')
                    if len(dwis) <= 2:
                        all_dwis.append(dwis)
                    else:
                        print("ERROR MORE DWI THAN EXPECTED")
            elif len(dwis) == 2:
                all_dwis.append(dwis)
            else:
                print("ERROR MORE DWI THAN EXPECTED")

    return all_dwis


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_outputs_exist(parser, args, args.out_json)

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.getLogger().setLevel(log_level)
    coloredlogs.install(level=log_level)

    data = []
    layout = BIDSLayout(args.in_bids, validate=False,
                        ignore=_load_bidsignore_(args.in_bids))
    subjects = layout.get_subjects()

    if args.participants_label:
        subjects = [nSub for nSub in args.participants_label if nSub in subjects]

    subjects.sort()

    logging.info("Found {} subject(s)".format(len(subjects)))

    for nSub in subjects:
        mess = '# Validating subject: {}'.format(nSub)
        logging.info("-" * len(mess))
        logging.info(mess)
        dwis = associate_dwis(layout, nSub)

        fs_inputs = []
        t1s = []

        if args.fs:
            logging.info("# Looking for FS files")
            t1_fs = glob(os.path.join(args.fs, 'sub-' + nSub, 'mri/T1.mgz'))
            wmparc = glob(os.path.join(args.fs, 'sub-' + nSub, 'mri/wmparc.mgz'))
            aparc_aseg = glob(os.path.join(args.fs, 'sub-' + nSub,
                                           'mri/aparc+aseg.mgz'))
            if len(t1_fs) == 1 and len(wmparc) == 1 and len(aparc_aseg) == 1:
                fs_inputs = [t1_fs[0], wmparc[0], aparc_aseg[0]]
        else:
            logging.info("# Looking for T1 files")
            t1s = layout.get(subject=nSub,
                             datatype='anat', extension='nii.gz',
                             suffix='T1w')

        fmaps = layout.get(subject=nSub,
                           datatype='fmap', extension='nii.gz',
                           suffix='epi')

        bvals = layout.get(subject=nSub,
                           datatype='dwi', extension='bval',
                           suffix='dwi')
        bvecs = layout.get(subject=nSub,
                           datatype='dwi', extension='bvec',
                           suffix='dwi')
        sbrefs = layout.get(subject=nSub,
                            datatype='dwi', extension='nii.gz',
                            suffix='sbref')

        # Get associations relatives to DWIs
        associations = get_dwi_associations(fmaps, bvals, bvecs, sbrefs)

        # Get the data for each run of DWIs
        for dwi in dwis:
            data.append(get_data(nSub, dwi, t1s, fs_inputs, associations,
                                 args.readout, args.clean))

    if args.clean:
        data = [d for d in data if d]

    with open(args.out_json, 'w') as outfile:
        json.dump(data,
                  outfile,
                  indent=4,
                  separators=(',', ': '),
                  sort_keys=True)
        # Add trailing newline for POSIX compatibility
        outfile.write('\n')


if __name__ == '__main__':
    main()
