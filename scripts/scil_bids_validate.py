#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a json file from a BIDS dataset detailling all info
needed for tractoflow
- DWI/rev_DWI
- T1
- fmap/sbref (based on IntendedFor entity)
- Freesurfer (optional - could be one per participant
              or one per participant/session)

The BIDS dataset MUST be homogeneous.
The metadata need to be uniform across all participants/sessions/runs

Mandatory entity: IntendedFor
Sensitive entities: PhaseEncodingDirection, TotalReadoutTime, direction

Formerly: scil_validate_bids.py
"""

import os

import argparse
from bids import BIDSLayout, BIDSLayoutIndexer
from bids.layout import Query
from glob import glob
import json
import logging
import pathlib

import coloredlogs

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


conversion = {"i": "x",
              "i-": "x-",
              "j": "y",
              "j-": "y-",
              "k": "z",
              "k-": "z-",
              "LR": "x",
              "RL": "x-",
              "AP": "y",
              "PA": "y-"}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    p.add_argument("in_bids",
                   help="Input BIDS folder.")

    p.add_argument("out_json",
                   help="Output json file.")

    p.add_argument('--bids_ignore',
                   help="If you want to ignore some subjects or some files, "
                        "you can provide an extra bidsignore file."
                        "Check: https://github.com/bids-standard"
                        "/bids-validator#bidsignore")

    p.add_argument("--fs",
                   help='Output freesurfer path. It will add keys wmparc and '
                        'aparc+aseg.')

    p.add_argument('--clean',
                   action='store_true',
                   help='If set, it will remove all the participants that '
                        'are missing any information.')

    p.add_argument("--readout", type=float, default=0.062,
                   help="Default total readout time value [%(default)s].")

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _load_bidsignore_(bids_root, additional_bidsignore=None):
    """Load .bidsignore file from a BIDS dataset, returns list of regexps"""
    bids_root = pathlib.Path(bids_root)
    bids_ignore_path = bids_root / ".bidsignore"
    bids_ignores = []
    if bids_ignore_path.exists():
        bids_ignores = bids_ignores +\
                bids_ignore_path.read_text().splitlines()

    if additional_bidsignore:
        bids_ignores = bids_ignores + \
            pathlib.Path(os.path.abspath(additional_bidsignore)).read_text().splitlines()

    if bids_ignores:
        import re
        import fnmatch
        return tuple(
            [
                re.compile(fnmatch.translate(bi))
                for bi in bids_ignores
                if len(bi) and bi.strip()[0] != "#"
            ]
        )
    return tuple()


def get_opposite_pe_direction(phase_encoding_direction):
    """ Return opposite direction (works with direction
        or PhaseEncodingDirection)

    Parameters
    ----------
    phase_encoding_direction: String
        Phase encoding direction either AP/LR or j/j- i/i- format

    Returns
    -------
        Opposite phase direction
    """
    if len(phase_encoding_direction) == 2 and phase_encoding_direction[1] != '-':
        return phase_encoding_direction[::-1]
    elif len(phase_encoding_direction) == 2:
        return phase_encoding_direction[0]
    else:
        return phase_encoding_direction+'-'


def get_data(layout, nSub, dwis, t1s, fs, default_readout, clean):
    """ Return subject data

    Parameters
    ----------
    layout: BIDS layout
        Current BIDS layout

    nSub : String
        Subject name

    dwis : list of BIDSFile object
        DWI objects

    t1s : List of BIDSFile object
        List of T1s associated to the current subject

    fs : List of fs paths
        List of freesurfer path

    default_readout : Float
        Default readout time

    clean: Boolean
        If set, if some critical files are missing it will
        remove this specific subject/session/run

    Returns
    -------
    Dictionnary containing the metadata
    """

    bvec_path = ['todo', '']
    bval_path = ['todo', '']
    dwi_path = ['todo', '']
    totalreadout = default_readout
    PE = ['todo', '']
    topup_suffix = {'epi': ['', ''], 'sbref': ['', '']}
    nSess = 0
    nRun = 0

    if len(dwis) == 2:
        dwi_path[1] = dwis[1].path
        bvec_path[1] = layout.get_bvec(dwis[1].path)
        bval_path[1] = layout.get_bval(dwis[1].path)
        if 'direction' in dwis[1].entities:
            PE[1] = conversion[dwis[1].entities['direction']]
        elif 'PhaseEncodingDirection' in dwis[1].entities:
            PE[1] = conversion[dwis[1].entities['PhaseEncodingDirection']]

    curr_dwi = dwis[0]
    dwi_path[0] = curr_dwi.path
    bvec_path[0] = layout.get_bvec(curr_dwi.path)
    bval_path[0] = layout.get_bval(curr_dwi.path)

    if 'TotalReadoutTime' in curr_dwi.entities:
        totalreadout = curr_dwi.entities['TotalReadoutTime']

    if 'session' in curr_dwi.entities:
        nSess = curr_dwi.entities['session']

    if 'run' in curr_dwi.entities:
        nRun = curr_dwi.entities['run']

    IntendedForPath = os.path.sep.join(curr_dwi.relpath.split(os.path.sep)[1:])
    related_files = layout.get(part="mag",
                               IntendedFor=IntendedForPath,
                               regex_search=True,
                               TotalReadoutTime=totalreadout,
                               invalid_filters='drop') +\
        layout.get(part=Query.NONE,
                   IntendedFor=IntendedForPath,
                   regex_search=True,
                   TotalReadoutTime=totalreadout,
                   invalid_filters='drop')
    related_files_filtered = []
    for curr_related in related_files:
        if curr_related.entities['suffix'] != 'dwi' and\
           curr_related.entities['extension'] == '.nii.gz':
            related_files_filtered.append(curr_related)

    related_files = related_files_filtered
    direction_key = False
    if 'direction' in curr_dwi.entities:
        direction_key = 'direction'
    elif 'PhaseEncodingDirection' in curr_dwi.entities:
        direction_key = 'PhaseEncodingDirection'

    dwi_direction = curr_dwi.entities[direction_key]
    PE[0] = conversion[dwi_direction]

    if related_files and direction_key:
        related_files_suffixes = []
        for curr_related in related_files:
            related_files_suffixes.append(curr_related.entities['suffix'])
            if dwi_direction == get_opposite_pe_direction(curr_related.entities[direction_key]):
                PE[1] = conversion[curr_related.entities[direction_key]]
                topup_suffix[curr_related.entities['suffix']][1] = curr_related.path
            else:
                topup_suffix[curr_related.entities['suffix']][0] = curr_related.path

        if related_files_suffixes.count('epi') > 2 or related_files_suffixes.count('sbref') > 2:
            topup_suffix = {'epi': ['', ''], 'sbref': ['', '']}
            logging.warning("Too many files "
                            "pointing to {}.".format(dwis[0].path))
    else:
        topup = ['', '']
        logging.warning("IntendedFor: No file"
                        " pointing to {}".format(dwis[0].path))

    if len(dwis) == 2:
        if not any(s == '' for s in topup_suffix['sbref']):
            topup = topup_suffix['sbref']
        elif not any(s == '' for s in topup_suffix['epi']):
            topup = topup_suffix['epi']
        else:
            topup = ['', '']
    elif len(dwis) == 1:
        # If one DWI you cannot have a reverse sbref
        # since sbref is a derivate of multi-band dwi
        if topup_suffix['epi'][1] != '':
            topup = topup_suffix['epi']
        elif not any(s == '' for s in topup_suffix['sbref']):
            logging.warning("You have two sbref but "
                            "only one dwi this scheme is not accepted.")
            topup = ['', '']
        else:
            topup = ['', '']
    else:
        print(dwis)
        logging.warning("""
                        BIDS structure unkown.Please send an issue:
                        https://github.com/scilus/scilpy/issues
                        """)
        return {}

    if not any(s == '' for s in topup):
        logging.info("Found rev b0 and b0 images "
                     "to correct for geometrical distorsion")
    elif not topup[1]:
        logging.warning("No rev image found "
                        "to correct for geometrical distorsion")
    elif topup[1]:
        logging.info("Found rev b0 to correct "
                     "for geometrical distorsion")
    else:
        logging.warning("Only found one b0 with same "
                        "PhaseEncodedDirection won't be enough to "
                        "correct for geometrical distorsion")

    # T1 setup
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
            if 'session' in t1.entities:
                if t1.entities['session'] == nSess:
                    t1_nSess.append(t1)
            else:
                t1_nSess.append(t1)

        if len(t1_nSess) == 1:
            t1_path = t1_nSess[0].path
        elif len(t1_nSess) == 0:
            logging.warning('No T1 file found.')
        else:
            t1_paths = [curr_t1.path for curr_t1 in t1_nSess]
            logging.warning('More than one T1 file found.'
                            ' [{}]'.format(','.join(t1_paths)))

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
        List of lists of dwis
    """
    all_dwis = []
    base_dict = {'subject': nSub,
                 'datatype': 'dwi',
                 'extension': 'nii.gz',
                 'suffix': 'dwi'}

    # Get possible directions
    phaseEncodingDirection = [Query.ANY, Query.ANY]
    directions = layout.get_direction(**base_dict)

    directions.sort()

    if not directions and 'PhaseEncodingDirection' in layout.get_entities():
        logging.info("Found no directions")
        directions = [Query.ANY, Query.ANY]
        phaseEncodingDirection = layout.get_PhaseEncodingDirection(**base_dict)
        if len(phaseEncodingDirection) == 1:
            logging.info("Found one phaseEncodingDirection")
            return [[el] for el in layout.get(part=Query.NONE, **base_dict) +
                    layout.get(part='mag', **base_dict)]
        elif len(phaseEncodingDirection) == 0:
            logging.warning("PhaseEncodingDirection exists in this "
                            "dataset, but no DWI was found")
            return []
    elif len(directions) == 1:
        logging.info("Found one direction.")
        return [[el] for el in layout.get(part=Query.NONE, **base_dict) +
                layout.get(part='mag', **base_dict)]
    elif not directions:
        logging.info("Found no directions or PhaseEncodingDirections")
        return [[el] for el in layout.get(part=Query.NONE, **base_dict) +
                layout.get(part='mag', **base_dict)]

    if len(phaseEncodingDirection) > 2 or len(directions) > 2:
        logging.warning("These acquisitions have "
                        "too many encoding directions")
        return []

    all_dwis = layout.get(part=Query.NONE,
                          PhaseEncodingDirection=phaseEncodingDirection[0],
                          direction=directions[0],
                          **base_dict) +\
        layout.get(part='mag',
                   PhaseEncodingDirection=phaseEncodingDirection[0],
                   direction=directions[0],
                   **base_dict)

    all_rev_dwis = layout.get(part=Query.NONE,
                              PhaseEncodingDirection=phaseEncodingDirection[1],
                              direction=directions[1],
                              **base_dict) +\
        layout.get(part='mag',
                   PhaseEncodingDirection=phaseEncodingDirection[1],
                   direction=directions[1],
                   **base_dict)

    all_associated_dwis = []
    logging.info('Number of dwi: {}'.format(len(all_dwis)))
    logging.info('Number of rev_dwi: {}'.format(len(all_rev_dwis)))
    while len(all_dwis) > 0:
        curr_dwi = all_dwis[0]

        curr_association = [curr_dwi]

        # Fake reverse so it can be used to compare with real rev
        rev_curr_entity = curr_dwi.get_entities()

        rev_iter_to_rm = []
        for iter_rev, rev_dwi in enumerate(all_rev_dwis):
            # At this stage, we need to check only direction
            direction = False
            if 'direction' in curr_dwi.entities:
                direction = 'direction'
            elif 'PhaseEncodingDirection' in curr_dwi.entities:
                direction = 'PhaseEncodingDirection'

            if direction:
                rev_curr_entity[direction] = get_opposite_pe_direction(rev_curr_entity[direction])
                if rev_curr_entity == rev_dwi.get_entities():
                    curr_association.append(rev_dwi)
                    rev_iter_to_rm.append(iter_rev)
                else:
                    if rev_curr_entity[direction] == rev_dwi.entities[direction]:
                        # Print difference between entities
                        logging.warning('DWIs {} and {} have opposite phase encoding directions but different entities.'
                                        'Please check their respective json files.'.format(curr_dwi, rev_dwi))

        # drop all rev_dwi used
        logging.info('Checking dwi {}'.format(all_dwis[0]))
        for item_to_remove in rev_iter_to_rm[::-1]:
            logging.info('Found rev_dwi {}'.format(all_rev_dwis[item_to_remove]))
            del all_rev_dwis[item_to_remove]

        # Add to associated list
        if len(curr_association) < 3:
            all_associated_dwis.append(curr_association)
        else:
            logging.warning("These acquisitions have "
                            "too many associated dwis.")
        del all_dwis[0]

    if len(all_rev_dwis):
        for curr_rev_dwi in all_rev_dwis:
            all_associated_dwis.append([curr_rev_dwi])

    return all_associated_dwis


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    coloredlogs.install(level=logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [], args.bids_ignore)
    assert_outputs_exist(parser, args, args.out_json)

    data = []
    bids_indexer = BIDSLayoutIndexer(validate=False,
                                     ignore=_load_bidsignore_(os.path.abspath(args.in_bids),
                                                              args.bids_ignore))
    layout = BIDSLayout(os.path.abspath(args.in_bids), indexer=bids_indexer)

    subjects = layout.get_subjects()
    subjects.sort()

    logging.info("Found {} subject(s)".format(len(subjects)))

    for nSub in subjects:
        mess = 'Validating subject: {}'.format(nSub)
        logging.info("-" * len(mess))
        logging.info(mess)
        dwis = associate_dwis(layout, nSub)

        # Get the data for each run of DWIs
        for dwi in dwis:
            fs_inputs = []
            t1s = []
            if args.fs:
                abs_fs = os.path.abspath(args.fs)

                logging.info("Looking for FS files")
                test_fs_sub_path = os.path.join(abs_fs, 'sub-' + nSub)
                fs_sub_path = ""
                if os.path.exists(test_fs_sub_path):
                    fs_sub_path = test_fs_sub_path
                elif 'session' in dwi[0].entities:
                    nSess = dwi[0].entities['session']
                    test_fs_sub_path = os.path.join(abs_fs,
                                                    'sub-' + nSub + '_ses-' + nSess)
                    if os.path.exists(test_fs_sub_path):
                        fs_sub_path = test_fs_sub_path

                if fs_sub_path:
                    t1_fs = glob(os.path.join(fs_sub_path, 'mri/T1.mgz'))
                    wmparc = glob(os.path.join(fs_sub_path, 'mri/wmparc.mgz'))
                    aparc_aseg = glob(os.path.join(fs_sub_path,
                                                   'mri/aparc+aseg.mgz'))

                    if len(t1_fs) == 1 and len(wmparc) == 1 and len(aparc_aseg) == 1:
                        fs_inputs = [t1_fs[0], wmparc[0], aparc_aseg[0]]
                        logging.info("Found FS files")
                else:
                    logging.info("NOT Found FS files")
            else:
                logging.info("Looking for T1 files")
                t1s = layout.get(subject=nSub,
                                 datatype='anat', extension='nii.gz',
                                 suffix='T1w')
                if t1s:
                    logging.info("Found {} T1 files".format(len(t1s)))

            data.append(get_data(layout,
                                 nSub,
                                 dwi,
                                 t1s,
                                 fs_inputs,
                                 args.readout,
                                 args.clean))

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
