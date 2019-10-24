#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a json file with DWI, T1 and fmap informations from BIDS folder
"""

import os

import argparse
from bids import BIDSLayout
import json

from scilpy.io.utils import add_overwrite_arg, assert_outputs_exist


def _build_args_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    parser.add_argument("bids_folder",
                        help="Input BIDS folder.")

    parser.add_argument("output_json",
                        help="Output json file.")

    parser.add_argument("--readout", type=float, default=0.062,
                        help="Default total readout time value [%(default)s].")

    add_overwrite_arg(parser)

    return parser


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
        '.' + bf.get_entities()['extension'], '')
    with open(filename + '.json', 'r') as handle:
        return json.load(handle)


def get_dwi_associations(fmaps, bvals, bvecs):
    """ Return DWI associations

    Parameters
    ----------
    fmaps : List of BIDSFile object
        List of field maps

    bvals : List of BIDSFile object
        List of b-value files

    bvecs : List of BIDSFile object
        List of b-vector files

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

    return associations


def get_data(nSub, dwi, t1s, associations, nRun, default_readout):
    """ Return subject data

    Parameters
    ----------
    nSub : String
        Subject name

    dwi : BIDSFile object
        DWI object

    t1s : List of BIDSFile object
        List of T1s associated to the current subject

    associations : Dictionnary
        Dictionnary containing files associated to the DWI

    nRun : int
        Run index

    Returns
    -------
    Dictionnary containing the metadata
    """
    nSess = ''
    if 'session' in dwi.get_entities().keys():
        nSess = dwi.get_entities()['session']

    fmaps = []
    bval_path = ''
    bvec_path = ''
    if dwi.filename in associations.keys():
        if "bval" in associations[dwi.filename].keys():
            bval_path = associations[dwi.filename]['bval']
        if "bvec" in associations[dwi.filename].keys():
            bvec_path = associations[dwi.filename]['bvec']
        if "fmap" in associations[dwi.filename].keys():
            fmaps = associations[dwi.filename]['fmap']

    dwi_PE = 'todo'
    dwi_revPE = -1
    conversion = {"i": "x", "j": "y", "k": "z"}
    dwi_metadata = get_metadata(dwi)
    if 'PhaseEncodingDirection' in dwi_metadata:
        dwi_PE = dwi_metadata['PhaseEncodingDirection']
        dwi_PE = dwi_PE.replace(dwi_PE[0], conversion[dwi_PE[0]])
        if len(dwi_PE) == 1:
            dwi_revPE = dwi_PE + '-'
        else:
            dwi_revPE = dwi_PE[0]

    # Find b0 for topup, take the first one
    revb0_path = ''
    totalreadout = ''
    for nfmap in fmaps:
        nfmap_metadata = get_metadata(nfmap)
        if 'PhaseEncodingDirection' in nfmap_metadata:
            fmap_PE = nfmap_metadata['PhaseEncodingDirection']
            fmap_PE = fmap_PE.replace(fmap_PE[0], conversion[fmap_PE[0]])
            if fmap_PE == dwi_revPE:
                if 'TotalReadoutTime' in dwi_metadata:
                    if 'TotalReadoutTime' in nfmap_metadata:
                        dwi_RT = dwi_metadata['TotalReadoutTime']
                        fmap_RT = nfmap_metadata['TotalReadoutTime']
                        if dwi_RT != fmap_RT and totalreadout == '':
                            totalreadout = 'error_readout'
                            revb0_path = 'error_readout'
                        elif dwi_RT == fmap_RT:
                            revb0_path = nfmap.path
                            totalreadout = dwi_RT
                            break
                else:
                    revb0_path = nfmap.path
                    totalreadout = default_readout

    t1_path = 'todo'
    t1_nSess = []
    for t1 in t1s:
        if 'session' in t1.get_entities().keys() and\
                t1.get_entities()['session'] == nSess:
            t1_nSess.append(t1)
        elif 'session' not in t1.get_entities().keys():
            t1_nSess.append(t1)

    # Take the right T1, if multiple T1s the field must be completed ('todo')
    if len(t1_nSess) == 1:
        t1_path = t1_nSess[0].path
    elif 'run' in dwi.path:
        for t1 in t1_nSess:
            if 'run-' + str(nRun + 1) in t1.path:
                t1_path = t1.path

    return {'subject': nSub,
            'session': nSess,
            'run': nRun,
            't1': t1_path,
            'dwi': dwi.path,
            'bvec': bvec_path,
            'bval': bval_path,
            'rev_b0': revb0_path,
            'DWIPhaseEncodingDir': dwi_PE,
            'TotalReadoutTime': totalreadout}


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_outputs_exist(parser, args, args.output_json)

    data = []
    layout = BIDSLayout(args.bids_folder, index_metadata=False)
    subjects = layout.get_subjects()
    for nSub in subjects:
        dwis = layout.get(subject=nSub,
                          datatype='dwi', extension='nii.gz',
                          suffix='dwi')
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

        # Get associations relatives to DWIs
        associations = get_dwi_associations(fmaps, bvals, bvecs)

        # Get the data for each run of DWIs
        for nRun, dwi in enumerate(dwis):
            data.append(get_data(nSub, dwi, t1s, associations, nRun,
                                 args.readout))

    with open(args.output_json, 'w') as outfile:
        json.dump(data,
                  outfile,
                  indent=4,
                  separators=(',', ': '),
                  sort_keys=True)
        # Add trailing newline for POSIX compatibility
        outfile.write('\n')


if __name__ == '__main__':
    main()
