#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bids_json.zip'])
tmp_dir = tempfile.TemporaryDirectory()

conversion = {"LR": "i",
              "RL": "i-",
              "AP": "j",
              "PA": "j-"}


def generate_fake_metadata_json(
    intended_for=None, phase_dir=None, readout=None, **kwargs):

    base_dict = {}
    if intended_for:
        base_dict["IntendedFor"] = intended_for
    if phase_dir:
        base_dict["PhaseEncodingDirection"] = conversion[phase_dir]
    if readout:
        base_dict["TotalReadoutTime"] = readout

    return {
        **base_dict,
        **kwargs
    }


def generate_image_packet(
    data,directory, prefix,
    intended_for=None,
    phase_dir=None,
    readout=None,
    **kwargs):

    nib.save(nib.Nifti1Image(data, np.eye(4)),
             os.path.join(directory, "{}.nii.gz".format(prefix)))
    metadata = generate_fake_metadata_json(
        intended_for, phase_dir, readout, **kwargs)
    if metadata:
        with open(os.path.join(directory, "{}.json".format(prefix)), "w+") as f:
            json.dump(metadata, f)


def generate_fake_bids_structure(
    output_dir,
    n_subjects,
    n_sessions=1,
    gen_anat_t1=False,
    gen_epi=False,
    gen_fieldmap=False,
    gen_rev_dwi=False,
    gen_gre_mag=False,
    gen_gre_phase=False,
    gen_sbref=False,
    gen_rev_sbref=False,
    split_gre_mag=False,
    split_gre_phase=False,
    complex_dwi=False,
    complex_sbref=False,
    complex_rev_dwi=False
):
    if gen_gre_mag and not (gen_fieldmap or gen_gre_phase):
        raise ValueError("GRE magnitude requires either a fieldmap "
                         "or a phasediff to be generated")

    if split_gre_phase and not split_gre_mag:
        raise ValueError("Splitted GRE phase images are only "
                         "usable with split magnitudes")

    structure_dir = tempfile.mkdtemp(prefix="test_bids", dir=output_dir)
    data_3d, data_dwi = np.empty((10, 10, 10)), np.empty((10, 10, 10, 10))
    base_metadata = {
        "RepetitionTime": 8.4,
        "FlipAngle": 90}
    dwi_metadata = {
        **base_metadata,
        "EchoTime": 0.09,
        "EffectiveEchoSpacing": 0.0003927297862}
    split_metadata = {
        **base_metadata,
        "EchoTime1": 0.00519,
        "EchoTime2": 0.00765}

    for sub_id in range(1, n_subjects + 1):
        sub_tag = "sub-{}".format(sub_id)
        for ses_id in range(1, n_sessions + 1):
            ses_tag = "ses-{}".format(ses_id)
            ses_dir = os.path.join(structure_dir, sub_tag, ses_tag)
            prefix = "{}_{}".format(sub_tag, ses_tag)
            relpath = os.path.join(ses_tag, "dwi")

            intended_for = []
            dwi_dir = os.path.join(ses_dir, "dwi")
            os.makedirs(dwi_dir)

            if complex_dwi:
                generate_image_packet(
                    data_dwi, dwi_dir,
                    "{}_dir-AP_part-mag_dwi".format(prefix),
                    phase_dir="AP",
                    readout=0.069,
                    **dwi_metadata)
                generate_image_packet(
                    data_dwi, dwi_dir,
                    "{}_dir-AP_part-phase_dwi".format(prefix),
                    phase_dir="AP",
                    readout=0.069,
                    **dwi_metadata)

                dwi_name = "{}_dir-AP_part-mag_dwi.nii.gz".format(prefix)
                intended_for.append(os.path.join(relpath, dwi_name))
            else:
                generate_image_packet(
                    data_dwi, dwi_dir,
                    "{}_dir-AP_dwi".format(prefix),
                    phase_dir="AP",
                    readout=0.069,
                    **dwi_metadata)

                dwi_name = "{}_dir-AP_dwi.nii.gz".format(prefix)
                intended_for.append(os.path.join(relpath, dwi_name))

            np.savetxt(
                os.path.join(dwi_dir, "{}_dir-AP_dwi.bval".format(prefix)),
                np.ones((10,)) * 1000)
            np.savetxt(
                os.path.join(dwi_dir, "{}_dir-AP_dwi.bvec".format(prefix)),
                np.ones((3, 10)))

            if gen_rev_dwi:
                if complex_rev_dwi:
                    generate_image_packet(
                        data_dwi, dwi_dir,
                        "{}_dir-PA_part-mag_dwi".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        readout=0.069,
                        **dwi_metadata)
                    generate_image_packet(
                        data_dwi, dwi_dir,
                        "{}_dir-PA_part-phase_dwi".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        readout=0.069,
                        **dwi_metadata)

                    dwi_name = "{}_dir-PA_part-mag_dwi.nii.gz".format(prefix)
                    intended_for.append(os.path.join(relpath, dwi_name))
                else:
                    generate_image_packet(
                        data_dwi, dwi_dir,
                        "{}_dir-PA_dwi".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        readout=0.069,
                        **dwi_metadata)

                    dwi_name = "{}_dir-PA_dwi.nii.gz".format(prefix)
                    intended_for.append(os.path.join(relpath, dwi_name))

                np.savetxt(
                    os.path.join(dwi_dir, "{}_dir-PA_dwi.bval".format(prefix)),
                    np.ones((10,)) * 1000)
                np.savetxt(
                    os.path.join(dwi_dir, "{}_dir-PA_dwi.bvec".format(prefix)),
                    np.ones((3, 10)) / np.sqrt(3))

            if gen_sbref:
                if complex_sbref:
                    generate_image_packet(
                        data_3d, dwi_dir,
                        "{}_dir-AP_part-mag_sbref".format(prefix),
                        intended_for,
                        phase_dir="AP",
                        readout=0.069)
                    generate_image_packet(
                        data_3d, dwi_dir,
                        "{}_dir-AP_part-phase_sbref".format(prefix),
                        intended_for,
                        phase_dir="AP",
                        readout=0.069)
                else:
                    generate_image_packet(
                        data_3d, dwi_dir,
                        "{}_dir-AP_sbref".format(prefix),
                        intended_for,
                        phase_dir="AP",
                        readout=0.069)

                if gen_rev_sbref:
                    if complex_sbref:
                        generate_image_packet(
                            data_3d, dwi_dir,
                            "{}_dir-PA_part-mag_sbref".format(prefix),
                            intended_for,
                            phase_dir="PA",
                            readout=0.069)
                        generate_image_packet(
                            data_3d, dwi_dir,
                            "{}_dir-PA_part-phase_sbref".format(prefix),
                            intended_for,
                            phase_dir="PA",
                            readout=0.069)
                    else:
                        generate_image_packet(
                            data_3d, dwi_dir,
                            "{}_dir-PA_sbref".format(prefix),
                            intended_for,
                            phase_dir="PA",
                            readout=0.069)

            fmap_dir = os.path.join(ses_dir, "fmap")
            if gen_epi:
                os.makedirs(fmap_dir, exist_ok=True)
                generate_image_packet(
                    data_3d, fmap_dir,
                    "{}_dir-PA_epi".format(prefix),
                    intended_for,
                    phase_dir="PA",
                    readout=0.069)

            if gen_fieldmap:
                os.makedirs(fmap_dir, exist_ok=True)
                generate_image_packet(
                    data_3d, fmap_dir,
                    "{}_dir-PA_fieldmap".format(prefix),
                    intended_for,
                    phase_dir="PA",
                    readout=0.069)

            if gen_gre_mag:
                os.makedirs(fmap_dir, exist_ok=True)
                if split_gre_mag:
                    generate_image_packet(
                        data_3d, fmap_dir,
                        "{}_dir-PA_magnitude1".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        **split_metadata)
                    generate_image_packet(
                        data_3d, fmap_dir,
                        "{}_dir-PA_magnitude2".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        **split_metadata)
                else:
                    generate_image_packet(
                        data_3d, fmap_dir,
                        "{}_dir-PA_magnitude".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        readout=0.069)

            if gen_gre_phase:
                os.makedirs(fmap_dir, exist_ok=True)
                if split_gre_phase:
                    generate_image_packet(
                        data_3d, fmap_dir,
                        "{}_dir-PA_phase1".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        **split_metadata)
                    generate_image_packet(
                        data_3d, fmap_dir,
                        "{}_dir-PA_phase2".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        **split_metadata)
                else:
                    generate_image_packet(
                        data_3d, fmap_dir,
                        "{}_dir-PA_phasediff".format(prefix),
                        intended_for,
                        phase_dir="PA",
                        readout=0.069)

            anat_dir = os.path.join(ses_dir, "anat")
            if gen_anat_t1:
                os.makedirs(anat_dir)
                generate_image_packet(
                    data_3d, anat_dir, "{}_T1w".format(prefix))

    dataset_description_file = os.path.join(
        structure_dir, "dataset_description.json")
    with open(dataset_description_file, "w+") as f:
        json.dump({"Name": "Test dataset", "BIDSVersion": "1.8.0"}, f)

    return structure_dir


def compare_jsons(json_output, test_dir):  
    # Open test json file
    with open(os.path.join(test_dir, json_output), 'r') as f:
        test_json = json.load(f)[0]

    # Clean test_json
    for key, value in test_json.items():
        if isinstance(value, str):
            test_json[key] = value.replace(test_dir + os.path.sep,'')

    # Open correct json file
    result_json = os.path.join(SCILPY_HOME, 'bids_json', json_output.replace('test', 'result'))

    with open(result_json, 'r') as f:
        result = json.load(f)[0]

    if not sorted(test_json.items()) == sorted(result.items()):
        print(test_json)
        print(result)

    # Compare json files
    return sorted(test_json.items()) == sorted(result.items())


def test_help_option(script_runner):
    ret = script_runner.run('scil_bids_validate.py', '--help')
    assert ret.success



@pytest.mark.parametrize(
    "dwi_is_complex,json_output",
    [(False, 'test_real_dwi_epi.json'),
     (True, 'test_complex_dwi_epi.json')]
)
def test_bids_epi(tmpdir, script_runner, dwi_is_complex, json_output):

    test_dir = generate_fake_bids_structure(
        tmpdir, 1, 1,
        gen_anat_t1=True,
        gen_epi=True,
        complex_dwi=dwi_is_complex)

    ret = script_runner.run(
        'scil_bids_validate.py',
        test_dir,
        os.path.join(test_dir, json_output),
        '-f', '-v')

    if ret.success:
        assert compare_jsons(json_output, test_dir)
    else:
        assert False


@pytest.mark.parametrize(
    "dwi_is_complex,sbref_is_complex,json_output",
    [(False, False, 'test_real_dwi_real_sbref.json'),
     (True, True, 'test_complex_dwi_complex_sbref.json')]
)
def test_bids_sbref(
    tmpdir, script_runner, dwi_is_complex, sbref_is_complex, json_output):
    test_dir = generate_fake_bids_structure(
        tmpdir, 1, 1,
        gen_anat_t1=True,
        gen_sbref=True,
        gen_epi=True,
        complex_dwi=dwi_is_complex,
        complex_sbref=sbref_is_complex)

    ret = script_runner.run(
        'scil_bids_validate.py',
        test_dir,
        os.path.join(test_dir, json_output),
        '-f', '-v')

    if ret.success:
        assert compare_jsons(json_output, test_dir)
    else:
        assert False


@pytest.mark.parametrize(
    "dwi_is_complex,rev_is_complex,json_output",
    [(False, False, 'test_real_dwi_real_rev_dwi.json'),
     (True, True, 'test_complex_dwi_complex_rev_dwi.json')]
)
def test_bids_rev_dwi(
    tmpdir, script_runner, dwi_is_complex, rev_is_complex, json_output):
    test_dir = generate_fake_bids_structure(
        tmpdir, 1, 1,
        gen_anat_t1=True,
        gen_rev_dwi=True,
        complex_dwi=dwi_is_complex,
        complex_rev_dwi=rev_is_complex)

    ret = script_runner.run(
        'scil_bids_validate.py',
        test_dir,
        os.path.join(test_dir, json_output),
        '-f', '-v')

    if ret.success:
        assert compare_jsons(json_output, test_dir)
    else:
        assert False


@pytest.mark.parametrize(
    "dwi_is_complex,rev_is_complex,json_output",
    [(False, False, 'test_real_dwi_real_rev_dwi_sbref.json'),
     (True, True, 'test_complex_dwi_complex_rev_dwi_sbref.json')]
)
def test_bids_rev_dwi_sbref(
    tmpdir, script_runner, dwi_is_complex, rev_is_complex, json_output):
    test_dir = generate_fake_bids_structure(
        tmpdir, 1, 1,
        gen_anat_t1=True,
        gen_rev_dwi=True,
        gen_sbref=True,
        gen_rev_sbref=True,
        complex_dwi=dwi_is_complex,
        complex_rev_dwi=rev_is_complex)

    ret = script_runner.run(
        'scil_bids_validate.py',
        test_dir,
        os.path.join(test_dir, json_output),
        '-f', '-v')

    if ret.success:
        assert compare_jsons(json_output, test_dir)
    else:
        assert False
