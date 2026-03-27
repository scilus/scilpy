import json
import os
import shutil
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_bundle_score_many_bundles_one_tractogram',
                            '--help'])
    assert ret.success


def test_score_bundles(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'pft.trk')

    # Pretend we have our segmented bundles on disk
    os.mkdir('./VB_folder')
    shutil.copyfile(in_tractogram, './VB_folder/bundle1.trk')
    shutil.copyfile(in_tractogram, './VB_folder/bundle2.trk')
    fake_path = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')
    json_contents = {
        "bundle1": fake_path,
        "bundle2": fake_path,
        "bundle3": fake_path, # missing VB bundle. Should give score 0.
    }
    with open(os.path.join("config_file.json"), "w") as f:
        json.dump(json_contents, f)

    ret = script_runner.run(['scil_bundle_score_many_bundles_one_tractogram',
                            "config_file.json", "./VB_folder/"])
    assert ret.success


def test_score_bundles_part2(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'pft.trk')

    # Pretend we have our segmented bundles on disk
    shutil.copyfile(in_tractogram, './NC.trk')
    os.mkdir('./segmented_VB')
    shutil.copyfile(in_tractogram, './segmented_VB/bundle1_VS.trk')
    shutil.copyfile(in_tractogram, './segmented_VB/bundle2_VS.trk')
    os.mkdir('./segmented_IB')
    shutil.copyfile(in_tractogram, './segmented_IB/roi1_roi2_VS.trk')
    fake_path = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')
    json_contents = {
        "bundle1": {
            "gt_mask": fake_path,
        },
        "bundle2": {
            "gt_mask": fake_path,
        }
    }
    with open(os.path.join("config_file.json"), "w") as f:
        json.dump(json_contents, f)

    ret = script_runner.run(['scil_bundle_score_many_bundles_one_tractogram',
                            "config_file.json", "./", '--no_bbox_check',
                             '--part2_ROI_segmentation'])

    assert ret.success
