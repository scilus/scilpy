import json
import os
import shutil
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_score_many_bundles_one_tractogram.py',
                            '--help')
    assert ret.success


def test_score_bundles(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(get_home(), 'tracking', 'pft.trk')

    # Pretend we have our segmented bundles on disk
    shutil.copyfile(in_tractogram, './NC.trk')
    os.mkdir('./segmented_VB')
    shutil.copyfile(in_tractogram, './segmented_VB/bundle1_VS.trk')
    shutil.copyfile(in_tractogram, './segmented_VB/bundle2_VS.trk')
    os.mkdir('./segmented_IB')
    shutil.copyfile(in_tractogram, './segmented_IB/roi1_roi2_VS.trk')

    json_contents = {
        "bundle1": {
            "gt_mask": os.path.join(get_home(), 'tracking',
                                    'seeding_mask.nii.gz'),
        },
        "bundle2": {
            "gt_mask": os.path.join(get_home(), 'tracking',
                                    'seeding_mask.nii.gz'),
        }
    }
    with open(os.path.join("config_file.json"), "w") as f:
        json.dump(json_contents, f)

    ret = script_runner.run('scil_bundle_score_many_bundles_one_tractogram.py',
                            "config_file.json", "./", '--no_bbox_check')

    assert ret.success
