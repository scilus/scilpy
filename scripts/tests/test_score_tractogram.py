import json
import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_score_tractogram.py',
                            '--help')
    assert ret.success


def test_score_bundles(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(get_home(), 'tracking', 'pft.trk')

    json_contents = {
        "example_bundle": {
            "angle": 300,
            "length": [30, 190],
            "gt_mask": os.path.join(get_home(), 'tracking',
                                    'seeding_mask.nii.gz'),
            "endpoints": os.path.join(get_home(), 'tracking',
                                      'interface.nii.gz')
        }
    }
    with open(os.path.join("config_file.json"), "w") as f:
        json.dump(json_contents, f)

    ret = script_runner.run('scil_score_tractogram.py',
                            in_tractogram, "config_file.json",
                            'scoring_results/', '--no_empty',
                            '--use_gt_masks_as_limits_masks')
    assert ret.success
