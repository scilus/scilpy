import json
import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_segment_and_score.py', '--help')
    assert ret.success


def test_score_bundles(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'tracking', 'pft.trk')

    json_contents = {
        "example_bundle": {
            "angle": 300,
            "length": [30, 190],
            "any_mask": os.path.join(SCILPY_HOME, 'tracking',
                                     'seeding_mask.nii.gz'),
            "gt_mask": os.path.join(SCILPY_HOME, 'tracking',
                                    'seeding_mask.nii.gz'),
            "endpoints": os.path.join(SCILPY_HOME, 'tracking',
                                      'interface.nii.gz')
        }
    }
    with open(os.path.join("config_file.json"), "w") as f:
        json.dump(json_contents, f)

    ret = script_runner.run('scil_tractogram_segment_and_score.py',
                            in_tractogram, "config_file.json",
                            'scoring_tractogram/', '--no_empty',
                            '--use_gt_masks_as_all_masks')
    assert ret.success
