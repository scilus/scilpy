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
    in_tractogram = os.path.join(get_home(), 'tracking',
                                 'pft.trk')
    models = os.path.join(get_home(), 'tracking', 'seeding_mask.nii.gz')
    endpoints = os.path.join(get_home(), 'tracking', 'interface.nii.gz')
    ret = script_runner.run('scil_score_tractogram.py',
                            in_tractogram, models,
                            '--gt_endpoints', endpoints)
    assert ret.success
