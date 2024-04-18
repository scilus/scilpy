#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# If they already exist, this only takes 5 seconds (check md5sum)
# fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
# tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_viz_bundle.py', '--help')
    assert ret.success

# Tests including VTK do not work on a server without a display
# def test_image_create(script_runner):
#     monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
#     in_vol = os.path.join(
#         SCILPY_HOME, 'bundles', 'fibercup_atlas', 'bundle_all_1mm.nii.gz')

#     in_bundle = os.path.join(
#         SCILPY_HOME, 'bundles', 'fibercup_atlas', 'subj_1', 'bundle_0.trk')

#     ret = script_runner.run('scil_viz_bundle.py',
#                             in_vol, in_bundle, 'out.png')
#     assert ret.success
