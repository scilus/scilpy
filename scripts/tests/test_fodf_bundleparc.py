import os
import pytest
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


@pytest.mark.ml
def test_help_option(script_runner, monkeypatch):
    ret = script_runner.run('scil_fodf_bundleparc.py', '--help')

    assert ret.success


@pytest.mark.ml
def test_execution(script_runner, monkeypatch):
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_fodf_bundleparc.py', in_fodf, in_mask,
                            '--half')
    assert ret.success


@pytest.mark.ml
def test_execution_100_labels(script_runner, monkeypatch):
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')

    ret = script_runner.run('scil_fodf_bundleparc.py', in_fodf, in_mask,
                            '--nb_pts', '100', '--half')
    assert ret.success
