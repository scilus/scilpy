# -*- coding: utf-8 -*-
import os

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.tractograms.lazy_tractogram_operations import \
    lazy_streamlines_count, lazy_concatenate

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])
main_path = os.path.join(SCILPY_HOME, 'tractograms', 'streamline_operations')


def test_lazy_tractogram_count():
    in_file = os.path.join(main_path, 'bundle_4.tck')
    nb = lazy_streamlines_count(in_file)
    assert nb == 10


def test_lazy_concatenate():
    in_file1 = os.path.join(main_path, 'bundle_4.tck')
    in_file2 = os.path.join(main_path, 'bundle_4_cut_endpoints.tck')

    out_trk, out_header = lazy_concatenate([in_file1, in_file2], '.tck')
    assert len(out_trk) == 20
