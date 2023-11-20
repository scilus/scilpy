# -*- coding: utf-8 -*-

from scilpy.surfaces.utils import (extract_xform)
from scilpy.tests.arrays import (xform, xform_matrix_ref)


def test_convert_freesurfer_into_polydata():
    pass


def test_flip_LPS():
    pass


def test_extract_xform():

    out = extract_xform(xform)

    assert xform_matrix_ref == out
