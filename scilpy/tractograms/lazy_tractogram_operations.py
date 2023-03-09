# -*- coding: utf-8 -*-
import logging
import os

import nibabel as nib
import numpy as np
from dipy.io.utils import is_header_compatible
from nibabel.streamlines import LazyTractogram


def lazy_streamlines_count(in_tractogram_path):
    """ Gets the number of streamlines as written in the tractogram header.

    Parameters
    ----------
    in_tractogram_path: str
        Tractogram filepath, must be .trk or .tck.

    Return
    ------
    count: int
        Number of streamlines present in the tractogram.
    """
    _, ext = os.path.splitext(in_tractogram_path)
    if ext == '.trk':
        key = 'nb_streamlines'
    elif ext == '.tck':
        key = 'count'
    else:
        raise IOError('{} is not supported for lazy loading'.format(ext))

    tractogram_file = nib.streamlines.load(in_tractogram_path,
                                           lazy_load=True)
    return tractogram_file.header[key]


def lazy_concatenate(in_tractograms, out_ext):
    """
    Parameters
    ----------
    in_tractograms: list
        List of filenames to concatenate
    out_ext: str
        Output format. Accepting .trk and .tck.

    Returns
    -------
    out_tractogram: Lazy tractogram
        The concatenated data
    header: nibabel header or None
        Depending on the data type.
    """
    def list_generator_from_nib(filenames):
        for in_file in filenames:
            logging.info("Lazy-loading file {}".format(in_file))
            tractogram_file = nib.streamlines.load(in_file, lazy_load=True)
            for s in tractogram_file.streamlines:
                yield s

    # Verifying headers
    # Header will stay None for tck output. Will become a trk header (for
    # trk output) if we find at least one trk input.
    header = None
    for in_file in in_tractograms:
        _, ext = os.path.splitext(in_file)
        if ext == '.trk' and out_ext == '.trk':
            if header is None:
                header = nib.streamlines.load(
                    in_file, lazy_load=True).header
            elif not is_header_compatible(header, in_file):
                logging.warning('Incompatible headers in the list.')

    if out_ext == '.trk' and header is None:
        raise ValueError("No trk file encountered in the input list. "
                         "Result cannot be saved as a .trk.")

    # Now preparing data
    generator = list_generator_from_nib(in_tractograms)
    out_tractogram = LazyTractogram(lambda: generator,
                                    affine_to_rasmm=np.eye(4))
    return out_tractogram, header
