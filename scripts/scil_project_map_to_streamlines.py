#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projects metrics extracted from a map onto the endpoints of streamlines. 

The default options will take data from a nifti image (3D or ND) and 
project it onto the endpoints of streamlines.
"""

import argparse
import logging
from copy import deepcopy

from numpy import asarray, reshape, repeat

from scilpy.io.image import assert_same_resolution, load_img
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_reference_arg,
                             add_verbose_arg)

from scilpy.image.volume_space_management import DataVolume

from dipy.io.streamline import save_tractogram, StatefulTractogram

def project_metric_to_streamlines(sft, metric, endpoints_only=False):
    """
    Projects a metric onto the points of streamlines.

    Parameters
    ----------
    sft: StatefulTractogram
        Input tractogram.
    metric: DataVolume
        Input metric.

    Optional:
    ---------
    endpoints_only: bool
        If True, will only project the metric onto the endpoints of the 
        streamlines (all values along streamlines set to zero). If False, 
        will project the metric onto all points of the streamlines.
        
    Returns
    -------
    streamline_data: 
        metric projected to each point of the streamlines.
    """
    if len(metric.data.shape) == 4:
        dimension = metric.data.shape[3]
    else:
        dimension = 1

    streamline_data = []
    if endpoints_only:
        for s in sft.streamlines:
            p1_data = metric.get_value_at_coordinate(s[0][0], s[0][1], s[0][2], space=sft.space, origin=sft.origin)
            p2_data = metric.get_value_at_coordinate(s[-1][0], s[-1][1], s[-1][2], space=sft.space, origin=sft.origin)
            thisstreamline_data = []
            for p in s:
                thisstreamline_data.append(asarray(repeat(0, p1_data.shape[0])))

            thisstreamline_data[0] = p1_data
            thisstreamline_data[-1] = p2_data
            thisstreamline_data = asarray(thisstreamline_data)

            streamline_data.append(reshape(thisstreamline_data, (len(thisstreamline_data), dimension)))
    else:
        for s in sft.streamlines:
            thisstreamline_data = []
            for p in s:
                thisstreamline_data.append(metric.get_value_at_coordinate(p[0], p[1], p[2], space=sft.space, origin=sft.origin))

            streamline_data.append(reshape(thisstreamline_data, (len(thisstreamline_data), dimension)))

    return streamline_data
    
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    # Mandatory arguments input and output tractogram must be in trk format
    p.add_argument('in_tractogram', 
                   help='Fiber bundle file.')
    p.add_argument('in_metric',
                   help='Nifti metric to project onto streamlines.')
    p.add_argument('out_tractogram', 
                   help='Output file.')

    # Optional arguments
    p.add_argument('--trilinear', action='store_true',
                   help='If set, will use trilinear interpolation \n'
                        'else will use nearest neighbor interpolation \n'
                        'by default.')
    
    p.add_argument('--endpoints_only', action='store_true',
                     help='If set, will only project the metric onto the endpoints \n'
                            'of the streamlines (all other values along streamlines set to zero). \n'
                            'If not set, will project the metric onto all points of the streamlines.')

    p.add_argument('--dpp_name', default='metric',
                   help='Name of the data_per_point to be saved in the output tractogram. \n'
                        '(Default: %(default)s)')
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_metric])
    
    assert_outputs_exist(parser, args, [args.out_tractogram])

    logging.debug("Loading the tractogram...")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()

    if len(sft.streamlines) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return
    
    logging.debug("Loading the metric...")
    metric_img,metric_dtype = load_img(args.in_metric)
    assert_same_resolution((args.in_tractogram, args.in_metric))
    metric_data = metric_img.get_fdata(caching='unchanged', dtype=float)
    metric_res = metric_img.header.get_zooms()[:3]
    
    if args.trilinear:
        interp = "trilinear" 
    else:
        interp = "nearest"
    
    metric = DataVolume(metric_data, metric_res, interp)
    
    logging.debug("Projecting metric onto streamlines")
    streamline_data = project_metric_to_streamlines(sft, metric, 
                            endpoints_only=args.endpoints_only)

    logging.debug("Saving the tractogram...")
    data_per_point = {}
    data_per_point[args.dpp_name] = streamline_data
    out_sft = StatefulTractogram(sft.streamlines, metric_img,
                                 sft.space, sft.origin, data_per_point=data_per_point)
    save_tractogram(out_sft, args.out_tractogram)
    print(out_sft.data_per_point[args.dpp_name][0][0])
    print(out_sft.data_per_point[args.dpp_name][0][-1])

if __name__ == '__main__':
    main()
