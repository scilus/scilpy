#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs an operation on data per point from streamlines 
resulting in data per streamline.  The supported
operations are:

If the input is singular per point then:
mean: mean across points per streamline
sum: sum across points per streamline
min: min value across points per streamline
max: max value across points per streamline

endpoints_only is set, min/max/mean/sum will only 
be calculated using the streamline endpoints

If the input is multivalued per point then: 
mean: mean calculated per point for each streamline
sum: sum calculated per point for each streamline
min: min value calculated per point for each streamline
max: max value calculated per point for each streamline

endpoints_only is set, min/max/mean/sum will only
be calculated at the streamline endpoints

Endpoint only operations:
correlation: correlation calculated between arrays extracted from
streamline endpoints (data must be multivalued per point)
"""

import argparse
import logging

from copy import deepcopy

from dipy.io.streamline import save_tractogram, StatefulTractogram
import numpy as np
from numpy import asarray, reshape

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_bbox_arg,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

def perform_operation_per_point(op_name, sft, dpp_name='metric', endpoints_only=False):
    """Peforms an operation per point for all streamlines.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per point (4D) and
        returns a list of streamline data per point.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.
    endpoints_only: bool
        If True, will only perform operation on endpoints
    
    Returns
    -------
    new_sft: StatefulTractogram
        sft with data per streamline resulting from the operation.
    """
    # Performing operation
    call_op = OPERATIONS[op_name]
    if endpoints_only:
        new_data_per_point = []
        for s in sft.data_per_point[dpp_name]:
            this_data_per_point = []
            for p in s:
                this_data_per_point.append(asarray(0.))
            this_data_per_point[0] = call_op(s[0])
            this_data_per_point[-1] = call_op(s[-1])
            new_data_per_point.append(reshape(this_data_per_point, (len(this_data_per_point), 1)))
    else:
        new_data_per_point = []
        for s in sft.data_per_point[dpp_name]:
            this_data_per_point = []
            for p in s:
                this_data_per_point.append(call_op(p))
            new_data_per_point.append(reshape(this_data_per_point, (len(this_data_per_point), 1)))

    # Extracting streamlines
    return new_data_per_point

def perform_operation_per_streamline(op_name, sft, dpp_name='metric', endpoints_only=False):
    """Peforms an operation across points for each streamline.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per streamline and
        returns a list of data per streamline.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.
    endpoints_only: bool
        If True, will only perform operation on endpoints
    
    Returns
    -------
    new_sft: StatefulTractogram
        sft with data per streamline resulting from the operation.
    """
    # Performing operation
    call_op = OPERATIONS[op_name]
    if endpoints_only:
        new_data_per_streamline = []
        for s in sft.data_per_point[dpp_name]:
            new_data_per_streamline.append(call_op([s[0], s[-1]]))
    else:
        new_data_per_streamline = []
        for s in sft.data_per_point[dpp_name]:
            new_data_per_streamline.append(call_op(s))

    return new_data_per_streamline

def perform_operation_across_endpoints(op_name, sft, dpp_name='metric'):
    """Peforms an operation across endpoints for each streamline.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per streamline and
        returns a list of data per streamline.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.
    
    Returns
    -------
    new_sft: StatefulTractogram
        sft with data per streamline resulting from the operation.
    """
    # Performing operation
    call_op = OPERATIONS[op_name]
    new_data_per_streamline = []
    for s in sft.data_per_point[dpp_name]:
        new_data_per_streamline.append(call_op(s[0], s[-1])[0,1])

    return new_data_per_streamline

def mean(array):
    return np.mean(array)

def sum(array):
    return np.sum(array)

def min(array):
    return np.min(array)

def max(array):
    return np.max(array)

def correlation(array1, array2):
    return np.corrcoef(array1, array2)

OPERATIONS = {
    'mean': mean,
    'sum': sum,
    'min': min,
    'max': max,
    'correlation': correlation,
}

def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation', metavar='OPERATION',
                   choices=['mean', 'sum', 'min',
                            'max', 'correlation'],
                   help='The type of operation to be performed on the '
                        'streamlines. Must\nbe one of the following: '
                        '%(choices)s.')
    p.add_argument('in_tractogram', metavar='INPUT_FILE',
                   help='Input tractogram containing streamlines and '
                        'metadata.')
    p.add_argument('out_tractogram', metavar='OUTPUT_FILE',
                   help='The file where the remaining streamlines '
                        'are saved.')
    
    p.add_argument('--endpoints_only', action='store_true', default=False,
                        help='If set, will only perform operation on endpoints \n'
                        'If not set, will perform operation on all streamline points.')
    p.add_argument('--dpp_name', default='metric',
                   help='Name of the data_per_point for operation to be '
                        'performed on. (Default: %(default)s)')
    
    p.add_argument('--output_dpp_name', default='metric_math',
                   help='Name of the resulting data_per_point to be saved in the output '
                        'tractogram. (Default: %(default)s)')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_bbox_arg(p)

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    # Load the input files.
    logging.info("Loading file {}".format(args.in_tractogram))
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    if len(sft.streamlines) == 0:
        logging.info("Input tractogram contains no streamlines. Exiting.")
        return

    # Check to see if the data per point exists.
    if args.dpp_name not in sft.data_per_point:
        logging.info('Data per point {} not found in input tractogram.'
                         .format(args.dpp_name))
        return
    
    # check size of first streamline data_per_point
    if sft.data_per_point[args.dpp_name][0].shape[1] == 1:
        is_singular = True
    else:
        is_singular = False

    if args.operation == 'correlation' and is_singular:
        logging.info('Correlation operation requires multivalued data per '
                         'point. Exiting.')
        return
    
    # Perform the requested operation.
    if not is_singular:
        if args.operation == 'correlation':
            logging.info('Performing {}} across endpoint data')
            new_data_per_streamline = perform_operation_across_endpoints(args.operation,sft, args.dpp_name)
        
            # Adding data per streamline to new_sft
            new_sft = StatefulTractogram(sft.streamlines, sft.space_attributes,
                                         sft.space, sft.origin,
                                         data_per_streamline={args.output_dpp_name: new_data_per_streamline})
        else:
            # Results in new data per point
            logging.info('Performing {} on data from each streamine point.'.format(args.operation))
            new_data_per_point = perform_operation_per_point(
                args.operation, sft, args.dpp_name, args.endpoints_only)

            # Adding data per point to new_sft
            new_sft = StatefulTractogram(sft.streamlines, sft.space_attributes,
                                         sft.space, sft.origin,
                                         data_per_point={args.output_dpp_name: new_data_per_point})
    else:
        # Results in new data per streamline
        logging.info('Performing {} across each streamline.'.format(args.operation))
        new_data_per_streamline = perform_operation_per_streamline(
            args.operation, sft, args.output_dpp_name, args.endpoints_only)
        
        # Adding data per streamline to new_sft
        new_sft = StatefulTractogram(sft.streamlines, sft.space_attributes,
                                     sft.space, sft.origin,
                                     data_per_streamline={args.dpp_name: new_data_per_streamline})

    if len(new_sft) == 0 and not args.save_empty:
        logging.info("Empty resulting tractogram. Not saving results.")
        return

    # Save the new streamlines (and metadata)
    logging.info('Saving {} streamlines to {}.'.format(len(new_sft),
                                                       args.out_tractogram))
    save_tractogram(new_sft, args.out_tractogram,
                    bbox_valid_check=args.bbox_check)

if __name__ == "__main__":
    main()
