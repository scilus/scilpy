#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean a bundle (inliers/outliers) using hiearchical clustering.
http://archive.ismrm.org/2015/2844.html

If spurious streamlines are dense, it is possible they will not be recognized
as outliers. Manual cleaning may be required to overcome this limitation.
"""

import argparse
import json
import logging

from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_verbose_arg,
                             add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             check_tracts_same_format)
from scilpy.tractanalysis.bundle_operations import remove_outliers


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Fiber bundle file to remove outliers from.')
    p.add_argument('out_bundle',
                   help='Fiber bundle without outliers.')
    p.add_argument('--remaining_bundle',
                   help='Removed outliers.')
    p.add_argument('--alpha', type=float, default=0.6,
                   help='Percent of the length of the tree that clusters '
                   'of individual streamlines will be pruned. [%(default)s]')
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')

    add_json_args(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundle, args.reference)
    assert_outputs_exist(parser, args, args.out_bundle, args.remaining_bundle)
    if args.alpha <= 0 or args.alpha > 1:
        parser.error('--alpha should be ]0, 1]')

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    if len(sft) == 0:
        logging.warning("Bundle file contains no streamline")
        return

    check_tracts_same_format(parser, [args.in_bundle, args.out_bundle,
                                      args.remaining_bundle])
    outliers, inliers = remove_outliers(sft.streamlines, args.alpha)

    inliers_sft = sft[inliers]
    outliers_sfts = sft[outliers]

    if len(inliers) == 0:
        logging.warning("All streamlines are considered outliers."
                        "Please lower the --alpha parameter")
    else:
        save_tractogram(inliers_sft, args.out_bundle)

    if args.display_counts:
        sc_bf = len(sft.streamlines)
        sc_af = len(inliers_sft.streamlines)
        print(json.dumps({'streamline_count_before_filtering': int(sc_bf),
                         'streamline_count_after_filtering': int(sc_af)},
                         indent=args.indent))

    if len(outliers) == 0:
        logging.warning("No outlier found. Please raise the --alpha parameter")
    elif args.remaining_bundle:
        save_tractogram(outliers_sfts, args.remaining_bundle)


if __name__ == '__main__':
    main()
