import argparse
import glob
import nibabel as nib
import os
import random

import numpy as np

from fury import window, actor

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundles',
                   help='List of tractography files supported by nibabel.')
    p.add_argument('--random_coloring', metavar='SEED', type=int,
                   help='Assign a random color to bundles')
    p.add_argument('--subsample', metavar='N', type=int, default=1,
                   help='Only visualize 1 in N streamlines. This is useful' +
                        ' in case tractograms are too heavy')

    add_overwrite_arg(p)
    return p


def random_rgb():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b]) / 255.0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isdir(args.in_bundles):
        bundle_filenames = [os.path.join(args.in_bundles, f)
                            for f in os.listdir(args.in_bundles)]
    else:
        bundle_filenames = glob.glob(args.in_bundles)
        print(bundle_filenames)


    scene = window.Scene()
    for filename in bundle_filenames:
        streamlines = nib.streamlines.load(filename).tractogram.streamlines

        color = None
        if args.random_coloring:
            np.random.seed(int(args.random_coloring))
            color = random_rgb()

        line_actor = actor.streamtube(streamlines[::args.subsample], colors=color)
        scene.add(line_actor)

    showm = window.ShowManager(scene, reset_camera=True)
    showm.initialize()
    showm.start()


if __name__ == '__main__':
    main()
