import argparse
import nibabel as nib
import random

import numpy as np

from fury import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_seed_map',
                   help='Seed density map')
    p.add_argument('--tractogram', type=str,
                   help='Tractogram coresponding to the seeds')
    p.add_argument('--colormap', type=str, default='bone',
                   help='Name of the map for the density coloring. Can be'
                   ' any colormap that matplotlib offers.')
    p.add_argument('--seed_opacity', type=float, default=0.5,
                   help='Opacity of the contour generated.')
    p.add_argument('--tractogram_opacity', type=float, default=0.5,
                   help='Opacity of the streamlines')

    return p


def random_rgb():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b]) / 255.0


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    seed_map_img = nib.load(args.in_seed_map)
    seed_map_data = seed_map_img.get_fdata()
    seed_map_affine = seed_map_img.affine

    scene = window.Scene()

    values = np.delete(np.unique(seed_map_data), 0)
    cmap = actor.create_colormap(values, name=args.colormap, auto=False)

    cmap = np.concatenate((cmap, np.full((cmap.shape[0], 1), args.seed_opacity)), axis=-1)

    seedroi_actor = actor.contour_from_label(
        seed_map_data, seed_map_affine, color=cmap)

    scene.add(seedroi_actor)

    if args.tractogram:
        tractogram = nib.streamlines.load(args.tractogram).tractogram

        line_actor = actor.streamtube(tractogram.streamlines,
                                      opacity=args.tractogram_opacity)
        scene.add(line_actor)

    showm = window.ShowManager(scene, reset_camera=True)
    showm.initialize()
    showm.start()


if __name__ == '__main__':
    main()
