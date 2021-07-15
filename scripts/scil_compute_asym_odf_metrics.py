#!/usr/bin/env python3
import argparse
from scilpy.direction.peaks import peak_directions_asym

import numpy as np
from dipy.sims.voxel import multi_tensor, multi_tensor_odf
from dipy.data import get_sphere

from scilpy.reconst.multi_processes import peaks_from_sh

from dipy.viz import window, actor


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh', help='Input SH image.')
    p.add_argument('--out_asym', default='asym_map.nii.gz')
    p.add_argument('--out_odd_power', default='odd_power_map.nii.gz')


def main():
    mevals = np.array([[0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]])

    angles = [(0, 0), (60, 0)]

    fractions = [50, 50]

    sphere = get_sphere('repulsion724')
    sphere = sphere.subdivide(2)

    odf = multi_tensor_odf(sphere.vertices, mevals, angles, fractions)
    odf += np.random.normal(scale=0.001, size=odf.shape)

    # Enables/disables interactive visualization
    interactive = True

    scene = window.Scene()

    odf_actor = actor.odf_slicer(odf[None, None, None, :],
                                 sphere=sphere,
                                 colormap='plasma')
    odf_actor.RotateX(90)

    scene.add(odf_actor)

    peaks, _, _ = peak_directions_asym(odf, sphere)
    peak_actor = actor.peak_slicer(peaks[None, None, None, :], symmetric=False)
    peak_actor.RotateX(90)
    scene.add(peak_actor)

    if interactive:
        window.show(scene)


if __name__ == '__main__':
    main()
