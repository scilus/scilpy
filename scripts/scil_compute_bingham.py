#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from fury import window, actor
from dipy.data import get_sphere
from dipy.sims.voxel import multi_tensor_odf

from scilpy.reconst.bingham import (compute_fiber_density,
                                    compute_fiber_spread,
                                    bingham_fit_multi_peaks)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    return p


def visualize(sf, sphere, show_axes=True):
    odf_slicer = actor.odf_slicer(sf[None, None, None, ...],
                                  scale=1., norm=False,
                                  sphere=sphere)
    scene = window.Scene()
    scene.add(odf_slicer)
    if show_axes:
        scene.add(actor.axes())

    window.show(scene)


def visualize_overlay(sf, bingham_fit, sphere):
    scene = window.Scene()
    sf_slicer = actor.odf_slicer(sf[None, None, None, ...],
                                 scale=1., norm=False,
                                 sphere=sphere, opacity=0.4,
                                 colormap=[255, 255, 255])
    scene.add(sf_slicer)

    for lobe in bingham_fit.lobes:
        sf_approx = lobe.evaluate(sphere.vertices)
        approx_slicer = actor.odf_slicer(sf_approx[None, None, None, ...],
                                         scale=1., norm=False,
                                         sphere=sphere, opacity=0.4)
        scene.add(approx_slicer)

    window.show(scene)


def gen_odf(sphere):
    mevals = np.array(([0.0015, 0.0002, 0.0006],
                       [0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    angles = [(0, 0), (60, 0), (60, 60)]
    odf = multi_tensor_odf(sphere.vertices, mevals, angles, [30, 30, 40])
    odf /= odf.max()
    return odf


def mean_relative_error(sf, sf_approx):
    sf[sf < 0.] = 0.
    sf_approx[sf_approx < 0.] = 0.
    error = np.abs(sf - sf_approx)
    error = error[sf > 0.] / sf[sf > 0.]
    return np.mean(error)


def log_mean_rel_error(sf, sf_approx):
    print('Mean relative error:', mean_relative_error(sf, sf_approx))


def main():
    parser = _build_arg_parser()
    sphere = get_sphere('symmetric724').subdivide(2)

    # f0 = 15.
    # k1, k2 = (5., 25.)
    # mu1 = np.array([0.5, 0.5, 0.])
    # mu2 = np.array([0., 1., 0.])

    # bingham = BinghamFunction(f0, mu1, mu2, k1, k2)

    # sf = bingham.evaluate(sphere.vertices)
    # bingham_approx = bingham_fit_peak(sf, np.array([0.0, 0.0, 1.0]),
    #                                   sphere, max_angle=20)
    # sf_approx = bingham_approx.evaluate(sphere.vertices)

    # visualize(sf, sphere)
    # visualize(sf_approx, sphere)

    # log_mean_rel_error(sf, sf_approx)

    odf = gen_odf(sphere)

    bingham_odf_fit = bingham_fit_multi_peaks(odf, sphere)
    visualize_overlay(odf, bingham_odf_fit, sphere)

    lobe = bingham_odf_fit.lobes[0]
    fd = compute_fiber_density(lobe, sphere)
    fs = compute_fiber_spread(lobe, sphere)
    print('fd: ', fd)
    print('fs: ', fs)
    return


if __name__ == '__main__':
    main()
