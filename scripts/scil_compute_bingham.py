#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
from fury import window, actor
from dipy.data import get_sphere

from scilpy.reconst.bingham import BinghamFunction, bingham_fit_peak


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    return p


def visualize(sf, sphere):
    odf_slicer = actor.odf_slicer(sf[None, None, None, ...],
                                  scale=1., norm=False,
                                  sphere=sphere)
    scene = window.Scene()
    scene.add(odf_slicer)
    scene.add(actor.axes())

    window.show(scene)


def main():
    parser = _build_arg_parser()
    f0 = 1.
    k1, k2 = (5., 5.)
    mu1 = np.array([2., 0., 0.])
    mu2 = np.array([0., 1., 0.])

    bingham = BinghamFunction(f0, mu1, mu2, k1, k2)
    sphere = get_sphere('symmetric724').subdivide(2)

    sf = bingham.evaluate(sphere.vertices)
    bingham_approx = bingham_fit_peak(sf, np.array([0.0, 0.0, 1.0]), sphere)
    sf_approx = bingham_approx.evaluate(sphere.vertices)

    visualize(sf, sphere)
    visualize(sf_approx, sphere)

    print('Mean error:', np.mean(np.abs(sf_approx - sf)))
    return


if __name__ == '__main__':
    main()
