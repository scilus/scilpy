#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt
from dipy.data import get_sphere
from mpl_toolkits.mplot3d import proj3d

sphere_choices = ['symmetric362', 'symmetric642', 'symmetric724',
                  'repulsion724','repulsion100', 'repulsion200']


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('sphere_name', choices=sphere_choices,
                   help="Sphere to show, amongst:\n{}".format(sphere_choices))

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vertices = get_sphere(args.sphere_name).vertices

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    plt.show()


if __name__ == "__main__":
    main()
