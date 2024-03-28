# -*- coding: utf-8 -*-

"""
Most code here is from https://github.com/fengwangPhysics/matplotlib-chord-diagram.
It was adapted to our specific needs: size of matrix, order of magnitude of
max/min values, alpha for visualisation, etc.
""" # noqa

import math
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

from scilpy.viz.color import get_lookup_table


def polar2xy(r, theta):
    """
    Plot each shell

    Parameters
    ----------
    r: float
        The norm/radius
    theta: float
        The angle in radian
    Return
    ------
    numpy.ndarray
        the x/y coordinates
    """
    return np.array([r*np.cos(theta), r*np.sin(theta)])


def alpha_from_angle(angles_1, threshold, alpha, angles_2=None):
    importance = np.abs(angles_1[0] - angles_1[1])
    if angles_2 is not None:
        importance += np.abs(angles_2[0] - angles_2[1])

    importance = math.degrees(importance)
    if importance > threshold:
        alpha = 0.90
    elif threshold > 0:
        alpha = importance/threshold * alpha
    else:
        alpha = 0.90
    return alpha


def IdeogramArc(start=0, end=60, radius=1.0, width=0.2, ax=None,
                color=(1, 0, 0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    # optimal distance to the control points
    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4./3. * np.tan((end-start) / 4.) * radius
    inner = radius*(1-width)
    verts = [polar2xy(radius, start),
             polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
             polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
             polar2xy(radius, end),
             polar2xy(inner, end),
             polar2xy(inner, end) + polar2xy(opt*(1-width), end-0.515*np.pi),
             polar2xy(inner, start) + polar2xy(opt *
                                               (1-width), start+0.515*np.pi),
             polar2xy(inner, start),
             polar2xy(radius, start), ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.LINETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CLOSEPOLY, ]

    if ax is None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,),
                                  edgecolor=color+(0.4,), lw=0.3, alpha=1.0)
        ax.add_patch(patch)


def ChordArc(start1=0, end1=60, start2=180, end2=240,
             radius=1.0, chordwidth=0.7, ax=None, color=(1, 0, 0),
             angle_threshold=1, alpha=0.1):
    # start, end should be in [0, 360)
    if start1 > end1:
        start1, end1 = end1, start1
    if start2 > end2:
        start2, end2 = end2, start2
    start1 *= np.pi/180.
    end1 *= np.pi/180.
    start2 *= np.pi/180.
    end2 *= np.pi/180.
    opt1 = 4./3. * np.tan((end1-start1) / 4.) * radius
    opt2 = 4./3. * np.tan((end2-start2) / 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [polar2xy(radius, start1),
             polar2xy(radius, start1) + polar2xy(opt1, start1+0.5*np.pi),
             polar2xy(radius, end1) + polar2xy(opt1, end1-0.5*np.pi),
             polar2xy(radius, end1),
             polar2xy(rchord, end1),
             polar2xy(rchord, start2),
             polar2xy(radius, start2),
             polar2xy(radius, start2) + polar2xy(opt2, start2+0.5*np.pi),
             polar2xy(radius, end2) + polar2xy(opt2, end2-0.5*np.pi),
             polar2xy(radius, end2),
             polar2xy(rchord, end2),
             polar2xy(rchord, start1),
             polar2xy(radius, start1), ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4, ]

    if ax is None:
        return verts, codes
    else:
        path = Path(verts, codes)

        alpha_to_set = alpha_from_angle((start1, end1),
                                        threshold=angle_threshold, alpha=alpha,
                                        angles_2=(start2, end2))
        patch = patches.PathPatch(
            path, facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=0.3,
            alpha=alpha_to_set)
        ax.add_patch(patch)


def selfChordArc(start=0, end=60, radius=1.0, chordwidth=0.7, ax=None,
                 color=(1, 0, 0), angle_threshold=1, alpha=0.1):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    opt = 4./3. * np.tan((end-start) / 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [polar2xy(radius, start),
             polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
             polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
             polar2xy(radius, end),
             polar2xy(rchord, end),
             polar2xy(rchord, start),
             polar2xy(radius, start), ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4, ]

    if ax is None:
        return verts, codes
    else:
        path = Path(verts, codes)
        path = Path(verts, codes)
        alpha_to_set = alpha_from_angle((start, end),
                                        threshold=angle_threshold, alpha=alpha)
        patch = patches.PathPatch(
            path, facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=0.3,
            alpha=alpha_to_set)
        ax.add_patch(patch)


def chordDiagram(X, ax, colors=None, width=0.1, pad=2, chordwidth=0.7,
                 angle_threshold=1, alpha=0.1, text_dist=1.1,
                 colormap='plasma'):
    """Plot a chord diagram

    Parameters
    ----------
    X:
        flux data, X[i, j] is the flux from i to j
    ax:
        matplotlib `axes` to show the plot
    colors: optional
        user defined colors in rgb format.
    width: optional
        width/thickness of the ideogram arc
    pad: optional
        gap pad between two neighboring ideogram arcs, unit: degree,
        default: 2 degree
    chordwidth: optional
        position of the control points for the chords,
        controlling the shape of the chords
    """
    # X[i, j]:  i -> j
    x = X.sum(axis=1)  # sum over rows
    ax.set_xlim(-text_dist, text_dist)
    ax.set_ylim(-text_dist, text_dist)

    if colors is None:
        cmap = get_lookup_table(colormap)
        colors = [cmap(i)[0:3] for i in np.linspace(0, 1, len(x))]

    # find position for each start and end
    y = x/np.sum(x).astype(float) * (360 - pad*len(x))

    pos = {}
    arc = []
    nodePos = []
    start = 0
    for i in range(len(x)):
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5*(start+end)

        if 90 <= angle <= 270:
            angle += 180
        if angle >= 360:
            angle -= 360
        nodePos.append(
            tuple(polar2xy(text_dist, 0.5*(start+end)*np.pi/180.)) + (angle,))
        z = (X[i, :]/x[i].astype(float)) * (end - start)
        ids = np.argsort(z)
        z0 = start
        for j in ids:
            pos[(i, j)] = (z0, z0+z[j])
            z0 += z[j]
        start = end + pad

    for i in range(len(x)):
        start, end = arc[i]
        IdeogramArc(start=start, end=end, radius=1.0,
                    ax=ax, color=colors[i], width=width)
        start, end = pos[(i, i)]
        selfChordArc(start, end, radius=1.-width,
                     color=colors[i], chordwidth=chordwidth*0.7, ax=ax,
                     alpha=alpha, angle_threshold=angle_threshold)
        for j in range(i):
            start1, end1 = pos[(i, j)]
            start2, end2 = pos[(j, i)]
            ChordArc(start1, end1, start2, end2,
                     radius=1.-width, color=colors[i], chordwidth=chordwidth,
                     ax=ax, alpha=alpha, angle_threshold=angle_threshold)

    return nodePos
