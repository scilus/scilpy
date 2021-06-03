# -*- coding: utf-8 -*-

from math import cos, radians
import numpy as np

from dipy.direction import peak_directions


class BinghamDistribution(object):
    def __init__(self, f0, mu1, mu2, k1, k2):
        self.f0 = f0  # scaling factor
        self.mu1 = mu1.reshape((1, 3))  # vec3
        self.mu2 = mu2.reshape((1, 3))  # vec3
        self.k1 = k1  # scalar
        self.k2 = k2  # scalar

    def evaluate(self, vertices):
        bu = np.exp(- self.k1 * self.mu1.dot(vertices.T)**2.
                    - self.k2 * self.mu2.dot(vertices.T)**2.)
        bu *= self.f0

        return bu.reshape((-1))  # (1, N)

    def peak_direction(self):
        return np.cross(self.mu1, self.mu2)


class MultiPeakBingham(object):
    def __init__(self):
        self.lobes = []

    def add_lobe(self, bingham_function):
        self.lobes.append(bingham_function)

    def evaluate(self, vertices):
        sf = np.zeros(len(vertices))
        for lobe in self.lobes:
            sf += lobe.evaluate(vertices)
        return sf


def bingham_fit_peak(sf, peak, sphere, max_angle=6., verbose=False):
    # abs for twice the number of pts to fit
    dot_prod = np.abs(sphere.vertices.dot(peak))
    min_dot = cos(radians(max_angle))

    p = sphere.vertices[dot_prod > min_dot].astype('float64')
    v = sf[dot_prod > min_dot].reshape((-1, 1)).astype('float64')  # (N, 1)
    x, y, z = (p[:, 0:1], p[:, 1:2], p[:, 2:])

    # create an orientation matrix to approximate mu0, mu1 and mu2
    T = np.zeros((3, 3), dtype='float64')
    T[0, 0] = np.sum(x**2 * v)
    T[1, 1] = np.sum(y**2 * v)
    T[2, 2] = np.sum(z**2 * v)
    T[1, 0] = np.sum(x * y * v)
    T[2, 0] = np.sum(x * z * v)
    T[2, 1] = np.sum(y * z * v)
    T[0, 1] = T[1, 0]
    T[0, 2] = T[2, 0]
    T[1, 2] = T[2, 1]
    T = T / np.sum(v)

    eval, evec = np.linalg.eig(T)
    ordered = np.argsort(eval)
    mu1 = evec[:, ordered[1]].reshape((3, 1))
    mu2 = evec[:, ordered[0]].reshape((3, 1))
    f0 = v.max()

    A = np.zeros((len(v), 2), dtype=float)  # (N, 2)
    A[:, 0:1] = p.dot(mu1)**2
    A[:, 1:] = p.dot(mu2)**2

    B = np.log(v / f0)  # (N, 1)
    k = np.abs(np.linalg.inv(A.T.dot(A)).dot(A.T).dot(B))
    k1 = k[0]
    k2 = k[1]
    if k[0] > k[1]:
        k1 = k[1]
        k2 = k[0]
        mu2 = evec[:, ordered[1]].reshape((3, 1))
        mu1 = evec[:, ordered[0]].reshape((3, 1))

    if verbose:
        print('f0: {0}\nmu1:\n {1}\nmu2:\n {2}\nk1: {3}\nk2: {4}'
              .format(f0, mu1, mu2, k1, k2))

    return BinghamDistribution(f0, mu1, mu2, k1, k2)


def bingham_fit_multi_peaks(odf, sphere, max_angle=15.):
    """
    Peak extraction followed by Bingham fit for each peak
    """
    multi_peaks_bingham = MultiPeakBingham()
    peaks, _, _ = peak_directions(odf, sphere, relative_peak_threshold=0.,
                                  min_separation_angle=max_angle)

    for peak in peaks:  # peaks can be fitted in parallel
        peak_fit = bingham_fit_peak(odf, peak, sphere,
                                    max_angle=max_angle,
                                    verbose=False)
        multi_peaks_bingham.add_lobe(peak_fit)

    return multi_peaks_bingham


def compute_fiber_density(bingham_lobe, sphere):
    """
    Fiber density (FS) is the integral of the bingham function
    over the sphere.
    """
    delta = []
    for edge in sphere.edges:
        delta.append(np.abs(sphere.vertices[edge[0]] -
                            sphere.vertices[edge[1]]))
    delta = np.mean(delta)
    sf = np.ones(len(sphere.vertices))  # bingham_lobe.evaluate(sphere.vertices)
    # we approximate the area under the area under the curve
    # with a sum of cone elements of radius delta
    # TODO: Replace with better approximation or analytical formula
    fd = np.sum(sf * np.pi * (0.5 * delta)**2 / 3.)
    return fd


def compute_fiber_spread(bingham_lobe, sphere):
    """
    Fiber spread (FS) characterizes the spread of the lobe.
    The higher FS is, the wider the lobe.
    """
    fd = compute_fiber_density(bingham_lobe, sphere)
    afd_max = bingham_lobe.f0
    return fd / afd_max


def compute_complexity(multi_peaks_bingham, sphere):
    pass
