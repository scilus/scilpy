# -*- coding: utf-8 -*-

from math import cos, radians
from dipy.data import get_sphere
import numpy as np
from scipy.integrate import nquad

from dipy.direction import peak_directions
from dipy.reconst.shm import sh_to_sf_matrix
from scilpy.reconst.utils import get_sh_order_and_fullness


def bingham_fit_sh_volume(data, max_lobes, abs_th=0.,
                          rel_th=0., min_sep_angle=25.):
    order, full_basis = get_sh_order_and_fullness(data.shape[-1])
    shape = data.shape
    out = np.zeros((shape[0], shape[1], shape[2], max_lobes*9))

    sphere = get_sphere('symmetric724')
    B_mat = sh_to_sf_matrix(sphere, order,
                            full_basis=full_basis,
                            return_inv=False)

    for ii in range(shape[0]):
        for jj in range(shape[1]):
            for kk in range(shape[2]):
                print(ii, jj, kk)
                odf = data[ii, jj, kk].dot(B_mat)
                odf[odf < abs_th] = 0.
                if (odf > 0.).any():
                    peaks, _, _ = peak_directions(odf, sphere, rel_th,
                                                  min_sep_angle)
                    n = peaks.shape[0]\
                        if peaks.shape[0] < max_lobes\
                        else max_lobes
                    for nn in range(n):
                        fit = bingham_fit_peak(odf, peaks[nn], sphere)
                        out[ii, jj, kk, nn*9:(nn+1)*9] = fit.get_flatten()

    return out


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

    def get_flatten(self):
        ret = np.zeros((9))
        ret[:3] = self.mu1.reshape((-1))
        ret[3:6] = self.mu2.reshape((-1))
        ret[6:] = np.array([self.f0, self.k1, self.k2])
        return ret


class MultiPeakBingham(object):
    def __init__(self):
        self.lobes = []

    def add_lobe(self, bingham_function):
        self.lobes.append(bingham_function)


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
    print(A)

    # Test that AT.A is invertible for pseudo-inverse
    ATA = A.T.dot(A)
    if np.linalg.matrix_rank(ATA) != ATA.shape[0]:
        return BinghamDistribution(0, np.zeros(3), np.zeros(3), 0, 0)

    B = np.log(v / f0)  # (N, 1)
    k = np.abs(np.linalg.inv(ATA).dot(A.T).dot(B))
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
    r = 1.
    print('surface: ', 4.*np.pi*r**2.)
    print('volume: ', 4./3.*np.pi*r**3.)

    def _integrate_bingham(phi, theta):
        u = np.array([[np.cos(phi) * np.sin(theta),
                       np.sin(phi) * np.sin(theta),
                       np.cos(theta)]])
        return r**2. * np.sin(theta)  # bingham_lobe.evaluate(u)*np.sin(theta)

    volume = nquad(_integrate_bingham, [(0, np.pi * 2), (0, np.pi)])
    return volume[0]


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
