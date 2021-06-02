# -*- coding: utf-8 -*-

from math import cos, radians
import numpy as np


class BinghamFunction(object):
    def __init__(self, f0, mu1, mu2, k1, k2):
        self.f0 = f0  # scaling factor
        self.mu1 = mu1.reshape((1, 3))  # vec3
        self.mu2 = mu2.reshape((1, 3))  # vec3
        self.k1 = k1  # scalar
        self.k2 = k2  # scalar

    def evaluate(self, vertices):
        if vertices.shape[0] != 3:
            vertices = vertices.T
        bu = np.exp(- self.k1 * self.mu1.dot(vertices)**2.
                    - self.k2 * self.mu2.dot(vertices)**2.)
        bu *= self.f0

        return bu.reshape((-1))  # (1, N)


def bingham_fit_peak(sf, peak, sphere, max_angle=6.):
    dot_prod = sphere.vertices.dot(peak)
    min_dot = cos(radians(max_angle))

    p = sphere.vertices[dot_prod > min_dot].astype('float64')
    v = sf[dot_prod > min_dot].reshape((-1, 1)).astype('float64')  # (N, 1)
    x, y, z = (p[:, 0:1], p[:, 1:2], p[:, 2:])

    T = np.zeros((3, 3), dtype='float64')

    T[0, 0] = np.sum(np.multiply(x**2, v))
    T[1, 1] = np.sum(np.multiply(y**2, v))
    T[2, 2] = np.sum(np.multiply(z**2, v))
    T[1, 0] = np.sum(np.multiply(np.multiply(x, y), v))
    T[2, 0] = np.sum(np.multiply(np.multiply(x, z), v))
    T[2, 1] = np.sum(np.multiply(np.multiply(y, z), v))
    T[0, 1] = T[1, 0]
    T[0, 2] = T[2, 0]
    T[1, 2] = T[2, 1]
    T = T / np.sum(v)

    eval, evec = np.linalg.eig(T)
    ordered = np.argsort(eval)
    mu1 = evec[:, ordered[1]].reshape((3, 1))
    mu2 = evec[:, ordered[0]].reshape((3, 1))

    # mu1 = np.array([1., 0., 0.]).reshape((3, 1))
    # mu2 = np.array([0., 1., 0.]).reshape((3, 1))
    f0 = v.max()

    A = np.zeros((len(v), 2), dtype=float)  # (N, 2)
    A[:, 0:1] = p.dot(mu1)**2
    A[:, 1:] = p.dot(mu2)**2

    B = np.log(v / f0)  # (N, 1)
    k = np.abs(np.linalg.inv(A.T.dot(A)).dot(A.T).dot(B))
    k1 = k[0]
    k2 = k[1]

    print('f0: {0}\nmu1:\n {1}\nmu2:\n {2}\nk1: {3}\nk2: {4}'
          .format(f0, mu1, mu2, k1, k2))
    return BinghamFunction(f0, mu1, mu2, k1, k2)
