################################################
# Author: Emmanuel Caruyer <caruyer@gmail.com> #
#                                              #
# Code to generate multiple-shell sampling     #
# schemes, with optimal angular coverage. This #
# implements the method described in Caruyer   #
# et al., MRM 69(6), pp. 1534-1540, 2013.      #
# This software comes with no warranty, etc.   #
################################################
from __future__ import division
from scipy import optimize
import numpy as np


def equality_constraints(vects, *args):
    """
    Spherical equality constraint. Returns 0 if vects lies on the unit 
    sphere.
    
    Parameters
    ----------
    vects : array-like shape (N * 3)
    
    Returns
    -------
    array shape (N,) : Difference between squared vector norms and 1.
    """
    N = int(vects.shape[0] / 3)
    vects = vects.reshape((N, 3))
    return (vects ** 2).sum(1) - 1.0


def grad_equality_constraints(vects, *args):
    """
    Return normals to the surface constraint (wich corresponds to 
    the gradient of the implicit function).

    Parameters
    ----------
    vects : array-like shape (N * 3)

    Returns
    -------
    array shape (N, N * 3). grad[i, j] contains
    $\partial f_i / \partial x_j$
    """
    N = vects.shape[0] / 3
    vects = vects.reshape((N, 3))
    vects = (vects.T / np.sqrt((vects ** 2).sum(1))).T
    grad = np.zeros((N, N * 3))
    for i in range(3):
        grad[:, i * N:(i+1) * N] = np.diag(vects[:, i])
    return grad

    
def f(vects, weight_matrix, alpha=1.0):
    """
    Electrostatic-repulsion objective function. The alpha paramter controls
    the power repulsion (energy varies as $1 / ralpha$).

    Paramters
    ---------
    vects : array-like shape (N * 3,)
    weight_matrix: array-like, shape (N, N)
        The contribution weight of each pair of points.
    alpha : floating-point. controls the power of the repulsion. Default is 1.0

    Returns
    -------
    energy : sum of all interactions between any two vectors.
    """
    epsilon = 1e-9
    N = vects.shape[0] // 3
    vects = vects.reshape((N, 3))
    energy = 0.0
    for i in range(N):
        indices = (np.arange(N) > i)
        diffs = ((vects[indices] - vects[i]) ** 2).sum(1) ** alpha
        sums  = ((vects[indices] + vects[i]) ** 2).sum(1) ** alpha
        energy += (weight_matrix[i, indices] * \
                   (1.0 / (diffs + epsilon) + 1.0 / (sums + epsilon))).sum()
    return energy


def grad_f(vects, weight_matrix, alpha=1.0):
    """
    1st-order derivative of electrostatic-like repulsion energy.

    Parameters
    ----------
    vects : array-like shape (N * 3,)
    weight_matrix: array-like, shape (N, N)
        The contribution weight of each pair of points.
    alpha : floating-point. controls the power of the repulsion. Default is 1.0
    
    Returns
    -------
    grad : gradient of the objective function 
    """
    N = vects.shape[0] // 3
    vects = vects.reshape((N, 3))
    grad = np.zeros((N, 3))
    for i in range(N):
        indices = (np.arange(N) != i)
        diffs = ((vects[indices] - vects[i]) ** 2).sum(1) ** (alpha + 1)
        sums  = ((vects[indices] + vects[i]) ** 2).sum(1) ** (alpha + 1)
        grad[i] += (- 2 * alpha * weight_matrix[i, indices] * \
                    (vects[i] - vects[indices]).T / diffs).sum(1)
        grad[i] += (- 2 * alpha * weight_matrix[i, indices] * \
                    (vects[i] + vects[indices]).T / sums).sum(1)
    grad = grad.reshape(N * 3)
    return grad


def cost(vects, S, Ks, weights):
    """
    Objective function for multiple-shell energy. 

    Parameters
    ----------
    vects : array-like shape (N * 3,)
    S : number of shells
    Ks : list of ints, len(Ks) = S. Number of points per shell.
    weights : array-like, shep (S, S)
        weighting parameter, control coupling between shells and how this
        balances.
    """
    K = np.sum(Ks)
    cost = 0.
    indices = np.cumsum(Ks).tolist()
    indices.insert(0, 0)
    weight_matrix = np.zeros((K, K))
    for s1 in range(S):
        for s2 in range(S):
            weight_matrix[indices[s1]:indices[s1 + 1], 
                          indices[s2]:indices[s2 + 1]] = weights[s1, s2]
    return f(vects, weight_matrix)


def grad_cost(vects, S, Ks, weights):
    """
    gradient of the objective function for multiple shells sampling.

    Parameters
    ----------
    vects : array-like shape (N * 3,)
    S : number of shells
    Ks : list of ints, len(Ks) = S. Number of points per shell.
    weights : array-like, shep (S, S)
        weighting parameter, control coupling between shells and how this
        balances.
    """
    K = int(vects.shape[0] / 3)
    grad = np.zeros(3 * K)
    indices = np.cumsum(Ks).tolist()
    indices.insert(0, 0)
    weight_matrix = np.zeros((K, K))
    for s1 in range(S):
        for s2 in range(S):
            weight_matrix[indices[s1]:indices[s1 + 1], 
                          indices[s2]:indices[s2 + 1]] = weights[s1, s2]

    return grad_f(vects, weight_matrix)


def multiple_shell(nb_shells, nb_points_per_shell, weights, max_iter=100, verbose = 2):
    """
    Creates a set of sampling directions on the desired number of shells.

    Parameters
    ----------
    nb_shells : the number of shells
    nb_points_per_shell : list, shape (nb_shells,)
        A list of integers containing the number of points on each shell.
    weights : array-like, shep (S, S)
        weighting parameter, control coupling between shells and how this
        balances.

    Returns
    -------
    vects : array shape (K, 3) where K is the total number of points
            The points are stored by shell.
    """
    # Total number of points
    K = np.sum(nb_points_per_shell)
    
    # Initialized with random directions
    vects = random_uniform_on_sphere(K)
    vects = vects.reshape(K * 3)
    
    vects = optimize.fmin_slsqp(cost, vects.reshape(K * 3), 
        f_eqcons=equality_constraints, fprime=grad_cost, iter=max_iter, 
        acc=1.0e-9, args=(nb_shells, nb_points_per_shell, weights), iprint = verbose)
    vects = vects.reshape((K, 3))
    vects = (vects.T / np.sqrt((vects ** 2).sum(1))).T
    return vects


def write_multiple_shells(vects, nb_shells, nb_points_per_shell, filename):
    """
    Export multiple shells to text file.

    Parameters
    ----------
    vects : array-like shape (K, 3)
    nb_shells : the number of shells
    nb_points_per_shell : array-like shape (nb_shells, )
        A list of integers containing the number of points on each shell.
    filename : string
    """
    datafile = open(filename, 'w')
    datafile.write('#shell-id\tx\ty\tz\n')
    k = 0
    for s in range(nb_shells):
        for n in range(nb_points_per_shell[s]):
            datafile.write("%d\t%f\t%f\t%f\n" % \
                (s, vects[k,0], vects[k,1], vects[k,2]))
            k += 1
    datafile.close()


def random_uniform_on_sphere(K):
    """
    Creates a set of K pseudo-random unit vectors, following a uniform 
    distribution on the sphere.
    """
    phi = 2 * np.pi * np.random.rand(K)

    r = 2 * np.sqrt(np.random.rand(K))
    theta = 2 * np.arcsin(r / 2)
    
    vects = np.zeros((K, 3))
    vects[:, 0] = np.sin(theta) * np.cos(phi)
    vects[:, 1] = np.sin(theta) * np.sin(phi)
    vects[:, 2] = np.cos(theta)

    return vects


def compute_weights(nb_shells, nb_points_per_shell, shell_groups, alphas):
    """
    Computes the weights array from a set of shell groups to couple, and 
    coupling weights.

    """
    weights = np.zeros((nb_shells, nb_shells))
    for shell_group, alpha in zip(shell_groups, alphas):
        total_nb_points = 0
        for shell_id in shell_group:
            total_nb_points += nb_points_per_shell[shell_id]
        for i in shell_group:
            for j in shell_group:
                weights[i, j] += alpha / total_nb_points**2
    return weights
