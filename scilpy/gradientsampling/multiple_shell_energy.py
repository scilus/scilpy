# -*- coding: utf-8 -*-


def write_multiple_shells(bvecs, nb_shells, nb_points_per_shell, filename):
    """
    Export multiple shells to text file.

    Parameters
    ----------
    bvecs : array-like shape (K, 3)
        vectors
    nb_shells: int
        Number of shells
    nb_points_per_shell: array-like shape (nb_shells, )
        A list of integers containing the number of points on each shell.
    filename : str
        output filename
    """
    datafile = open(filename, 'w')
    datafile.write('#shell-id\tx\ty\tz\n')
    k = 0
    for s in range(nb_shells):
        for n in range(nb_points_per_shell[s]):
            datafile.write("%d\t%f\t%f\t%f\n" %
                           (s, bvecs[k, 0], bvecs[k, 1], bvecs[k, 2]))
            k += 1
    datafile.close()
