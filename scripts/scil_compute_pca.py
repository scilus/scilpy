#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute PCA analysis on diffusion metrics. Output returned is all significant principal components
(e.g. presenting eigenvalues > 1) in a connectivity matrix format. This script can take into account all
edges from every subject in a population or only non-zero edges across all subjects.

The script can take directly as input a connectoflow output folder. Simply use the --connectoflow flag. For
other type of folder input, the script expects a single folder containing all matrices for all subjects.
Example:
        |--- in_folder
        |       |--- sub-01_ad.npy
        |       |--- sub-01_md.npy
        |       |--- sub-02_ad.npy
        |       |--- sub-02_md.npy
        |       |--- ...

Output connectivity matrix will be saved next to the other metrics in the input folder. The plots and tables
will be outputted in the designated folder from the <output> argument.

EXAMPLE USAGE:
scil_compute_pca.py input_folder/ output_folder/ --metrics ad fa md rd [...] --list_ids list_ids.txt --common true
"""

# Import required libraries.
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scilpy.io.utils import (load_matrix_in_any_format,
                             save_matrix_in_any_format,
                             add_verbose_arg,
                             add_overwrite_arg,
                             assert_output_dirs_exist_and_empty)


EPILOG = """
[1] Chamberland M, Raven EP, Genc S, Duffy K, Descoteaux M, Parker GD, Tax CMW, Jones DK. Dimensionality 
    reduction of diffusion MRI measures for improved tractometry of the human brain. Neuroimage. 2019 Oct 
    15;200:89-100. doi: 10.1016/j.neuroimage.2019.06.020. Epub 2019 Jun 20. PMID: 31228638; PMCID: PMC6711466.
    """


# Build argument parser.
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EPILOG)

    p.add_argument('in_folder',
                   help='Path to the input folder.')
    p.add_argument('out_folder',
                   help='Path to the output folder to export graphs and tables. \n'
                        '*** Please note, PC connectivity matrix will be outputted in the original input folder'
                        'next to all other metrics ***')
    p.add_argument('--metrics', nargs='+', required=True,
                   help='List of all metrics to include in PCA analysis.')
    p.add_argument('--list_ids', required=True,
                   help='List containing all ids to use in PCA computation.')
    p.add_argument('--common', choices=['true', 'false'], default='true',
                   help='If true, will include only connections found in all subjects of the population (Recommended) '
                        '[True].')
    p.add_argument('--connectoflow', action='store_true',
                   help='If true, script will assume the input folder is a Connectoflow output.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def generate_pca_input(d):
    """
    Function to create PCA input from matrix.
    :param d:       Dictionary containing all matrix for all metrics.
    :return:        Numpy array.
    """
    for a in d.keys():
        mat = np.rollaxis(np.array(d[f'{a}']), axis=1, start=3)
        matrix_shape = mat.shape[1:3]
        n_group = mat.shape[0]
        mat = mat.reshape((np.prod(matrix_shape) * n_group, 1))
        d[f'{a}'] = mat

    out = np.hstack([d[f'{i}'] for i in d.keys()])

    return out


def restore_zero_values(orig, new):
    """
    Function to restore 0 values in a numpy array.
    :param orig:    Original numpy array containing 0 values to restore.
    :param new:     Array in which 0 values need to be restored.
    :return:        Numpy array with data from the new array but zeros from the original array.
    """
    mask = np.copy(orig)
    mask[mask != 0] = 1

    return np.multiply(new, mask)


def autolabel(rects, axs):
    """
    Attach a text label above each bar displaying its height (or bar value).
    :param rects:   Graphical object.
    :param axs:     Axe number.
    :return:
    """
    for rect in rects:
        height = rect.get_height()
        axs.text(rect.get_x() + rect.get_width()/2., height*1.05,
                 '%.3f' % float(height), ha='center', va='bottom')


def extracting_common_cnx(d, ind):
    """
    Function to create a binary mask representing common connections across the population
    containing non-zero values.
    :param d:       Dictionary containing all matrices for all metrics.
    :param ind:     Indice of the key to use to generate the binary mask from the dictionary.
    :return:        Binary mask.
    """
    keys = list(d.keys())
    met = np.copy(d[keys[ind]])

    # Replacing all non-zero values by 1.
    for i in range(0, len(met)):
        met[i][met[i] != 0] = 1

    # Adding all matrices together.
    orig = np.copy(met[0])
    mask = [orig]
    for i in range(1, len(met)):
        orig = np.add(orig, met[i])
        mask.append(orig)

    # Converting resulting values to 0 and 1.
    mask_f = mask[(len(met) - 1)]
    mask_f[mask_f != len(met)] = 0
    mask_f[mask_f == len(met)] = 1

    return mask_f


def apply_binary_mask(d, mask):
    """
    Function to apply a binary mask to all matrices contained in a dictionary.
    :param d:       Dictionary of all matrices from all metrics.
    :param mask:    Binary mask.
    :return:        Dictionary with the same shape as input.
    """
    for a in d.keys():
        for i in range(0, len(d[f'{a}'])):
            d[f'{a}'][i] = np.multiply(d[f'{a}'][i], mask)

    return d


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_output_dirs_exist_and_empty(parser, args, args.out_folder, create_dir=True)

    subjects = open(args.list_ids).read().split()

    if args.connectoflow:
        # Loading all matrix.
        logging.info('Loading all matrices from a Connectoflow output...')
        d = {f'{m}': [load_matrix_in_any_format(f'{args.in_folder}/{a}/Compute_Connectivity/{m}.npy') for a in subjects]
             for m in args.metrics}
    else:
        logging.info('Loading all matrices...')
        d = {f'{m}': [load_matrix_in_any_format(f'{args.in_folder}/{a}_{m}.npy') for a in subjects]
             for m in args.metrics}
        # Assert that all metrics have the same number of subjects.
        nb_sub = [len(d[f'{m}']) for m in args.metrics]
        assert all(x == len(subjects) for x in nb_sub), "Error, the number of matrices for each metric doesn't match" \
                                                        "the number of subject in the id list." \
                                                        "Please validate input folder."

    # Setting individual matrix shape.
    mat_shape = d[f'{args.metrics[0]}'][0].shape

    if args.common == 'true':
        m1 = extracting_common_cnx(d, 0)
        m2 = extracting_common_cnx(d, 1)

        if m1.shape != mat_shape:
            parser.error("Extracted binary mask doesn't match the shape of individual input matrix.")

        if np.sum(m1) != np.sum(m2):
            parser.error("Different binary mask have been generated from 2 different metrics, \n "
                         "please validate input data.")
        else:
            logging.info('Data shows {} common connections across the population.'.format(np.sum(m1)))

        d = apply_binary_mask(d, m1)

    # Creating input structure.
    logging.info('Creating PCA input structure...')
    df = generate_pca_input(d)
    df[df == 0] = 'nan'

    # Standardized the data.
    logging.info('Standardizing data...')
    x = StandardScaler().fit_transform(df)
    df_pca = x[~np.isnan(x).any(axis=1)]
    x = np.nan_to_num(x, nan=0.)

    # Perform the PCA.
    logging.info('Performing PCA...')
    pca = PCA(n_components=len(args.metrics))
    principalcomponents = pca.fit_transform(df_pca)
    principaldf = pd.DataFrame(data=principalcomponents, columns=[f'PC{i}' for i in range(1, len(args.metrics) + 1)])

    # Plot the eigenvalues.
    logging.info('Plotting results...')
    eigenvalues = pca.explained_variance_
    pos = list(range(1, len(args.metrics)+1))
    plt.clf()
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    bar_eig = ax.bar(pos, eigenvalues, align='center', tick_label=principaldf.columns)
    ax.set_xlabel('Principal Components', fontsize=10)
    ax.set_ylabel('Eigenvalues', fontsize=10)
    ax.set_title('Eigenvalues for each principal components.', fontsize=10)
    autolabel(bar_eig, ax)
    plt.savefig(f'{args.out_folder}/eigenvalues.pdf')

    # Plot the explained variance.
    explained_var = pca.explained_variance_ratio_
    plt.clf()
    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    bar_var = ax.bar(pos, explained_var, align='center', tick_label=principaldf.columns)
    ax.set_xlabel('Principal Components', fontsize=10)
    ax.set_ylabel('Explained variance', fontsize=10)
    ax.set_title('Amount of explained variance for all principal components.', fontsize=10)
    autolabel(bar_var, ax)
    plt.savefig(f'{args.out_folder}/explained_variance.pdf')

    # Plot the contribution of each measures to principal component.
    component = pca.components_
    output_component = pd.DataFrame(component, index=principaldf.columns, columns=args.metrics)
    output_component.to_excel(f'{args.out_folder}/loadings.xlsx', index=True, header=True)
    plt.clf()
    plt.cla()
    fig, axs = plt.subplots(2)
    fig.suptitle('Graph of the contribution of each measures to the first and second principal component.', fontsize=10)
    bar_pc1 = axs[0].bar(pos, component[0], align='center', tick_label=args.metrics)
    bar_pc2 = axs[1].bar(pos, component[1], align='center', tick_label=args.metrics)
    autolabel(bar_pc1, axs[0])
    autolabel(bar_pc2, axs[1])
    for ax in axs.flat:
        ax.set(xlabel='Diffusion measures', ylabel='Loadings')
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig(f'{args.out_folder}/contribution.pdf')

    # Extract the derived newly computed measures from the PCA analysis.
    logging.info('Saving matrices for PC with eigenvalues > 1...')
    out = pca.transform(x)
    out = restore_zero_values(x, out)
    out = out.swapaxes(0, 1).reshape(len(args.metrics), len(subjects), mat_shape[0], mat_shape[1])

    # Save matrix for significant components
    nb_pc = eigenvalues[eigenvalues >= 1]
    for i in range(0, len(nb_pc)):
        for s in range(0, len(subjects)):
            if args.connectoflow:
                save_matrix_in_any_format(f'{args.in_folder}/{subjects[s]}/Compute_Connectivity/PC{i+1}.npy',
                                          out[i, s, :, :])
            else:
                save_matrix_in_any_format(f'{args.in_folder}/{subjects[s]}_PC{i+1}.npy',
                                          out[i, s, :, :])


if __name__ == "__main__":
    main()
