#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute PCA analysis on diffusion metrics. Output returned is all
significant principal components (e.g. presenting eigenvalues > 1) in a
connectivity matrix format. This script can take into account all edges from
every subject in a population or only non-zero edges across all subjects.

The script can take directly as input a connectoflow output folder. Simply use
the --input_connectoflow flag. For other type of folder input, the script
expects a single folder containing all matrices for all subjects.
Example:
        [in_folder]
        |--- sub-01_ad.npy
        |--- sub-01_md.npy
        |--- sub-02_ad.npy
        |--- sub-02_md.npy
        |--- ...

The plots, tables and principal components matrices will be outputted in the
designated folder from the <out_folder> argument. If you want to move back your
principal components matrices in your connectoflow output, you can use a
similar bash command for all principal components:
for sub in `cat list_id.txt`;
do
    cp out_folder/${sub}_PC1.npy connectoflow_output/$sub/Compute_Connectivity/
done

Interpretation of resulting principal components can be done by evaluating the
loadings values for each metrics. A value near 0 means that this metric doesn't
contribute to this specific component whereas high positive or negative values
mean a larger contribution. Components can then be labeled based on which
metric contributes the highest. For example, a principal component showing a
high loading for afd_fixel and near 0 loading for all other metrics can be
interpreted as axonal density (see Gagnon et al. 2022 for this specific example
or ref [3] for an introduction to PCA).

EXAMPLE USAGE:
scil_connectivity_compute_pca.py input_folder/ output_folder/
    --metrics ad fa md rd [...] --list_ids list_ids.txt
"""

# Import required libraries.
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scilpy.io.utils import (load_matrix_in_any_format,
                             save_matrix_in_any_format,
                             add_verbose_arg,
                             add_overwrite_arg,
                             assert_output_dirs_exist_and_empty)


EPILOG = """
[1] Chamberland M, Raven EP, Genc S, Duffy K, Descoteaux M, Parker GD, Tax CMW,
 Jones DK. Dimensionality reduction of diffusion MRI measures for improved
 tractometry of the human brain. Neuroimage. 2019 Oct 15;200:89-100.
 doi: 10.1016/j.neuroimage.2019.06.020. Epub 2019 Jun 20. PMID: 31228638;
 PMCID: PMC6711466.
[2] Gagnon A., Grenier G., Bocti C., Gillet V., Lepage J.-F., Baccarelli A. A.,
 Posner J., Descoteaux M., Takser L. (2022). White matter microstructural
 variability linked to differential attentional skills and impulsive behavior
 in a pediatric population. Cerebral Cortex.
 https://doi.org/10.1093/cercor/bhac180
[3] https://towardsdatascience.com/what-are-pca-loadings-and-biplots-9a7897f2e559
    """


# Build argument parser.
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
        epilog=EPILOG)

    p.add_argument('in_folder',
                   help='Path to the input folder. See explanation above for '
                        'its expected organization.')
    p.add_argument('out_folder',
                   help='Path to the output folder to export graphs, tables '
                        'and principal \ncomponents matrices.')
    p.add_argument('--metrics', nargs='+', required=True,
                   help='Suffixes of all metrics to include in PCA analysis '
                        '(ex: ad md fa rd). \nThey must be immediately '
                        'followed by the .npy extension.')
    p.add_argument('--list_ids', required=True, metavar='FILE',
                   help='Path to a .txt file containing a list of all ids.')
    p.add_argument('--not_only_common', action='store_true',
                   help='If true, will include all edges from all subjects '
                        'and not only \ncommon edges (Not recommended)')
    p.add_argument('--input_connectoflow', action='store_true',
                   help='If true, script will assume the input folder is a '
                        'Connectoflow output.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def generate_pca_input(dictionary):
    """
    Function to create PCA input from matrix.
    :param dictionary:       Dictionary with metrics as keys containing list of matrices sorted by ids.
    :return:                 Numpy array.
    """
    for metric in dictionary.keys():
        mat = np.rollaxis(np.array(dictionary[metric]), axis=1, start=3)
        matrix_shape = mat.shape[1:3]
        n_group = mat.shape[0]
        mat = mat.reshape((np.prod(matrix_shape) * n_group, 1))
        dictionary[metric] = mat

    return np.hstack([dictionary[i] for i in dictionary.keys()])


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
        if height > 0:
            axs.text(rect.get_x() + rect.get_width() / 2., (height * 1.05),
                     '%.3f' % float(height), ha='center', va='bottom')
        else:
            axs.text(rect.get_x() + rect.get_width()/2., (height*1.05) - 0.15,
                     '%.3f' % float(height), ha='center', va='bottom')


def extracting_common_cnx(dictionary, ind):
    """
    Function to create a binary mask representing common connections across the population
    containing non-zero values.
    :param dictionary:       Dictionary with metrics as keys containing list of matrices sorted by ids.
    :param ind:              Indice of the key to use to generate the binary mask from the dictionary.
    :return:                 Binary mask.
    """
    keys = list(dictionary.keys())
    met = np.copy(dictionary[keys[ind]])

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


def apply_binary_mask(dictionary, mask):
    """
    Function to apply a binary mask to all matrices contained in a dictionary.
    :param dictionary:       Dictionary with metrics as keys containing list of matrices sorted by ids.
    :param mask:             Binary mask.
    :return:                 Dictionary with the same shape as input.
    """
    for a in dictionary.keys():
        for i in range(0, len(dictionary[a])):
            dictionary[a][i] = np.multiply(dictionary[a][i], mask)

    return dictionary


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_output_dirs_exist_and_empty(parser, args, args.out_folder, create_dir=True)

    subjects = open(args.list_ids).read().split()

    if args.input_connectoflow:
        # Loading all matrix.
        logging.info('Loading all matrices from a Connectoflow output...')
        dictionary = {m: [load_matrix_in_any_format(f'{args.in_folder}/{a}/Compute_Connectivity/{m}.npy')
                          for a in subjects]
                      for m in args.metrics}
    else:
        logging.info('Loading all matrices...')
        dictionary = {m: [load_matrix_in_any_format(f'{args.in_folder}/{a}_{m}.npy') for a in subjects]
                      for m in args.metrics}
        # Assert that all metrics have the same number of subjects.
        nb_sub = [len(dictionary[m]) for m in args.metrics]
        assert all(x == len(subjects) for x in nb_sub), "Error, the number of matrices for each metric doesn't match" \
                                                        "the number of subject in the id list." \
                                                        "Please validate input folder."

    # Setting individual matrix shape.
    mat_shape = dictionary[args.metrics[0]][0].shape

    if args.not_only_common:
        # Creating input structure using all edges from all subjects.
        logging.info('Creating PCA input structure with all edges...')
        df = generate_pca_input(dictionary)
    else:
        m1 = extracting_common_cnx(dictionary, 0)
        m2 = extracting_common_cnx(dictionary, 1)

        if m1.shape != mat_shape:
            parser.error("Extracted binary mask doesn't match the shape of individual input matrix.")

        if np.sum(m1) != np.sum(m2):
            parser.error("Different binary mask have been generated from 2 different metrics, \n "
                         "please validate input data.")
        else:
            logging.info('Data shows {} common connections across the population.'.format(np.sum(m1)))

        dictionary = apply_binary_mask(dictionary, m1)

        # Creating input structure.
        logging.info('Creating PCA input structure with common edges...')
        df = generate_pca_input(dictionary)

    # Setting 0 values to nan.
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
    ax.margins(0.1)
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
    ax.margins(0.1)
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
    axs[0].margins(0.2)
    axs[1].margins(0.2)
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
            save_matrix_in_any_format(f'{args.out_folder}/{subjects[s]}_PC{i+1}.npy',
                                      out[i, s, :, :])


if __name__ == "__main__":
    main()
