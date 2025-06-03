#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute PCA analysis on a set of connectivity matrices. The output is
all significant principal components in a connectivity matrix format.
This script can take into account all edges from every subject in a population
or only non-zero edges across all subjects.

Interpretation of resulting principal components can be done by evaluating the
loadings values for each metrics. A value near 0 means that this metric doesn't
contribute to this specific component whereas high positive or negative values
mean a larger contribution. Components can then be labeled based on which
metric contributes the highest. For example, a principal component showing a
high loading for afd_fixel and near 0 loading for all other metrics can be
interpreted as axonal density (see Gagnon et al. 2022 for this specific example
or ref [3] for an introduction to PCA).

The script can take directly as input a connectoflow output folder. Simply use
the --input_connectoflow flag. Else, the script expects a single folder
containing all matrices for all subjects. Those matrices can be obtained, for
instance, by scil_connectivity_compute_matrices.py.
Example: Default input
        [in_folder]
        |--- sub-01_ad.npy
        |--- sub-01_md.npy
        |--- sub-02_ad.npy
        |--- sub-02_md.npy
        |--- ...
Connectoflow input:
        [in_folder]
              [subj-01]
                   [Compute_Connectivity]
                       |--- ad.npy

The plots, tables and principal components matrices will be saved in the
designated folder from the <out_folder> argument. If you want to move back your
principal components matrices in your connectoflow output, you can use a
similar bash command for all principal components:
for sub in `cat list_id.txt`;
do
    cp out_folder/${sub}_PC1.npy connectoflow_output/$sub/Compute_Connectivity/
done

EXAMPLE USAGE:
scil_connectivity_compute_pca.py input_folder/ output_folder/
    --metrics ad fa md rd [...] --list_ids list_ids.txt

-------------------------------------------------------------------------------
References:
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
-------------------------------------------------------------------------------
"""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scilpy.io.utils import (load_matrix_in_any_format,
                             save_matrix_in_any_format,
                             add_verbose_arg,
                             add_overwrite_arg,
                             assert_output_dirs_exist_and_empty,
                             assert_inputs_dirs_exist, assert_inputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

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
    p.add_argument('--all_edges', action='store_true',
                   help='If true, will include all edges from all subjects '
                        'and not only \ncommon edges (Not recommended)')
    p.add_argument('--input_connectoflow', action='store_true',
                   help='If true, script will assume the input folder is a '
                        'Connectoflow output.')
    p.add_argument('--show', action='store_true',
                   help="If set, show matplotlib figures. Else, they are "
                        "only saved in the output folder.")

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def plot_eigenvalues(pca, principaldf, nb_metrics, out_file):
    # Plot the eigenvalues.
    logging.info('Plotting results...')
    eigenvalues = pca.explained_variance_
    pos = list(range(1, nb_metrics + 1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    bar_eig = ax.bar(pos, eigenvalues, align='center',
                     tick_label=principaldf.columns)
    ax.set_xlabel('Principal Components', fontsize=10)
    ax.set_ylabel('Eigenvalues', fontsize=10)
    ax.set_title('Eigenvalues for each principal components.', fontsize=10)
    ax.margins(0.1)
    autolabel(bar_eig, ax)
    plt.savefig(out_file)
    return eigenvalues


def plot_explained_variance(pca, principaldf, nb_metrics, out_file):
    # Plot the explained variance.
    explained_var = pca.explained_variance_ratio_
    pos = list(range(1, nb_metrics + 1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    bar_var = ax.bar(pos, explained_var, align='center',
                     tick_label=principaldf.columns)
    ax.set_xlabel('Principal Components', fontsize=10)
    ax.set_ylabel('Explained variance', fontsize=10)
    ax.set_title('Amount of explained variance for all principal components.',
                 fontsize=10)
    ax.margins(0.1)
    autolabel(bar_var, ax)
    plt.savefig(out_file)


def plot_contribution(pca, principaldf, metrics, out_excel, out_image):
    # Plot the contribution of each measures to principal component.
    component = pca.components_
    output_component = pd.DataFrame(component, index=principaldf.columns,
                                    columns=metrics)
    output_component.to_excel(out_excel, index=True, header=True)

    fig, axs = plt.subplots(2)
    fig.suptitle('Graph of the contribution of each measures to the first '
                 'and second principal component.', fontsize=10)
    pos = list(range(1, len(metrics) + 1))
    bar_pc1 = axs[0].bar(pos, component[0], align='center',
                         tick_label=metrics)
    bar_pc2 = axs[1].bar(pos, component[1], align='center',
                         tick_label=metrics)
    axs[0].margins(0.2)
    axs[1].margins(0.2)
    autolabel(bar_pc1, axs[0])
    autolabel(bar_pc2, axs[1])
    for ax in axs.flat:
        ax.set(xlabel='Diffusion measures', ylabel='Loadings')
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig(out_image)


def autolabel(rects, axs):
    """
    Attach a text label above each bar displaying its height (or bar value).
    """
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            axs.text(rect.get_x() + rect.get_width() / 2., (height * 1.05),
                     '%.3f' % float(height), ha='center', va='bottom')
        else:
            axs.text(rect.get_x() + rect.get_width()/2., (height*1.05) - 0.15,
                     '%.3f' % float(height), ha='center', va='bottom')


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_dirs_exist(parser, args.in_folder)
    assert_inputs_exist(parser, args.list_ids)
    assert_output_dirs_exist_and_empty(parser, args, args.out_folder,
                                       create_dir=True)
    out_eigenvalues = os.path.join(args.out_folder, 'eigenvalues.pdf')
    out_variance = os.path.join(args.out_folder, 'explained_variance.pdf')
    out_excel = os.path.join(args.out_folder, 'loadings.xlsx')
    out_contributions = os.path.join(args.out_folder, 'contribution.pdf')

    with open(args.list_ids) as f:
        subjects = f.read().split()

    # Loading all matrices.
    if args.input_connectoflow:
        logging.info('Loading all matrices from a Connectoflow output...')
        files_per_metric = [[os.path.join(
            args.in_folder, a, 'Compute_Connectivity', '{}.npy'.format(m))
            for a in subjects]
            for m in args.metrics]
    else:
        logging.info('Loading all matrices...')
        files_per_metric = [[os.path.join(
            args.in_folder, '{}_{}.npy'.format(a, m))
            for a in subjects]
            for m in args.metrics]

    assert_inputs_exist(parser, sum(files_per_metric, []))
    matrices_per_metric = [[load_matrix_in_any_format(f) for f in files]
                           for files in files_per_metric]

    # Setting individual matrix shape.
    mat_shape = matrices_per_metric[0][0].shape

    # Creating input structure
    if args.all_edges:
        logging.info('Creating PCA input structure with all edges...')
    else:
        logging.info('Creating PCA input structure with common edges...')
        nb_subjects = len(matrices_per_metric[0])

        # Using the first metric
        subj_masks = [m != 0 for m in matrices_per_metric[0]]  # Mask per subj
        common_edges_mask = np.sum(subj_masks, axis=0) == nb_subjects

        # Verifying that other metrics have the same common edges
        for i, matrices in enumerate(matrices_per_metric[1:]):
            tmp_subj_masks = [m != 0 for m in matrices]  # Binary per subj
            tmp_common = np.sum(tmp_subj_masks, axis=0) == nb_subjects
            if not np.array_equal(common_edges_mask, tmp_common):
                parser.error("Different binary masks (common edge) have been "
                             "generated from metric {} than other metrics. "
                             "Please validate your input data."
                             .format(args.metrics[i]))

        plt.figure()
        plt.imshow(common_edges_mask)
        plt.title("Common edges mask")

        logging.info('Data shows {} common connections across the population.'
                     .format(np.sum(common_edges_mask)))

        # Apply mask
        matrices_per_metric = [[m * common_edges_mask
                               for m in metric_matrices]
                               for metric_matrices in matrices_per_metric]

    # Creating input structure.
    # For each metric, combining all subjects together, flattening all
    # connectivity matrices.
    flat_mats = []
    for metric_matrices in matrices_per_metric:
        mat = np.rollaxis(np.array(metric_matrices), axis=1, start=3)
        mat = mat.reshape((np.prod(mat.shape), 1))
        flat_mats.append(mat)
    df = np.hstack(flat_mats)   # Shape: nb_metrics x (flattened data)
    df[df == 0] = np.nan  # Setting 0 values to nan.

    # Standardizing the data.
    logging.info('Standardizing data...')
    x = StandardScaler().fit_transform(df)
    df_pca = x[~np.isnan(x).any(axis=1)]
    x = np.nan_to_num(x, nan=0.)

    # Performing the PCA.
    logging.info('Performing PCA...')
    pca = PCA(n_components=len(args.metrics))
    principalcomponents = pca.fit_transform(df_pca)
    principaldf = pd.DataFrame(
        data=principalcomponents,
        columns=[f'PC{i}' for i in range(1, len(args.metrics) + 1)])

    # Plotting
    eigenvalues = plot_eigenvalues(pca, principaldf,
                                   nb_metrics=len(args.metrics),
                                   out_file=out_eigenvalues)
    plot_explained_variance(pca, principaldf, nb_metrics=len(args.metrics),
                            out_file=out_variance)
    plot_contribution(pca, principaldf, args.metrics, out_excel,
                      out_contributions)

    # Extract the derived newly computed measures from the PCA analysis.
    logging.info('Saving matrices for PC with eigenvalues > 1...')
    out = pca.transform(x)
    out = out * (x != 0)
    out = out.swapaxes(0, 1).reshape(len(args.metrics), len(subjects),
                                     mat_shape[0], mat_shape[1])

    # Save matrix for significant components
    nb_pc = eigenvalues[eigenvalues >= 1]
    for i in range(0, len(nb_pc)):
        for s in range(0, len(subjects)):
            filename = os.path.join(args.out_folder,
                                    f'{subjects[s]}_PC{i+1}.npy')
            save_matrix_in_any_format(filename, out[i, s, :, :])

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
