
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_stats(means, stds, title=None, xlabel=None,
                       ylabel=None, figlabel=None, fill_color=None,
                       display_means=False):
    """
    Plots the mean of a metric along n points with the standard deviation.

    Parameters
    ----------
    means: Numpy 1D (or 2D) array of size n
        Mean of the metric along n points.
    stds: Numpy 1D (or 2D) array of size n
        Standard deviation of the metric along n points.
    title: string
        Title of the figure.
    xlabel: string
        Label of the X axis.
    ylabel: string
        Label of the Y axis (suggestion: the metric name).
    figlabel: string
        Label of the figure (only metadata in the figure object returned).
    fill_color: string
        Hexadecimal RGB color filling the region between mean ± std. The
        hexadecimal RGB color should be formatted as #RRGGBB
    display_means: bool
        Display the subjects means as semi-transparent line
    Return
    ------
    The figure object.
    """
    matplotlib.style.use('ggplot')

    fig, ax = plt.subplots()

    # Set optional information to the figure, if required.
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if figlabel is not None:
        fig.set_label(figlabel)

    if means.ndim > 1:
        mean = np.average(means, axis=1)
        std = np.average(stds, axis=1)
        alpha = 0.5
    else:
        mean = np.array(means).ravel()
        std = np.array(stds).ravel()
        alpha = 0.9

    dim = np.arange(1, len(mean)+1, 1)

    if len(mean) <= 20:
        ax.xaxis.set_ticks(dim)

    ax.set_xlim(0, len(mean)+1)

    if means.ndim > 1 and display_means:
        for i in range(means.shape[-1]):
            ax.plot(dim, means[:, i], color="k", linewidth=1,
                    solid_capstyle='round', alpha=0.1)

    # Plot the mean line.
    ax.plot(dim, mean, color="k", linewidth=5, solid_capstyle='round')

    # Plot the std
    plt.fill_between(dim, mean - std, mean + std,
                     facecolor=fill_color, alpha=alpha)

    plt.close(fig)
    return fig


def plot_residuals(data_diff, mask, R_k, q1, q3, iqr, residual_basename):
    """
    Plots residual statistics for DWI.

    Parameters
    ----------
    data_diff: np.ndarray
        The 4D residuals between the DWI and predicted data.
    mask : Numpy 3D array or None
        Mask array indicating the region of interest for computing residuals.
        If None, residuals are computed for the entire dataset.
    R_k : Numpy 1D array
        Mean residual values for each DWI volume.
    q1 : Numpy 1D array
        First quartile values for each DWI volume.
    q3 : Numpy 1D array
        Third quartile values for each DWI volume.
    iqr : Numpy 1D array
        Interquartile range (Q3 - Q1) for each DWI volume.
    residual_basename : string
        Basename for saving the output plot file. The file will be saved as
        '<residual_basename>_residuals_stats.png'.
    Returns
    -------
    None

    The function generates a plot and saves it as a PNG file.

    """
    # Showing results in graph
    # Note that stats will be computed manually and plotted using bxp
    # but could be computed using stats = cbook.boxplot_stats
    # or pyplot.boxplot(x)

    # Initializing stats as a List[dict]
    stats = [dict.fromkeys(['label', 'mean', 'iqr', 'cilo', 'cihi',
                            'whishi', 'whislo', 'fliers', 'q1',
                            'med', 'q3'], [])
             for _ in range(data_diff.shape[-1])]

    nb_voxels = np.count_nonzero(mask)
    percent_outliers = np.zeros(data_diff.shape[-1], dtype=np.float32)
    for k in range(data_diff.shape[-1]):
        stats[k]['med'] = (q1[k] + q3[k]) / 2
        stats[k]['mean'] = R_k[k]
        stats[k]['q1'] = q1[k]
        stats[k]['q3'] = q3[k]
        stats[k]['whislo'] = q1[k] - 1.5 * iqr[k]
        stats[k]['whishi'] = q3[k] + 1.5 * iqr[k]
        stats[k]['label'] = k

        # Outliers are observations that fall below Q1 - 1.5(IQR) or
        # above Q3 + 1.5(IQR) We check if a voxel is an outlier only if
        # we have a mask, else we are biased.
        if mask is not None:
            x = data_diff[..., k]
            outliers = (x < stats[k]['whislo']) | (x > stats[k]['whishi'])
            percent_outliers[k] = np.sum(outliers) / nb_voxels * 100
            # What would be our definition of too many outliers?
            # Maybe mean(all_means)+-3SD?
            # Or we let people choose based on the figure.
            # if percent_outliers[k] > ???? :
            #    logger.warning('   Careful! Diffusion-Weighted Image'
            #                   ' i=%s has %s %% outlier voxels',
            #                   k, percent_outliers[k])

    if mask is None:
        fig, axe = plt.subplots(nrows=1, ncols=1, squeeze=False)
    else:
        fig, axe = plt.subplots(nrows=1, ncols=2, squeeze=False,
                                figsize=[10, 4.8])
        # Default is [6.4, 4.8]. Increasing width to see better.

    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    meanprops = dict(linestyle='-', linewidth=2.5, color='green')
    axe[0, 0].bxp(stats, showmeans=True, meanline=True, showfliers=False,
                  medianprops=medianprops, meanprops=meanprops)
    axe[0, 0].set_xlabel('DW image')
    axe[0, 0].set_ylabel('Residuals per DWI volume. Red is median,\n'
                         'green is mean. Whiskers are 1.5*interquartile')
    axe[0, 0].set_title('Residuals')
    axe[0, 0].set_xticks(range(0, q1.shape[0], 5))
    axe[0, 0].set_xticklabels(range(0, q1.shape[0], 5))

    if mask is not None:
        axe[0, 1].plot(range(data_diff.shape[-1]), percent_outliers)
        axe[0, 1].set_xticks(range(0, q1.shape[0], 5))
        axe[0, 1].set_xticklabels(range(0, q1.shape[0], 5))
        axe[0, 1].set_xlabel('DW image')
        axe[0, 1].set_ylabel('Percentage of outlier voxels')
        axe[0, 1].set_title('Outliers')
    plt.savefig(residual_basename + '_residuals_stats.png')
