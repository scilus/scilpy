#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('--in_score', nargs='+',
                   help='Path of the scoring json file.')
    p.add_argument('--out_score',
                   help='Path of the output graph.')

    return p


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(),  bbox_to_anchor=(1.15, 1), loc='upper right', fontsize=9)


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va, fontsize=5)                      # Vertically align label differently for
                                        # positive and negative values.



def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    # Opening JSON files
    score_dict = []
    labels = []
    file_count = 0

    for _, file in enumerate(args.in_score):
        with open(file) as json_files:
            score_file = json.load(json_files)
            score_dict.append(score_file)
            filename = os.path.splitext(os.path.basename(file))[0]
            labels.append(filename)
            file_count += 1

    titles_dict = ['tractogram_overlap', 'tc_bundle', 'fc_bundle', 'tc_dice']
    x_tick = ['fake', 'trac_overlap', 'dice_0', 'dice_1', 'dice_2', 'dice_3',
              'dice_4', 'dice_5', 'dice_6']
    bundle_dict = np.arange(7)
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    data = {}
    for k in labels:
        data[k] = []

    count = 0
    for tract_score in score_dict:
        for title in titles_dict[:1]:
            data[str(labels[count])].append(tract_score[title])
        for bundle in bundle_dict:
            data[str(labels[count])].append(tract_score["bundle_wise"]["true_connections"]
                    ["('rois/FiberCupGroundTruth_filtered_bundle_" + str(bundle) +
                    "_tail.nii.gz', 'rois/FiberCupGroundTruth_filtered_bundle_" +
                    str(bundle) + "_head.nii.gz')"][titles_dict[3]])
        count += 1

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9,5))

    ax.set_xticklabels(x_tick, fontsize=9)
    ax.set_ylim(0.5, 1)
    plt.title('Tractogram scores', fontsize=16)
    plt.ylabel('Score')
    bar_plot(ax, data, colors=colors_list, total_width=.8, single_width=1.1)
    #add_value_labels(ax)

    fig.tight_layout()
    plt.savefig(args.out_score)


if __name__ == "__main__":
    main()
