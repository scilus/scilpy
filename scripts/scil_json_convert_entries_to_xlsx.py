#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert a final aggregated json file to an Excel spreadsheet.
Typically used during the tractometry pipeline.

Formerly: scil_convert_json_to_xlsx.py
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)


def _get_all_bundle_names(stats):
    bnames = set()

    for bundles in iter(stats.values()):
        bnames |= set(bundles.keys())

    return list(bnames)


def _are_all_elements_scalars(bundle_stat):
    for v in iter(bundle_stat.values()):
        if type(v) is not int and type(v) is not float:
            return False

    return True


def _get_metrics_names(stats):
    mnames = set()

    for bundles in iter(stats.values()):
        for val in iter(bundles.values()):
            mnames |= set(val.keys())
    return mnames


def _get_labels(stats):
    labels = set()

    for bundles in iter(stats.values()):
        for lab in iter(bundles.values()):
            if type(lab[list(lab.keys())[0]]) is dict:
                for vals in iter(lab.values()):
                    labels |= set(vals.keys())
            else:
                labels |= set(lab.keys())

    return list(labels)


def _find_stat_name(stats):
    first_sub_stats = stats[list(stats.keys())[0]]
    first_bundle_stats = first_sub_stats[list(first_sub_stats.keys())[0]]

    return list(first_bundle_stats.keys())[0]


def _get_stats_parse_function(stats, stats_over_population):
    first_sub_stats = stats[list(stats.keys())[0]]
    first_bundle_stats = first_sub_stats[list(first_sub_stats.keys())[0]]
    first_bundle_substat = first_bundle_stats[list(
        first_bundle_stats.keys())[0]]

    if len(first_bundle_stats.keys()) == 1 and\
            _are_all_elements_scalars(first_bundle_stats):
        return _parse_scalar_stats
    elif len(first_bundle_stats.keys()) == 4 and \
            set(first_bundle_stats.keys()) == \
            set(['lesion_total_vol', 'lesion_avg_vol', 'lesion_std_vol',
                 'lesion_count']):
        return _parse_lesion
    elif len(first_bundle_stats.keys()) == 4 and \
            set(first_bundle_stats.keys()) == \
            set(['min_length', 'max_length', 'mean_length', 'std_length']):
        return _parse_lengths
    elif type(first_bundle_substat) is dict:
        sub_keys = list(first_bundle_substat.keys())
        if set(sub_keys) == set(['mean', 'std']):
            if stats_over_population:
                return _parse_per_label_population_stats
            else:
                return _parse_scalar_meanstd
        elif type(first_bundle_substat[sub_keys[0]]) is dict:
            return _parse_per_point_meanstd
        elif _are_all_elements_scalars(first_bundle_substat):
            return _parse_per_label_scalar

    raise IOError('Unable to recognize stats type!')


def _write_dataframes(dataframes, df_names, output_path):
    with pd.ExcelWriter(output_path) as writer:
        for df, df_name in zip(dataframes, df_names):
            df.to_excel(writer, sheet_name=df_name)


def _parse_scalar_stats(stats, subs, bundles):
    stat_name = _find_stat_name(stats)

    nb_subs = len(subs)
    nb_bundles = len(bundles)

    stats_array = np.full((nb_subs, nb_bundles), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            b_stat = stats[sub_name].get(bundle_name)

            if b_stat is not None:
                stats_array[sub_id, bundle_id] = b_stat[stat_name]

    dataframes = [pd.DataFrame(data=stats_array,
                               index=subs,
                               columns=bundles)]
    df_names = [stat_name]

    return dataframes, df_names


def _parse_scalar_meanstd(stats, subs, bundles):
    metric_names = _get_metrics_names(stats)

    nb_subs = len(subs)
    nb_bundles = len(bundles)
    nb_metrics = len(metric_names)

    means = np.full((nb_subs, nb_bundles, nb_metrics), np.NaN)
    stddev = np.full((nb_subs, nb_bundles, nb_metrics), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            for metric_id, metric_name in enumerate(metric_names):
                b_stat = stats[sub_name].get(bundle_name)

                if b_stat is not None:
                    m_stat = b_stat.get(metric_name)

                    if m_stat is not None:
                        means[sub_id, bundle_id, metric_id] = m_stat['mean']
                        stddev[sub_id, bundle_id, metric_id] = m_stat['std']

    dataframes = []
    df_names = []

    for metric_id, metric_name in enumerate(metric_names):
        dataframes.append(pd.DataFrame(data=means[:, :, metric_id],
                                       index=subs, columns=bundles))
        df_names.append(metric_name + "_mean")

        dataframes.append(pd.DataFrame(data=stddev[:, :, metric_id],
                                       index=subs, columns=bundles))
        df_names.append(metric_name + "_std")

    return dataframes, df_names


def _parse_scalar_lesions(stats, subs, bundles):
    metric_names = _get_metrics_names(stats)
    nb_subs = len(subs)
    nb_bundles = len(bundles)
    nb_metrics = len(metric_names)

    means = np.full((nb_subs, nb_bundles, nb_metrics), np.NaN)
    stddev = np.full((nb_subs, nb_bundles, nb_metrics), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            for metric_id, metric_name in enumerate(metric_names):
                b_stat = stats[sub_name].get(bundle_name)

                if b_stat is not None:
                    m_stat = b_stat.get(metric_name)

                    if m_stat is not None:
                        means[sub_id, bundle_id, metric_id] = m_stat['mean']
                        stddev[sub_id, bundle_id, metric_id] = m_stat['std']

    dataframes = []
    df_names = []

    for metric_id, metric_name in enumerate(metric_names):
        dataframes.append(pd.DataFrame(data=means[:, :, metric_id],
                                       index=subs, columns=bundles))
        df_names.append(metric_name + "_mean")

        dataframes.append(pd.DataFrame(data=stddev[:, :, metric_id],
                                       index=subs, columns=bundles))
        df_names.append(metric_name + "_std")

    return dataframes, df_names


def _parse_lengths(stats, subs, bundles):
    nb_subs = len(subs)
    nb_bundles = len(bundles)

    min_lengths = np.full((nb_subs, nb_bundles), np.NaN)
    max_lengths = np.full((nb_subs, nb_bundles), np.NaN)
    mean_lengths = np.full((nb_subs, nb_bundles), np.NaN)
    std_lengths = np.full((nb_subs, nb_bundles), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            b_stat = stats[sub_name].get(bundle_name)

            if b_stat is not None:
                min_lengths[sub_id, bundle_id] = b_stat['min_length']
                max_lengths[sub_id, bundle_id] = b_stat['max_length']
                mean_lengths[sub_id, bundle_id] = b_stat['mean_length']
                std_lengths[sub_id, bundle_id] = b_stat['std_length']

    dataframes = [pd.DataFrame(data=min_lengths,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=max_lengths,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=mean_lengths,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=std_lengths,
                               index=subs,
                               columns=bundles)]

    df_names = ["min_length", "max_length", "mean_length", "std_length"]

    return dataframes, df_names


def _parse_lesion(stats, subs, bundles):
    nb_subs = len(subs)
    nb_bundles = len(bundles)

    total_volume = np.full((nb_subs, nb_bundles), np.NaN)
    avg_volume = np.full((nb_subs, nb_bundles), np.NaN)
    std_volume = np.full((nb_subs, nb_bundles), np.NaN)
    lesion_count = np.full((nb_subs, nb_bundles), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            b_stat = stats[sub_name].get(bundle_name)

            if b_stat is not None:
                total_volume[sub_id, bundle_id] = b_stat['lesion_total_vol']
                avg_volume[sub_id, bundle_id] = b_stat['lesion_avg_vol']
                std_volume[sub_id, bundle_id] = b_stat['lesion_std_vol']
                lesion_count[sub_id, bundle_id] = b_stat['lesion_count']

    dataframes = [pd.DataFrame(data=total_volume,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=avg_volume,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=std_volume,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=lesion_count,
                               index=subs,
                               columns=bundles)]

    df_names = ["lesion_total_vol", "lesion_avg_vol",
                "lesion_std_vol", "lesion_count"]

    return dataframes, df_names


def _parse_per_label_scalar(stats, subs, bundles):
    labels = _get_labels(stats)
    labels.sort()

    nb_subs = len(subs)
    nb_bundles = len(bundles)
    nb_labels = len(labels)

    stats_array = np.full((nb_subs, nb_bundles * nb_labels), np.NaN)
    column_names = []
    for bundle_name in bundles:
        column_names.extend(["{}_{}".format(bundle_name, label)
                             for label in labels])

    stat_name = _find_stat_name(stats)
    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):

            b_stat = stats[sub_name].get(bundle_name)

            if b_stat is not None:
                m_stat = b_stat.get(stat_name)

                if m_stat is not None:
                    for label_id, label in enumerate(labels):
                        label_stat = m_stat.get(label)

                        if label_stat is not None:
                            stats_array[sub_id,
                                        bundle_id * len(labels) + label_id] =\
                                label_stat

    dataframes = [pd.DataFrame(data=stats_array,
                               index=subs,
                               columns=column_names)]
    df_names = ['{}_per_label'.format(stat_name)]

    return dataframes, df_names


def _parse_per_point_meanstd(stats, subs, bundles):
    labels = _get_labels(stats)
    labels.sort()

    metric_names = _get_metrics_names(stats)

    nb_subs = len(subs)
    nb_bundles = len(bundles)
    nb_labels = len(labels)
    nb_metrics = len(metric_names)

    means = np.full((nb_subs, nb_bundles * nb_labels, nb_metrics), np.NaN)
    stddev = np.full((nb_subs, nb_bundles * nb_labels, nb_metrics), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            b_stat = stats[sub_name].get(bundle_name)

            if b_stat is not None:
                for metric_id, metric_name in enumerate(metric_names):
                    m_stat = b_stat.get(metric_name)

                    if m_stat is not None:
                        for label_id, label in enumerate(labels):
                            label_stat = m_stat.get(label)

                            if label_stat is not None:
                                means[sub_id,
                                      bundle_id * len(labels) + label_id,
                                      metric_id] =\
                                    label_stat['mean']
                                stddev[sub_id,
                                       bundle_id * len(labels) + label_id,
                                       metric_id] =\
                                    label_stat['std']

    column_names = []
    for bundle_name in bundles:
        column_names.extend(["{}_{}".format(bundle_name, label)
                             for label in labels])

    dataframes = []
    df_names = []
    for metric_id, metric_name in enumerate(metric_names):
        dataframes.append(pd.DataFrame(data=means[:, :, metric_id],
                                       index=subs, columns=column_names))
        df_names.append(metric_name + "_mean")

        dataframes.append(pd.DataFrame(data=stddev[:, :, metric_id],
                                       index=subs, columns=column_names))
        df_names.append(metric_name + "_std")

    return dataframes, df_names


def _parse_per_label_population_stats(stats, bundles, metrics):
    labels = list(stats[bundles[0]][metrics[0]].keys())
    labels.sort()

    nb_bundles = len(bundles)
    nb_labels = len(labels)
    nb_metrics = len(metrics)

    means = np.full((nb_bundles, nb_labels, nb_metrics), np.NaN)
    stddev = np.full((nb_bundles, nb_labels, nb_metrics), np.NaN)

    for bundle_id, bundle_name in enumerate(bundles):
        b_stat = stats.get(bundle_name)

        if b_stat is not None:
            for metric_id, metric_name in enumerate(metrics):
                m_stat = b_stat.get(metric_name)

                if m_stat is not None:
                    for label_id, label in enumerate(labels):
                        label_stat = m_stat.get(label)

                        if label_stat is not None:
                            means[bundle_id, label_id, metric_id] =\
                                np.average(label_stat['mean'])
                            stddev[bundle_id, label_id, metric_id] =\
                                np.average(label_stat['std'])

    dataframes = []
    df_names = []
    for metric_id, metric_name in enumerate(metrics):
        dataframes.append(pd.DataFrame(data=np.array(means[:, :, metric_id]),
                                       index=bundles,
                                       columns=labels))
        df_names.append(metric_name + "_mean")

        dataframes.append(pd.DataFrame(data=np.array(stddev[:, :, metric_id]),
                                       index=bundles,
                                       columns=labels))
        df_names.append(metric_name + "_std")

    return dataframes, df_names


def _create_xlsx_from_json(json_path, xlsx_path,
                           sort_subs=True, sort_bundles=True,
                           ignored_bundles_fpath=None,
                           stats_over_population=False):
    with open(json_path, 'r') as json_file:
        stats = json.load(json_file)

    subs = list(stats.keys())
    if sort_subs:
        subs.sort()

    bundle_names = _get_all_bundle_names(stats)
    if sort_bundles:
        bundle_names.sort()

    if ignored_bundles_fpath is not None:
        with open(ignored_bundles_fpath, 'r') as f:
            bundles_to_ignore = [l.strip() for l in f]
        bundle_names = filter(lambda name: name not in bundles_to_ignore,
                              bundle_names)

    cur_stats_func = _get_stats_parse_function(stats, stats_over_population)

    dataframes, df_names = cur_stats_func(stats, subs, bundle_names)

    if len(dataframes):
        _write_dataframes(dataframes, df_names, xlsx_path)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_json',
                   help='File containing the json stats (.json).')

    p.add_argument('out_xlsx',
                   help='Output Excel file for the stats (.xlsx).')

    p.add_argument('--no_sort_subs', action='store_false',
                   help='If set, subjects won\'t be sorted alphabetically.')

    p.add_argument('--no_sort_bundles', action='store_false',
                   help='If set, bundles won\'t be sorted alphabetically.')
    p.add_argument('--ignore_bundles', metavar='FILE',
                   help='Path to a text file containing a list of bundles '
                        'to ignore (.txt).\nOne bundle, corresponding to keys '
                        'in the json, per line.')
    p.add_argument('--stats_over_population', action='store_true',
                   help='If set, consider the input stats to be over an '
                        'entire population and not subject-based.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_json)
    assert_outputs_exist(parser, args, args.out_xlsx)

    _create_xlsx_from_json(args.in_json, args.out_xlsx,
                           sort_subs=args.no_sort_subs,
                           sort_bundles=args.no_sort_bundles,
                           ignored_bundles_fpath=args.ignore_bundles,
                           stats_over_population=args.stats_over_population)


if __name__ == "__main__":
    main()
