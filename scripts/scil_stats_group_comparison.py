#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run group comparison statistics on metrics from tractometry
1) Separate the sample given a particular variable (group_by) into groups

2) Does Shapiro-Wilk test of normality for every sample
https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test

3) Does Levene or Bartlett (depending on normality) test of variance
homogeneity Levene:
https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm
Bartlett:
https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.htm

4) Test the group difference for every measure with the correct test depending
   on the sample (Student, Welch, Mannwhitneyu, ANOVA, Kruskall-Wallis)
Student :
https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test
Welch :
https://en.wikipedia.org/wiki/Welch%27s_t-test
Mann-Whitney U :
https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
ANOVA :
http://www.biostathandbook.com/onewayanova.html
Kruskall-Wallis :
https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance


5) If the group difference test is positive and number of group is greater than
   2, test the group difference two by two.

6) Generate the result for all metrics and bundles

Formerly: scil_group_comparison.py
"""

import argparse
import json
import logging
import os

from itertools import product

from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.stats.stats import (verify_group_difference,
                                verify_homoscedasticity,
                                verify_normality,
                                verify_post_hoc)
from scilpy.stats.utils import (data_for_stat,
                                get_group_data_sample,
                                visualise_distribution,
                                write_current_dictionnary)


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_json', metavar='IN_JSON',
                   help='Input JSON file from tractometry nextflow pipeline'
                        ' or equivalent.')

    p.add_argument('in_participants', metavar='IN_PARTICIPANTS',
                   help='Input tsv participants file.'
                        'See doc in https://scilpy.readthedocs.io/en/latest/documentation/construct_participants_tsv_file.html.')

    p.add_argument('group_by', metavar='GROUP_BY',
                   help='Variable that will be used to compare group '
                        'together.')

    p.add_argument('--out_dir', metavar='OUT_DIR',
                   default='stats',
                   help='Name of the output folder path. [%(default)s]')

    p.add_argument('--out_json',
                   help='The name of the result json output file otherwise '
                        'it will be printed.')

    p.add_argument('--bundles', '-b',
                   default='all', nargs='+',
                   help='Bundle(s) in which you want to do stats. '
                        '[%(default)s]')

    p.add_argument('--metrics', '-m',
                   default='all', nargs='+',
                   help='Metric(s) on which you want to do stats. '
                        '[%(default)s]')

    p.add_argument('--values', '--va',
                   default='all', nargs='+',
                   help='Value(s) on which you want to do stats (mean, std).'
                        ' [%(default)s]')

    p.add_argument('--alpha_error', '-a',
                   default=0.05,
                   help='Type 1 error for all the test. [%(default)s]')

    p.add_argument('--generate_graph',
                   '--gg', action='store_true',
                   help='Generate a simple plot of every metric across '
                        'groups.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    required_args = [args.in_json, args.in_participants]
    assert_inputs_exist(parser, required_args)

    req_folder = os.path.join(args.out_dir, 'Graph')
    assert_output_dirs_exist_and_empty(parser, args, req_folder)

    # We generated the stats object
    my_data = data_for_stat(args.in_json,
                            args.in_participants)

    bundles = args.bundles
    metrics = args.metrics
    values = args.values

    if args.bundles == 'all':
        bundles = my_data.get_bundles_list()
    if args.metrics == 'all':
        metrics = my_data.get_metrics_list()
    if args.values == 'all':
        values = my_data.get_values_list()

    alpha_error = float(args.alpha_error)

    my_group_dict = my_data.get_groups_dictionnary(args.group_by)
    # Initialise the result dictionnary
    result_dict = {}

    # We do the comparison for every single combianaison of metric-bundle-value
    for b, m, v in product(bundles, metrics, values):
        # First we extract the basic information on that comparison

        curr_comparison_measure = ('_').join([b, m, v])

        logging.info('______________________')
        logging.info('Measure to compare: {}'.format(curr_comparison_measure))

        # Check normality of that metric across all groups

        current_normality = {}
        overall_normality = True
        groups_array = []
        for group in my_group_dict:
            curr_sample = get_group_data_sample(my_group_dict, group, b, m, v)
            logging.info('Group {}'.format(group))
            current_normality[group] = verify_normality(
                                            curr_sample,
                                            alpha_error)
            if not current_normality[group][0]:
                overall_normality = False
            groups_array.append(curr_sample)
        logging.info('Normality result:')
        logging.info(current_normality)
        logging.info('Overall Normality:')
        logging.info(overall_normality)
        logging.info('Groups array:')
        logging.info(groups_array)

        # Generate graph of the metric
        if args.generate_graph:
            visualise_distribution(groups_array,
                                   my_data.get_participants_list(),
                                   b, m, v,
                                   args.out_dir,
                                   my_data.get_groups_list(args.group_by))

        # Quit if we didnt separate by group
        if len(my_data.get_groups_list(args.group_by)) == 1:
            logging.error('There is only 1 group generated. '
                          'We cannot continue the groups comparison')
            raise BaseException('Only 1 group generated from '
                                '{}'.format(args.group_by))

        # Check homoscedasticity
        variance_equality = verify_homoscedasticity(
                                            groups_array,
                                            normality=overall_normality,
                                            alpha=alpha_error)
        logging.info('Equality of variance result:')
        logging.info(variance_equality)

        # Now we compare the groups population

        difference = verify_group_difference(
                                    groups_array,
                                    normality=overall_normality,
                                    homoscedasticity=variance_equality[1],
                                    alpha=alpha_error)
        logging.info('Main test result:')
        logging.info(difference)

        # Finally if we have more than 2 groups and found a difference
        # We do a post hoc analysis to explore where is this difference
        if difference[1] and difference[0] == 'ANOVA':
            diff_2_by_2 = verify_post_hoc(
                                    groups_array,
                                    my_data.get_groups_list(args.group_by),
                                    test='Student',
                                    alpha=alpha_error)
        elif difference[1] and difference[0] == 'Kruskalwallis':
            diff_2_by_2 = verify_post_hoc(
                                    groups_array,
                                    my_data.get_groups_list(args.group_by),
                                    test='Mannwhitneyu',
                                    alpha=alpha_error)
        elif difference[1] and difference[0] == 'Friedmann':
            diff_2_by_2 = verify_post_hoc(
                                    groups_array,
                                    my_data.get_groups_list(args.group_by),
                                    test='Wilcoxon',
                                    alpha=alpha_error)
        else:
            diff_2_by_2 = []

        logging.info('Summary of difference 2 by 2:')
        logging.info(diff_2_by_2)

        # Write the current metric in the report
        curr_dict = write_current_dictionnary(curr_comparison_measure,
                                              current_normality,
                                              variance_equality,
                                              difference,
                                              diff_2_by_2)
        result_dict[curr_comparison_measure] = curr_dict

    # Saving the result dictionnary into a json file and csv if necessary
    if args.out_json:
        with open(os.path.join(args.out_dir, args.out_json), 'w') as outfile:
            json.dump(result_dict, outfile, indent=args.indent,
                      sort_keys=args.sort_keys)
    else:
        print(json.dumps(result_dict,
                         indent=args.indent,
                         sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
