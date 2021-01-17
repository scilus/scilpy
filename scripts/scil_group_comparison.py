#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import logging
import os

from itertools import product

from scilpy.io.utils import assert_inputs_exist
from scilpy.stats.stats import (verify_group_difference,
                                verify_homoscedasticity,
                                verify_normality,
                                verify_post_hoc)
from scilpy.stats.utils import (data_for_stat,
                                get_group_data_sample,
                                visualise_distribution,
                                write_current_dictionnary,
                                write_csv_from_json)
DESCRIPTION = """
Run group comparison statistics on measures from tractometry
1) Separate the sample given a particular variable (group_by) into groups

2) Does Shapiro-Wilk test of normality for every sample
https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test

3) Does Levene or Bartlett (depending on normality) test of variance homogeneity
Levene:
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

6) Generate the result for all measures and bundles
"""


def _build_arg_parser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_json', metavar='IN_JSON',
                   help='input JSON file from tractometry nextflow pipeline'
                        ' or equivalent')

    p.add_argument('in_participants', metavar='IN_PARTICIPANTS',
                   help='input tsv participants file'
                   'See doc in scilpy/doc/construct_participants_tsv_file.md')

    p.add_argument('group_by', metavar='GROUP',
                   help='variable that will be used to compare group together '
                        'Ex : Sex, Number of children, ...')

    p.add_argument('o_json', metavar='O_JSON',
                   help='The name of the result json output file.'
                        'No need to add the extension .json')

    p.add_argument('--bundles', '-b',
                   default='all',
                   help='bundle(s) in which you want to do stats '
                   'Ex : -b CC_front,AF_L')

    p.add_argument('--measures', '-m',
                   default='all',
                   help='metric(s) on which you want to do stats '
                   'Ex : -m fa,ad')

    p.add_argument('--values', '-v',
                   default='all',
                   help='value(s) on which you want to do stats '
                   'Ex : -v mean')

    p.add_argument('--output_directory', '-o',
                   default='.',
                   help='name of the output folder path')

    p.add_argument('--alpha_error', '-a',
                   default=0.05,
                   help='Type 1 error for all the test')

    p.add_argument('--generate_graph', '--gg',
                   action='store_true',
                   help='Generate a simple plot of every metric across groups')

    p.add_argument('--generate_csv', '--gc',
                   action='store_true',
                   help='Generate the result of the script in a csv format')

    p.add_argument('--logging', '-l',
                   default='WARNING',
                   choices=['DEBUG', 'WARNING', 'ERROR'],
                   help='logging level you want to see:'
                        'DEBUG,WARNING,ERROR')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=args.logging)

    # Verify if folder exist if not create it
    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    required_args = [args.in_json, args.in_participants]

    bundles = args.bundles.split(",")
    measures = args.measures.split(",")
    values = args.values.split(",")
    assert_inputs_exist(parser, required_args)

    alpha_error = float(args.alpha_error)

    # We generated the stats object
    my_data = data_for_stat(args.in_json,
                            args.in_participants)
    if bundles == ['all']:
        bundles = my_data.get_bundles_list()
    if measures == ['all']:
        measures = my_data.get_measures_list()
    if values == ['all']:
        values = my_data.get_values_list()

    my_group_dict = my_data.get_groups_dictionnary(args.group_by)
    # Initialise the result dictionnary
    result_dict = {}

    # We do the comparison for every single combianaison of metric-bundle-value
    for b, m, v in product(bundles, measures, values):
        # First we extract the basic information on that comparison

        curr_comparison_measure = ('_').join([b, m, v])

        logging.debug('______________________')
        logging.debug('Measure to compare: {}'.format(curr_comparison_measure))

        # Check normality of that metric across all groups

        current_normality = {}
        overall_normality = True
        groups_array = []
        for group in my_group_dict:
            curr_sample = get_group_data_sample(my_group_dict, group, b, m, v)
            logging.debug('Group {}'.format(group))
            current_normality[group] = verify_normality(
                                            curr_sample,
                                            alpha_error)
            if not current_normality[group][0]:
                overall_normality = False
            groups_array.append(curr_sample)
        logging.debug('Normality result:')
        logging.debug(current_normality)
        logging.debug('Overall Normality:')
        logging.debug(overall_normality)
        logging.debug('Groups array:')
        logging.debug(groups_array)

        # Generate graph of the metric
        if args.generate_graph:
            visualise_distribution(groups_array,
                                   my_data.get_participants_list(),
                                   b, m, v,
                                   args.output_directory,
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
        logging.debug('Equality of variance result:')
        logging.debug(variance_equality)

        # Now we compare the groups population

        difference = verify_group_difference(
                                    groups_array,
                                    normality=overall_normality,
                                    homoscedasticity=variance_equality[1],
                                    alpha=alpha_error)
        logging.debug('Main test result:')
        logging.debug(difference)

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

        logging.debug('Summary of difference 2 by 2:')
        logging.debug(diff_2_by_2)

        # Write the current metric in the report
        curr_dict = write_current_dictionnary(curr_comparison_measure,
                                              current_normality,
                                              variance_equality,
                                              difference,
                                              diff_2_by_2)
        result_dict[curr_comparison_measure] = curr_dict

    # Saving the result dictionnary into a json file and csv if necessary

    if args.o_json.endswith(".json"):
        base = os.path.splitext(args.o_json)
        with open(os.path.join(args.output_directory,
                               args.o_json), "w") as fp:
            json.dump(result_dict, fp, indent=4)
        if args.generate_csv:
            with open(os.path.join(args.output_directory,
                                   base[0] + ".csv"), "w") as fp:
                csv_writer = csv.writer(fp, delimiter=',')
                write_csv_from_json(csv_writer, result_dict)
    elif args.o_json.endswith(".csv"):
        base = os.path.splitext(args.o_json)
        with open(os.path.join(args.output_directory,
                               base[0] + ".json"), "w") as fp:
            json.dump(result_dict, fp, indent=4)
        if args.generate_csv:
            with open(os.path.join(args.output_directory,
                                   args.o_json), "w") as fp:
                csv_writer = csv.writer(fp, delimiter=',')
                write_csv_from_json(csv_writer, result_dict)
    else:
        with open(os.path.join(args.output_directory,
                               args.o_json + ".json"), "w") as fp:
            json.dump(result_dict, fp, indent=4)
        if args.generate_csv:
            with open(os.path.join(args.output_directory,
                                   args.o_json + ".csv"), "w") as fp:
                csv_writer = csv.writer(fp, delimiter=',')
                write_csv_from_json(csv_writer, result_dict)


if __name__ == '__main__':
    main()
