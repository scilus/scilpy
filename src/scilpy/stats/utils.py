#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import logging
import math
import os

from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


class data_for_stat(object):

    """
    Method 'init' in the name will initialise argument of the object
    Method 'get' in the name return an object generated from the object
    """

    def __init__(self, json_file, participants):
        # Initialise argument of data_for_stat generated

        self.data_dictionnary = {}

        # Initialise dictionnary

        """
        Open the json and tsv file and put the information in a dictionnary
        """

        with open(json_file) as f:
            json_info = json.load(f)

        with open(participants) as f:
            csv_reader = csv.DictReader(f, delimiter=' ')
            participants_info = [v for v in csv_reader]

        # Validate the files are compatible
        self.validation_participant_id(json_info, participants_info)

        # Construct the data dictionnary
        for participant in participants_info:
            self.data_dictionnary.update({
                participant['participant_id']:
                    {'bundles': json_info[participant['participant_id']]}})

            for variable in participant:
                if variable != 'participant_id':
                    self.data_dictionnary[participant['participant_id']]\
                        [variable] = participant[variable]

        logging.info('Data_dictionnary')
        logging.info(self.data_dictionnary[self.get_first_participant()])

        with open('data.json', 'w') as fp:
            json.dump(self.data_dictionnary, fp, indent=4)

    def validation_participant_id(self, json_info, participants_info):
        """
        Verify if the json and tsv file has the same participants id
        """
        # Create the list of participants id from the json dictionnary

        participants_from_json = list(json_info.keys())
        logging.info('participant list from json dictionnary:')
        logging.info(participants_from_json)

        # Create the list of participants id from the tsv list of dictionnary
        participants_from_tsv = []
        for participant in participants_info:
            participants_from_tsv.append(participant['participant_id'])
        logging.info('participant list from tsv file:')
        logging.info(participants_from_tsv)

        # Compare the two list
        participants_from_json.sort()
        participants_from_tsv.sort()

        if not participants_from_json == participants_from_tsv:
            if not len(participants_from_json) == len(participants_from_tsv):
                logging.info('The number of participants from json file is '
                             'not the same as the one in the tsv file.')
            is_in_tsv = np.in1d(participants_from_json, participants_from_tsv)
            is_in_json = np.in1d(participants_from_tsv, participants_from_json)

            logging.info('participants list from json file missing in tsv '
                         'file :')
            logging.info(np.asarray(participants_from_json)[~is_in_tsv])
            logging.info('participants list from tsv file missing in json '
                         'file :')
            logging.info(np.asarray(participants_from_tsv)[~is_in_json])

            logging.error('The subjects from the json file does not fit '
                          'with the subjects of the tsv file. '
                          'Impossible to build the data_for_stat object')
            raise BaseException('The subjects from the json file does not fit '
                                'with the subjects of the tsv file. '
                                'Impossible to build the data_for_stat object')
        else:
            logging.info('The json and the tsv are compatible')

    def get_participants_list(self):
        # Construct the list of participant_id from the data_dictionnary

        return list(self.data_dictionnary.keys())

    def get_first_participant(self):
        # Get the first participant
        return next(iter(self.data_dictionnary))

    def get_first_bundle(self, participant):
        # Get the first bundle
        return next(iter(self.data_dictionnary[participant]['bundles']))

    def get_first_metric(self, participant, bundle):
        # Get the first metric key
        return next(iter(self.data_dictionnary[participant]['bundles'][bundle]))

    def get_participant_attributes_list(self):
        # Construct the list of attribute from data_dictionnary
        # We take the attributes from first participant
        # (assume to be consistent across participants)
        first_participant = self.get_first_participant()
        attributes_list = list(self.data_dictionnary[first_participant].keys())
        attributes_list.remove('bundles')
        return attributes_list

    def get_bundles_list(self):
        # Construct the list of bundles_id from data_dictionnary
        # We take the bundles from first participant
        # (assume to be consistent across participants)
        first_participant = self.get_first_participant()
        return list(self.data_dictionnary[first_participant]['bundles'].keys())

    def get_metrics_list(self):
        # Construct the list of metrics_id from data_dictionnary
        # We take the metrics from first participant
        # (assume to be consistent across participants)
        first_participant = self.get_first_participant()
        first_bundle = self.get_first_bundle(first_participant)
        return list(self.data_dictionnary[first_participant]
                                         ['bundles'][first_bundle].keys())

    def get_values_list(self):
        # Construct the list of values_id from data_dictionnary
        # We take the values from first participant
        # (assume to be consistent across participants)

        first_participant = self.get_first_participant()
        first_bundle = self.get_first_bundle(first_participant)
        first_metric = self.get_first_metric(first_participant,
                                             first_bundle)
        return list(self.data_dictionnary[first_participant]
                                         ['bundles']
                                         [first_bundle][first_metric].keys())

    def get_groups_dictionnary(self, group_by):
        """
        Parameters
        ----------
        groups_by: string
            The attribute with which we generate our groups.
        Returns
        -------
        group_dict: dictionnary of groups
            keys: group id generated by group_by.
            values: dictionnary of participants of that specific group.
        """
        group_dict = {}
        # Verify if the group_by exist
        if group_by not in self.get_participant_attributes_list():
            logging.error('Participants doesn\'t contain the attribute '
                          '{} necessary to generate groups.'.format(group_by))
            raise BaseException('Object data_for_stat has no attirbute '
                                '{}.'.format(group_by))
        # Get the participants separated by group
        for participant in self.data_dictionnary:
            curr_group_id = (group_by + '_' +
                             self.data_dictionnary[participant][group_by])
            if curr_group_id in group_dict.keys():
                group_dict[curr_group_id][participant] = \
                    self.data_dictionnary[participant]

            else:
                group_dict.update({curr_group_id:
                                  {participant: self.data_dictionnary[participant]}})

        return group_dict

    def get_groups_list(self, group_by):
        """
        Parameters
        ----------
        groups_by: string
            The attribute with which we generate our groups.
        Returns
        -------
        group_list: list of string
            list of group id generated by group_by variable.
        """
        # Generated the list of group generated by group_by variable
        return list(self.get_groups_dictionnary(group_by).keys())

    def get_data_sample(self, bundle, metric, value):
        """
        Parameters
        ----------
        bundle: string
            The specific bundle with which we generate our sample.
        metric: string
            The specific metric with which we generate our sample.
        value: string
            The specific value with which we generate our sample.
        Returns
        -------
        data_sample: array of float
            The sample array associate with the parameters.
        """
        data_sample = []
        for participant in self.data_dictionnary:
            data_sample.append(self.data_dictionnary[participant]
                                                    ['bundles']
                                                    [bundle]
                                                    [metric][value])
        return data_sample


def get_group_data_sample(group_dict, group_id, bundle, metric, value):
    """
    Parameters
    ----------
    group_dict: dictionnary of groups
        keys: group id generated by group_by.
        values: dictionnary of participants of that specific group.
    group_id: string
        The name of the group with which we generate our sample.
    bundle: string
        The specific bundle with which we generate our sample.
    metric: string
        The specific metric with which we generate our sample.
    value: string
        The specific value with which we generate our sample.
    Returns
    -------
    data_sample: array of float
        The sample array associate with the parameters.
    """
    sample_size = len(group_dict[group_id].keys())
    data_sample = np.zeros(sample_size)
    for index, participant in enumerate(group_dict[group_id].keys()):
        if bundle in group_dict[group_id][participant]['bundles']:
            # Assure the participants has the bundle in the database
            if metric in group_dict[group_id][participant]['bundles'][bundle]:
                # Assure the participants has the metric in the database
                try:
                    data_sample[index] = group_dict[group_id][participant]\
                                                             ['bundles']\
                                                             [bundle]\
                                                             [metric][value]
                except KeyError:
                    print('{} is not a valid key.'.format(value))

    return data_sample


def write_current_dictionnary(metric, normality, variance_equality,
                              diff_result, diff_2_by_2):
    """
    Parameters
    ----------
    metric: string
        The name of the metric in which the group comparison was made on.
    normality: dictionnary of groups
        keys: group id
        values: (result, p-value)
    variance_equality: (string, bool)
        The result of the equality of variance test.

        - 1st dimension:
            Name of the equal variance test done.
        - 2nd dimension:
            Whether or not it equality of variance can be assumed.
    diff_result: (string, bool, float)
        The result of the groups difference analysis on the metric.

        - 1st dimension:
            Name of the test done.
        - 2nd dimension:
            Whether or not we detect a group difference on the metric.
        - 3rd dimension:
            p-value result
    diff_2_by_2: (list of (string, string, bool, float), string)
        The result of the pairwise groups difference a posteriori analysis.

        - 1st dimension:
            Name of the test done.
        - 2nd dimension:
            The result of every pairwise combinations of the groups
            (name of first group, name of second group, result, p-value).

    Returns
    -------
    curr_dict: dictionnary of test
    keys: The category of test done (Normality, Homoscedascticity,...)
    values: The result of those test.
    """
    # First, we create the structure
    curr_dict = {
            'Normality': {'Test': 'Shapiro-Wilk',
                          'P-value': {}},
            'Homoscedasticity': {'Test': variance_equality[0]},
            'Group difference': {'Test': diff_result[0]}
        }
    # Normality
    for group in normality:
        if normality[group][0]:
            curr_dict['Normality']['P-value'][group] = 'Normal ('\
                                                       + str(normality[group][1])\
                                                       + ')'
        else:
            curr_dict['Normality']['P-value'][group] = 'Not Normal ('\
                                                       + str(normality[group][1])\
                                                       + '*)'
    # Equality of variance
    if variance_equality[1]:
        curr_dict['Homoscedasticity']['P-value'] = 'Equal variance ('\
                                                   + str(variance_equality[2])\
                                                   + ')'
    else:
        curr_dict['Homoscedasticity']['P-value'] = 'Not equal variance ('\
                                                   + str(variance_equality[2])\
                                                   + '*)'

    # Main test
    if diff_result[1] and len(diff_2_by_2) != 0:
        curr_dict['Group difference']['P-value'] = str(diff_result[2]) + '*'
        # Pairwise difference
        curr_dict['Pairwise group difference'] = {'Test': diff_2_by_2[0],
                                                  'P-value': {}}
        for i in range(len(diff_2_by_2[1])):
            if diff_2_by_2[1][i][0] < diff_2_by_2[1][i][1]:
                curr_comparison = '{} vs {}'.format(diff_2_by_2[1][i][0],
                                                    diff_2_by_2[1][i][1])
            else:
                curr_comparison = '{} vs {}'.format(diff_2_by_2[1][i][1],
                                                    diff_2_by_2[1][i][0])
            if diff_2_by_2[1][i][2]:
                curr_dict['Pairwise group difference']\
                         ['P-value']\
                         [curr_comparison] = str(diff_2_by_2[1][i][3]) + '*'
            else:
                curr_dict['Pairwise group difference']\
                         ['P-value']\
                         [curr_comparison] = str(diff_2_by_2[1][i][3])
    elif diff_result[1]:
        curr_dict['Group difference']['P-value'] = str(diff_result[2]) + '*'
    else:
        curr_dict['Group difference']['P-value'] = str(diff_result[2])

    return curr_dict


def write_csv_from_json(writer, json_dict):
    # Header
    first = next(iter(json_dict))
    nb_group = len(json_dict[first]['Normality']['P-value'].keys())
    groups_list = list(json_dict[first]['Normality']['P-value'].keys())
    groups_list.sort()
    n_blank = [''] * nb_group
    n_pvalue = ['p-value ' + x for x in groups_list]

    if nb_group > 2:
        nb_pairwise = math.factorial(nb_group) / (2 * math.factorial(nb_group - 2))
        pairwise_list = []
        for x, y in combinations(range(nb_group), 2):
            pairwise_list.append(groups_list[x] + ' vs ' + groups_list[y])
        pairwise_list.sort()
        pd_blank = [''] * nb_pairwise
        pd_pvalue = ['p-value ' + x for x in pairwise_list]

        writer.writerow(['Test: ', 'Normality'] + n_blank +
                        ['Homoscedasticity', '', 'Group difference', '',
                        'Pairwise difference'] + pd_blank)
        writer.writerow(['Metric', 'Test name'] + n_pvalue +
                        ['Test name', 'p-value', 'Test name', 'p-value',
                        'Test name'] + pd_pvalue)
    else:
        writer.writerow(['Test: ', 'Normality'] + n_blank +
                        ['Homoscedasticity', '', 'Group difference', ''])
        writer.writerow(['Metric', 'Test name'] + n_pvalue +
                        ['Test name', 'p-value', 'Test name'])

    # Now the result for every metrics
    metrics = list(json_dict.keys())
    metrics.sort()
    for i in range(len(metrics)):
        # Normality
        curr_pvalue_n = []
        for j in range(len(groups_list)):
            curr_pvalue_n.append(json_dict[metrics[i]]
                                          ['Normality']
                                          ['P-value'][groups_list[j]])
        curr_n = [json_dict[metrics[i]]['Normality']['Test']] + curr_pvalue_n

        # Homoscedasticity
        curr_h = [json_dict[metrics[i]]['Homoscedasticity']['Test'],
                  json_dict[metrics[i]]['Homoscedasticity']['P-value']]

        # Group difference
        curr_gd = [json_dict[metrics[i]]['Group difference']['Test'],
                   json_dict[metrics[i]]['Group difference']['P-value']]

        # Pairwise difference
        if 'Pairwise group difference' in json_dict[metrics[i]].keys():
            curr_pvalue_pd = []
            for j in range(len(pairwise_list)):
                curr_pvalue_pd.append(json_dict[metrics[i]]
                                               ['Pairwise group difference']
                                               ['P-value'][pairwise_list[j]])
            curr_pd = [json_dict[metrics[i]]['Pairwise group difference']
                                            ['Test']] + curr_pvalue_pd
            writer.writerow([metrics[i]] + curr_n + curr_h + curr_gd + curr_pd)
        else:
            writer.writerow([metrics[i]] + curr_n + curr_h + curr_gd)


def visualise_distribution(data_by_group, participants_id, bundle, metric,
                           value, oFolder, groups_list):
    """
    Parameters
    ----------
    data_by_group: list of array_like
        The sample data separated by groups.
        Possibly of different lengths per group.
    participants_id: list of string
        Names of the participants id "name".
    metric: string
        The name of the metricment in which you want to look at the
        across groups.
    oFolder: path-like object
        Emplacement in which we want to save the graph of the distribution
        the measurement across groups.
    groups_list: list of string
        The names of each group.
    Returns
    -------
    outliers: list of (string, string)
        The list of participants that is considered outlier for their group
        (participant_id, group_id).
    """
    nb_group = len(data_by_group)
    outliers = []
    fig, ax = plt.subplots()
    ls = np.asarray(data_by_group, dtype=object)
    ax.boxplot(ls)
    boxdict = ax.boxplot(ls)
    fliers = boxdict['fliers']
    # loop over boxes in x direction
    for j in range(len(fliers)):
        # the y and x positions of the fliers
        yfliers = boxdict['fliers'][j].get_ydata()
        xfliers = boxdict['fliers'][j].get_xdata()

        # the unique locations of fliers in y
        ufliers = set(yfliers)

        # loop over unique fliers
        for i, uf in enumerate(ufliers):
            # search subject id
            curr_group = int(round(xfliers[i])) - 1
            subjects_id = np.nonzero(np.isclose(data_by_group[curr_group],
                                     yfliers[i]))[0].tolist()
            tag = ""
            for e in subjects_id:
                tag += "  " + participants_id[e] + " "
                outliers.append((participants_id[e], groups_list[curr_group]))
            # print number of fliers
            ax.text(1.005 * xfliers[i], 1.002 * uf, tag)
    # Name axes
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for k in range(nb_group):
        labels[k] = groups_list[k]
    ax.set_xticklabels(labels)
    plt.ylabel(metric)
    plt.title("Distribution of the {} data set in bundle {}.".format(metric,
                                                                     bundle))

    save_path = os.path.join(oFolder, 'Graph', bundle, 'plot_' + metric)

    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc:
            if exc.errno != exc.EEXIST:
                raise

    fig.savefig(os.path.join(oFolder, 'Graph', bundle, metric))

    logging.info('outliers:[(id, group)]')
    logging.info(outliers)
    return outliers
