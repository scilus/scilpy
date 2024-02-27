# -*- coding: utf-8 -*-

import logging

from itertools import combinations

import scipy.stats


def verify_normality(data, alpha=0.05):
    """
    Parameters
    ----------
    data: array_like
        Array of sample data to test normality on.
        Should be of 1 dimension.
    alpha: float
        Type 1 error of the normality test.
        Probability of false positive or rejecting null hypothesis
        when it is true.

    Returns
    -------
    normality: bool
        Whether or not the sample can be considered normal
    p_value: float
        Probability to obtain an effect at least as extreme as the one
        in the current sample, assuming the null hypothesis.
        We reject the null hypothesis when this value is lower than alpha.
    """

    normality = True
    # First, we verify if sample pass Shapiro-Wilk test
    W, p_value = scipy.stats.shapiro(data)
    if p_value < alpha and len(data) < 30:
        logging.info('The data sample can not be considered normal')
        normality = False
    else:
        logging.info('The data sample pass the normality assumption.')
        normality = True
    return normality, p_value


def verify_homoscedasticity(data_by_group, normality=False, alpha=0.05):
    """
    Parameters
    ----------
    data_by_group: list of array_like
        The sample data separated by groups.
        Possibly of different group size.
    normality: bool
        Whether or not the sample data of each groups can be considered normal
    alpha: float
        Type 1 error of the equality of variance test
        Probability of false positive or rejecting null hypothesis
        when it is true.

    Returns
    -------
    test: string
        Name of the test done to verify homoscedasticity
    homoscedasticity: bool
        Whether or not the equality of variance across groups can be assumed
    p_value: float
        Probability to obtain an effect at least as extreme as the one
        in the current sample, assuming the null hypothesis.
        We reject the null hypothesis when this value is lower than alpha.
    """
    total_nb = 0
    for group in data_by_group:
        nb_current_group = len(group)
        total_nb += nb_current_group
    mean_nb = total_nb / len(data_by_group)

    if normality:
        test = 'Bartlett'
        W, p_value = scipy.stats.bartlett(*data_by_group)
    else:
        test = 'Levene'
        W, p_value = scipy.stats.levene(*data_by_group)
    logging.info('Test name: {}'.format(test))
    if p_value < alpha and mean_nb < 30:
        logging.info('The sample didnt pass the equal variance assumption')
        homoscedasticity = False
    else:
        logging.info('The sample pass the equal variance assumption')
        homoscedasticity = True

    return test, homoscedasticity, p_value


def verify_group_difference(data_by_group, normality=False,
                            homoscedasticity=False, alpha=0.05):
    """
    Parameters
    ----------
    data_by_group: list of array_like
        The sample data separated by groups.
        Possibly of different group size.
    normality: bool
        Whether or not the sample data of each groups can be considered normal.
    homoscedasticity: bool
        Whether or not the equality of variance across groups can be assumed.
    alpha: float
        Type 1 error of the equality of variance test.
        Probability of false positive or rejecting null hypothesis
        when it is true.
    Returns
    -------
    test: string
        Name of the test done to verify group difference.
    difference: bool
        Whether or not the variable associated for groups has an effect on
        the current measurement.
    p_value: float
        Probability to obtain an effect at least as extreme as the one
        in the current sample, assuming the null hypothesis.
        We reject the null hypothesis when this value is lower than alpha.
    """

    if len(data_by_group) == 2:
        if normality and homoscedasticity:
            test = 'Student'
            T, p_value = scipy.stats.ttest_ind(data_by_group[0],
                                               data_by_group[1])
        elif normality:
            test = 'Welch'
            T, p_value = scipy.stats.ttest_ind(data_by_group[0],
                                               data_by_group[1],
                                               equal_var=False)
        else:
            test = 'Mannwhitneyu'
            T, p_value = scipy.stats.mannwhitneyu(data_by_group[0],
                                                  data_by_group[1])
            # We are doing a 2 tail test
            p_value = 2 * p_value
            size_too_small = [len(data_by_group[0]) < 15,
                              len(data_by_group[1]) < 15]
            if not all(size_too_small):
                logging.warning('The power of the mann and withney u test'
                                ' might be low due to small sample size')
    elif len(data_by_group) > 2:
        if normality and homoscedasticity:
            test = 'ANOVA'
            T, p_value = scipy.stats.f_oneway(*data_by_group)
        else:
            test = 'Kruskalwallis'
            T, p_value = scipy.stats.kruskal(*data_by_group)

    logging.info('Test name: {}'.format(test))
    if p_value < alpha:
        logging.info('There is a difference between groups')
        difference = True
    else:
        logging.info('We are not able to detect difference between'
                     ' the groups.')
        difference = False

    return test, difference, p_value


def verify_post_hoc(data_by_group, groups_list, test,
                    correction=True, alpha=0.05):
    """
    Parameters
    ----------
    data_by_group: list of array_like
        The sample data separated by groups.
        Possibly of different lengths group size.
    groups_list: list of string
        The names of each group in the same order as data_by_group.
    test: string
        The name of the post-hoc analysis test to do.
        Post-hoc analysis is the analysis of pairwise difference a posteriori
        of the fact that there is a difference across groups.
    correction: bool
        Whether or not to do a Bonferroni correction on the alpha threshold.
        Used to have a more stable type 1 error across multiple comparison.
    alpha: float
        Type 1 error of the equality of variance test.
        Probability of false positive or rejecting null hypothesis
        when it is true.

    Returns
    -------
    differences: list of (string, string, bool)
        The result of the post-hoc for every groups pairwise combinations.

        - 1st, 2nd dimension: Names of the groups chosen.
        - 3rd: Whether or not we detect a pairwise difference on the current
          measurement.
        - 4th: P-value of the pairwise difference test.
    test: string
        Name of the test done to verify group difference
    """
    logging.info('We need to do a post-hoc analysis since '
                 'there is a difference')
    logging.info('Post-hoc: {} pairwise'.format(test))
    differences = []
    nb_group = len(groups_list)

    # Bonferroni correction
    if correction:
        nb_combinaison = (nb_group * (nb_group - 1)) / 2
        alpha = alpha / nb_combinaison

    for x, y in combinations(range(nb_group), 2):
        if test == 'Student':
            T, p_value = scipy.stats.ttest_ind(
                data_by_group[x], data_by_group[y])
        elif test == 'Mannwhitneyu':
            T, p_value = scipy.stats.mannwhitneyu(
                data_by_group[x], data_by_group[y])
        elif test == 'Wilcoxon':
            T, p_value = scipy.stats.wilcoxon(
                data_by_group[x], data_by_group[y])
        differences.append((groups_list[x], groups_list[y],
                            p_value < alpha, p_value))
    logging.info('Result:')
    logging.info(differences)

    return test, differences
