# -*- coding: utf-8 -*-

import numpy as np


def merge_dict(dict_1, dict_2, no_list=False, recursive=False):
    """
    Merge two dictionary (used for tractometry)

    Parameters
    ------
    dict_1: dict
        dictionary to merge
    dict_2: dict
        dictionary to merge
    no_list: boolean
        If False it merge list else it will create a list of list

    recursive: boolean
        If true it will try to go as deep as possible in the dict
         to get to the list or value instead of updating dictionaries

    Returns
    -------
    new_dict:  dict
        Merged dictionary
    """
    new_dict = {}
    for key in dict_1.keys():
        new_dict[key] = dict_1[key]

    for key in dict_2.keys():
        if isinstance(dict_2[key], dict) and recursive:
            if key not in dict_1:
                dict_1[key] = {}
            new_dict[key] = merge_dict(dict_1[key], dict_2[key],
                                       no_list=no_list, recursive=recursive)
        elif key not in new_dict:
            new_dict[key] = dict_2[key]
        else:
            if not isinstance(new_dict[key], list) and not no_list:
                new_dict[key] = [new_dict[key]]

            if not isinstance(dict_2[key], list) and not no_list:
                new_dict[key].extend([dict_2[key]])
            else:
                if isinstance(dict_2[key], dict):
                    new_dict.update(dict_2)
                else:
                    new_dict[key] = new_dict[key] + dict_2[key]

    return new_dict


def average_dict(curr_dict):
    """
    Compute the mean and std of a metric in a json file

    Parameters
    ------
    curr_dict: dict


    Returns
    -------
    dictionary with mean and std computed
    """
    for key in curr_dict.keys():
        if isinstance(curr_dict[key], dict):
            curr_dict[key] = average_dict(curr_dict[key])
        elif isinstance(curr_dict[key], list) or np.isscalar(curr_dict[key]):
            new_dict = {}
            for subkey in curr_dict.keys():
                new_dict[subkey] = {'mean': np.average(curr_dict[subkey]),
                                    'std': np.std(curr_dict[subkey])}
            return new_dict

    return curr_dict
