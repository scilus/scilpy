# -*- coding: utf-8 -*-

from scilpy.tests.dict import (dict_to_average, expected_dict_averaged,
                               dict_1, dict_2,
                               expected_merged_dict_12,
                               expected_merged_dict_12_recursive,
                               expected_merged_dict_12_all_true)
from scilpy.tractanalysis.json_utils import merge_dict, average_dict


def test_merge_dict_default():
    assert expected_merged_dict_12 == merge_dict(dict_1, dict_2)


def test_merge_dict_recursive():
    assert expected_merged_dict_12_recursive == merge_dict(dict_1,
                                                           dict_2,
                                                           recursive=True)


def test_merge_dict_all_true():
    assert expected_merged_dict_12_all_true == merge_dict(dict_1, dict_2,
                                                          no_list=True,
                                                          recursive=True)


def test_average_dict():
    assert expected_dict_averaged == average_dict(dict_to_average)
