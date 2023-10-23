# -*- coding: utf-8 -*-

import pytest


def create_mock(module_name, object_name, mocker, apply_mocks,
                side_effect=None):

    if apply_mocks:
        return mocker.patch(
            "{}.{}".format(module_name, object_name),
                           side_effect=side_effect, create=True)

    return None
