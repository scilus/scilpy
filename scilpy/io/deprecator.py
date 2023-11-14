# -*- coding: utf-8 -*-

from dipy.utils.deprecator import (cmp_pkg_version,
                                   deprecate_with_version,
                                   ExpiredDeprecationError)
from functools import partial
from importlib import metadata
from packaging.version import parse


class ScilpyExpiredDeprecation(ExpiredDeprecationError):
    pass


DEFAULT_DEPRECATION_WINDOW = 1  # Wait for 1 major release
                                # before forcing script removal


def deprecate_script(message, from_version):
    # Get scilpy version using importlib, requires to wrap dipy comparator
    _version_cmp = partial(cmp_pkg_version,
                           pkg_version_str=metadata.version('scilpy'))

    _v = parse(from_version).major + DEFAULT_DEPRECATION_WINDOW
    return deprecate_with_version(message, version_comparator=_version_cmp,
                                  since=from_version, until=f"{_v}.0.0",
                                  error_class=ScilpyExpiredDeprecation)
