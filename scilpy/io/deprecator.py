# -*- coding: utf-8 -*-

from dipy.utils.deprecator import cmp_pkg_version, ExpiredDeprecationError
from functools import wraps
from packaging.version import parse
import warnings

from scilpy import version


class ScilpyExpiredDeprecation(ExpiredDeprecationError):
    pass


DEFAULT_SEPARATOR = "="
DEFAULT_DEPRECATION_WINDOW = 2  # Wait for 2 minor releases before rasing error

DEPRECATION_HEADER = """
!!! WARNING !!! THIS SCRIPT IS DEPRECATED !!!
"""

DEPRECATION_FOOTER = """
AS OF VERSION {EXP_VERSION}, CALLING THIS SCRIPT WILL RAISE {EXP_ERROR}
"""

EXPIRATION_FOOTER = """
SCRIPT {SCRIPT_NAME} HAS BEEN REMOVED SINCE {EXP_VERSION}
"""

SEPARATOR_BLOCK = """
{UP_SEPARATOR}
{MESSAGE}
{LOW_SEPARATOR}
"""


def _block(_msg, _sep_len=80):
    _sep = f"{DEFAULT_SEPARATOR * _sep_len}"
    return SEPARATOR_BLOCK.format(
        UP_SEPARATOR=_sep, MESSAGE=_msg, LOW_SEPARATOR=_sep)


def _header(_msg, _sep_len=80):
    _sep = f"{DEFAULT_SEPARATOR * _sep_len}"
    return SEPARATOR_BLOCK.format(
        UP_SEPARATOR=_sep, MESSAGE=_msg, LOW_SEPARATOR="").rstrip("\n")


def _raise_warning(header, footer, func, *args, **kwargs):
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(header, DeprecationWarning, stacklevel=4)

    try:
        return func(*args, **kwargs)
    finally:
        print("")
        warnings.warn(footer, DeprecationWarning, stacklevel=4)


def deprecate_script(script, message, from_version):
    from_version = parse(from_version)
    expiration_minor = from_version.minor + DEFAULT_DEPRECATION_WINDOW
    expiration_version = f"{from_version.major}.{expiration_minor}"
    current_version = f"{version._version_major}.{version._version_minor}"

    def _deprecation_decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            if cmp_pkg_version(current_version, expiration_version) >= 0:
                footer = f"""\
                          {message}
                          {EXPIRATION_FOOTER.format(
                              SCRIPT_NAME=script,
                              EXP_VERSION=expiration_version)}\
                          """

                raise ScilpyExpiredDeprecation(f"""\
                                                {_header(DEPRECATION_HEADER)}
                                                {_block(footer)}
                                                """)
            else:
                header = DEPRECATION_HEADER
                footer = f"""\
                         {message}
                         {DEPRECATION_FOOTER.format(
                             EXP_VERSION=expiration_version,
                             EXP_ERROR=ScilpyExpiredDeprecation)}\
                         """

                msg_length = max(
                    len(_l) for _l in (header + footer).splitlines())

                return _raise_warning(_block(header, msg_length),
                                      _block(footer, msg_length),
                                      func, *args, **kwargs)

        return _wrapper

    return _deprecation_decorator
