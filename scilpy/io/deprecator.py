# -*- coding: utf-8 -*-

from dipy.utils.deprecator import cmp_pkg_version, ExpiredDeprecationError
from functools import wraps
from importlib import metadata
from packaging.version import parse
import warnings



class ScilpyExpiredDeprecation(ExpiredDeprecationError):
    pass


DEFAULT_SEPARATOR = "="
DEFAULT_DEPRECATION_WINDOW = 2  # Wait for 2 minor releases
                                # before forcing script removal

DEPRECATION_HEADER = f"""
{DEFAULT_SEPARATOR * 68}

!!! WARNING !!! SCRIPT IS DEPRECATED AND WON'T BE AVAILABLE SOON !!!

{DEFAULT_SEPARATOR * 68}
"""


DEPRECATION_FOOTER = """
{SEPARATOR}

{MESSAGE}

AS OF VERSION {EXP_VERSION}, CALLING THIS SCRIPT WILL RAISE {EXP_ERROR}")

{SEPARATOR}
"""

EXPIRATION_FOOTER = """
{SEPARATOR}

{MESSAGE}

SCRIPT {SCRIPT_NAME} HAS BEEN DEPRECATED SINCE {EXP_VERSION}

{SEPARATOR}
"""

def _separator(_msg):
    _msg_width = max([len(_m) for _m in _msg.split("\n")])

    return f"{DEFAULT_SEPARATOR * _msg_width}"


def _format_expiration_message(_script, _msg, _exp_version):
    return EXPIRATION_FOOTER.format(
        SCRIPT_NAME=_script,
        MESSAGE=_msg,
        EXP_VERSION=_exp_version,
        SEPARATOR=_separator(EXPIRATION_FOOTER + _msg))


def _format_deprecation_message(_msg, _exp_version):
    _sep = DEPRECATION_FOOTER.format(
        MESSAGE=_msg,
        EXP_VERSION=_exp_version,
        EXP_ERROR=ScilpyExpiredDeprecation,
        SEPARATOR="")

    return DEPRECATION_FOOTER.format(
        MESSAGE=_msg,
        EXP_VERSION=_exp_version,
        EXP_ERROR=ScilpyExpiredDeprecation,
        SEPARATOR=_separator(_sep))


def _warn(_exp_version, _msg, _func, *_args, **_kwargs):
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(DEPRECATION_HEADER, DeprecationWarning, stacklevel=4)

    _error_to_raise = None
    try:
        _res = _func(*_args, **_kwargs)
    except BaseException as e:
        _error_to_raise = e

    warnings.warn(_format_deprecation_message(_msg, _exp_version),
                  DeprecationWarning, stacklevel=4)

    if _error_to_raise:
        raise _error_to_raise

    return _res


def deprecate_script(script, message, from_version):
    _from_version = parse(from_version)
    _exp_minor = _from_version.minor + DEFAULT_DEPRECATION_WINDOW
    _exp_version = f"{_from_version.major}.{_exp_minor}.0"

    def _deprecation_decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            if cmp_pkg_version(metadata.version('scilpy'), _exp_version) > 0:
                raise ScilpyExpiredDeprecation(
                    _format_expiration_message(script, message, _exp_version))
            else:
                return _warn(_exp_version, message, func, *args, **kwargs)

        return _wrapper
    
    return _deprecation_decorator
