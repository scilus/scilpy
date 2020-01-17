from __future__ import absolute_import, division, print_function
import glob

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "Scilpy: diffusion MRI tools and utilities"
# Long description will go up on the pypi page
long_description = """

Scilpy
========
Scilpy is a small library mainly containing small tools and utilities
to quickly work with diffusion MRI. Most of the tools are based
on or wrapper of the Dipy_ library.

.. _Dipy: http://dipy.org

License
=======
``scilpy`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2012--, Sherbrooke Connectivity Imaging Lab [SCIL],
Université de Sherbrooke.
"""

NAME = "scilpy"
MAINTAINER = "Jean-Christophe Houde"
MAINTAINER_EMAIL = "jean.christophe.houde@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/scilus/scilpy"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "The SCIL developers"
AUTHOR_EMAIL = ""
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
REQUIRES = ["bz2file (>=0.98)", "cycler (>=0.10.0)", "bz2file (>=0.10.0)",
            "cython (>=0.29.12)", "dipy (>=1.0.0)", "fury (>=0.2.0)",
            "future (>=0.17.1)", "h5py (>=2.9.0)", "kiwisolver (>=1.0.1)",
            "matplotlib (>=2.2.2)", "nibabel (>=2.2.2)", "numpy (>=1.16.2)",
            "Pillow (>=6.2.0)", "pybids (>=0.9.1)", "pyparsing (>=2.2.0)",
            "python_dateutil (>=2.7.2)", "pytz (>=2018.4)", "scipy (>=1.0.1)",
            "setuptools (>=39.0.1)", "six (>=1.11.0)", "coloredlogs (>=10.0)",
            "pytest_console_scripts (>=0.2.0)"]
SCRIPTS = glob.glob("scripts/*.py")
