# -*- coding: utf-8 -*-

import itertools
import glob
import os

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 2
_version_minor = 0
_version_micro = 1
_version_extra = ''

# Construct full version string from these.
_ver = [_version_major, _version_minor, _version_micro]

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
MAINTAINER = "Arnaud Boré"
MAINTAINER_EMAIL = "arnaud.bore@gmail.com"
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
LEGACY_SCRIPTS = filter(lambda s: not os.path.basename(s) == "__init__.py",
                        glob.glob("scripts/legacy/*.py"))
SCRIPTS = filter(lambda s: not os.path.basename(s) == "__init__.py",
                 glob.glob("scripts/*.py"))

PREVIOUS_MAINTAINERS = ["Jean-Christophe Houde"]
