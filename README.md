# Scilpy
[![Build Status](https://travis-ci.org/scilus/scilpy.svg?branch=master)](https://travis-ci.org/scilus/scilpy)
[![Documentation Status](https://readthedocs.org/projects/scilpy/badge/?version=latest)](https://scilpy.readthedocs.io/en/latest/?badge=latest)

**Scilpy** is the main library supporting research and development at the Sherbrooke Connectivity Imaging Lab
([SCIL]).

**Scilpy** mainly comprises tools and utilities to quickly work with diffusion MRI. Most of the tools are based
on or are wrappers of the [DIPY] library, and most of them will eventually be migrated to [DIPY]. Those tools implement the recommended workflows and parameters used in the lab.

The library's structure is mostly aligned on that of [DIPY].
The library and scripts can be installed locally by using:
```
pip install -e .
```
Note that using this technique will make it harder to remove the scripts when changing versions.
We highly recommend working in a [Python Virtual Environment].

[SCIL]:http://scil.dinf.usherbrooke.ca/
[DIPY]:http://dipy.org
[Python Virtual Environment]:https://virtualenv.pypa.io/en/latest/

**Scilpy** documentation is available: https://scilpy.readthedocs.io/en/latest/
