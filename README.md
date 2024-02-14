# Scilpy
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/scilus/scilpy)](https://github.com/scilus/scilpy/releases)
[![Build Status](https://travis-ci.com/scilus/scilpy.svg?branch=master)](https://travis-ci.com/scilus/scilpy)
[![codecov](https://codecov.io/github/scilus/scilpy/graph/badge.svg?token=oXjDog4YZG)](https://codecov.io/github/scilus/scilpy)
[![Documentation Status](https://readthedocs.org/projects/scilpy/badge/?version=latest)](https://scilpy.readthedocs.io/en/latest/?badge=latest)

[![PyPI version badge](https://img.shields.io/pypi/v/scilpy?logo=pypi&logoColor=white)](https://pypi.org/project/scilpy)
[![PyPI - Downloads](https://static.pepy.tech/badge/scilpy)](https://pypi.org/project/scilpy)

[![Docker container badge](https://img.shields.io/docker/v/scilus/scilus?label=docker&logo=docker&logoColor=white)](https://hub.docker.com/r/scilus/scilus)

**Scilpy** is the main library supporting research and development at the Sherbrooke Connectivity Imaging Lab
([SCIL]).

**Scilpy** mainly comprises tools and utilities to quickly work with diffusion MRI. Most of the tools are based
on or are wrappers of the [DIPY] library, and most of them will eventually be migrated to [DIPY]. Those tools implement the recommended workflows and parameters used in the lab.

The library is now built for Python 3.10 so be sure to create a virtual environnement for Python 3.10. If this version is not installed on your computer:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.10 python3.10-dev python3.10-venv python3.10-minimal python3.10-tk
```

Make sure your pip is up-to-date before trying to install:
```
pip install --upgrade pip
```

The library's structure is mostly aligned on that of [DIPY].

The library and scripts can be installed locally by using:
```
pip install -e .
```

If you don't want to install legacy scripts:
```
export SCILPY_LEGACY='False'
pip install -e .
```

(Then, without the legacy scripts, if you want to use pytest, use:)
```
pytest --ignore=scripts/legacy
```

On Linux, most likely you will have to install libraries for COMMIT/AMICO
```
sudo apt install libblas-dev liblapack-dev
```

While on MacOS you will have to use (most likely)
```
brew install openblas lapack
```

On Ubuntu >=20.04, you will have to install libraries for matplotlib
```
sudo apt install libfreetype6-dev
```

Note that using this technique will make it harder to remove the scripts when changing versions.
We highly recommend working in a [Python Virtual Environment].

[SCIL]:http://scil.dinf.usherbrooke.ca/
[DIPY]:http://dipy.org
[Python Virtual Environment]:https://virtualenv.pypa.io/en/latest/

**Scilpy** documentation is available: https://scilpy.readthedocs.io/en/latest/
