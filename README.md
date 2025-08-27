# Scilpy
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/scilus/scilpy)](https://github.com/scilus/scilpy/releases)
[![codecov](https://codecov.io/github/scilus/scilpy/graph/badge.svg?token=oXjDog4YZG)](https://codecov.io/github/scilus/scilpy)
[![Documentation Status](https://readthedocs.org/projects/scilpy/badge/?version=latest)](https://scilpy.readthedocs.io/en/latest/?badge=latest)

[![PyPI version badge](https://img.shields.io/pypi/v/scilpy?logo=pypi&logoColor=white)](https://pypi.org/project/scilpy)
[![PyPI - Downloads](https://static.pepy.tech/badge/scilpy)](https://pypi.org/project/scilpy)

[![Docker container badge](https://img.shields.io/docker/v/scilus/scilus?label=docker&logo=docker&logoColor=white)](https://hub.docker.com/r/scilus/scilpy)

**Scilpy** is the main library supporting research and development at the Sherbrooke Connectivity Imaging Lab
([SCIL]).

**Scilpy** mainly comprises tools and utilities to quickly work with diffusion MRI. Most of the tools are based
on or are wrappers of the [DIPY] library, and most of them will eventually be migrated to [DIPY]. Those tools implement the recommended workflows and parameters used in the lab.

The library is now built for Python 3.12 so be sure to create a virtual environnement for Python 3.12. If this version is not installed on your computer:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.12 python3.12-dev python3.12-venv python3.12-tk
```

:warning: We highly suggest to install uv to speedup scilpy installation: https://docs.astral.sh/uv/getting-started/installation/

:point_up: BUT, if you don't want to use uv, scilpy can still be installed by omitting the uv from all the installation command lines below.

Make sure your pip is up-to-date before trying to install:
```
uv pip install --upgrade pip
```

The library's structure is mostly aligned on that of [DIPY].

We highly encourage to install scilpy in a virtual environnement. Once done and you're in your virtual environnement, the library and scripts can be installed locally by running these commands:

## Install scilpy as a user

```
# If you are using Python3.10 or Python3.11, export this variable before installing
export SETUPTOOLS_USE_DISTUTILS=stdlib

uv pip install scilpy # For the most recent release from PyPi
```

## Install scilpy as a developer

```
# If you are using Python3.10 or Python3.11, export this variable before installing
export SETUPTOOLS_USE_DISTUTILS=stdlib

uv pip install -e . # Install from source code (for development)
```

## EXTRAS

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
